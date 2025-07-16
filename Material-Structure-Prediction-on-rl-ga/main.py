import numpy as np
import torch

from chgnet.model import CHGNetCalculator

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Lattice, Species, Element
from pymatgen.io.cif import CifWriter

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue, Event, Process

import os
import random
import time
from itertools import islice
from collections import Counter
from crystal import ensure_min_distances
import constant
import crystal
import build

calc = None

class Gene:
    def __init__(self, atom, energy = constant.init_energy, fitness = 0):
        self.atoms = atom
        self.energy = energy
        self.fitness = fitness

def evaluate_gpu_batch(atom_list: list) -> list:
    """批量评估能量（CHGNet 自动建图）"""
    energies = []
    forces_all = []
    for at in atom_list:
        at.calc = calc  # ASE 标准做法
        e = at.get_potential_energy() # calc.calculate() 在内部被调用
        f = at.get_forces()
        energies.append(float(e))
        forces_all.append(f.copy())
    return [Gene(a, e, None) for a, e in zip(atom_list, energies)]

def structure_upgrade(pop: list,
                      fmax=constant.f_max1,
                      n_step=constant.max_step1):
    """
    对 Gene 列表做粗松弛：一次批预测能量/力 → 每个结构走 1 步 FIRE。
    pop 会被原地更新并返回。
    """
    if not pop:
        return pop

    # 为每个 Gene 配一个 FIRE 优化器（禁止写日志文件）
    opts = [FIRE(g.atoms, dt=0.1, maxmove=0.2) for g in pop]

    active = list(range(len(pop)))            # 仍在跑的索引
    for _ in range(n_step):
        if not active:
            break

        # ---- 1) GPU 批预测 ----
        atoms_batch = [pop[i].atoms for i in active]
        e_batch, f_batch = [], []
        for at in atoms_batch:
            at.calc = calc  # ASE 标准做法
            e = at.get_potential_energy()  # calc.calculate() 在内部被调用
            f = at.get_forces()
            e_batch.append(float(e))
            f_batch.append(f.copy())
        f_batch = np.concatenate(f_batch, axis=0)  # shape (Σnatoms, 3)

        # ---- 2) 写回 + FIRE 单步 ----
        idx = 0
        next_active = []
        for k, gid in enumerate(active):
            atoms = pop[gid].atoms
            n = len(atoms)
            forces = f_batch[idx:idx+n]; idx += n

            atoms.calc = SinglePointCalculator(atoms,
                                               energy=float(e_batch[k]),
                                               forces=forces)
            opts[gid].step()

            if np.abs(forces).max() > fmax:          # 还没收敛
                next_active.append(gid)
            else:                                    # 收敛
                pop[gid].energy = float(e_batch[k])

        active = next_active
    for g in pop:
        g.atoms.calc = calc
        g.energy = float(g.atoms.get_potential_energy())

    return pop

def chunked(iterable):
    """按 size 大小切片生成小批次"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, constant.init_batch_size))
        if not batch:
            break
        yield batch

def evaluate_fitness(structure_list: list) -> list:
    new_population = sorted(structure_list, key=lambda g: g.energy)
    min_energy = new_population[0].energy
    max_energy = new_population[-1].energy
    energy_range = max_energy - min_energy
    # 达到容忍能量都要
    if energy_range < constant.tolerant_energy:
        for i in new_population:
            i.fitness = 1
    else:
        # 计算归一化能量
        for i in new_population:
            i.fitness = (max_energy - i.energy)/energy_range
    return new_population

def queue_maintain(q: Queue, candidate_groups: list)->None:
    seed = int(time.time())
    while True:
        # 阻塞式填充，满了就不会轮询，不会使cpu空转
        item = build.build_structure(candidate_groups,seed)
        item = Gene(item)
        q.put(item)

def select_parents(pop: list,) -> list:
    # 使用锦标赛法筛选亲代个体
    # 备选个体的随机抽样
    candidate_parents = random.sample(pop, constant.rank_candidates)

    for k in range(constant.turn_max):
        i, j = 0, 1
        next_turn = []
        # 备选个体的逐一锦标赛顺序
        tournament_rank = random.sample(range(int(constant.rank_candidates/(k + 1))),
                                        int(constant.rank_candidates/(k + 1)))
        while (i and j) <= len(tournament_rank):
            if candidate_parents[tournament_rank[i]].fitness \
                < candidate_parents[tournament_rank[j]].fitness:
                next_turn.append(candidate_parents[tournament_rank[j]])
            else:
                next_turn.append(candidate_parents[tournament_rank[i]])
            i += 2
            j += 2
        candidate_parents = next_turn
    return candidate_parents

def lattice_crossover(lat_a: Lattice, lat_b: Lattice, eta: float = 1.0) -> Lattice:
    """
    对两个 pymatgen Lattice 做 SBX 交叉
    步骤:
      1. QR 分解取 R（上三角）参数化
      2. 对 R 的上三角元素逐一做 SBX
      3. 重建回晶格矩阵并返回 Lattice
      4. 若数值不稳定则退回父本 A
    """
    mat_a, mat_b = lat_a.matrix, lat_b.matrix
    tri_a = np.triu(np.linalg.qr(mat_a)[1])
    tri_b = np.triu(np.linalg.qr(mat_b)[1])
    tri_c = np.zeros_like(tri_a)
    for i in range(3):
        for j in range(i, 3):
            u = random.random()
            beta1 = (2 * u) ** (1 / (eta + 1))
            beta2 = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            tri_c[i, j] = 0.5 * ((1 + beta1) * tri_a[i, j] + (1 - beta1) * tri_b[i, j])
    try:
        # 由 R ≈ Qᵀ·A 重构 A
        mat_c = np.linalg.inv(np.linalg.qr(tri_c.T)[0]).T @ tri_c
        return Lattice(mat_c)
    except np.linalg.LinAlgError:
        return Lattice(mat_a)

def repair_structure_by_ratio(
    struct: Structure,
    random_seed: int | None = None
) -> Structure:
    """
    基于给定化学计量比和总原子数，修正 pymatgen Structure 的元素数目。

    参数:
      struct: 待修复的 pymatgen Structure，会被浅复制，不修改原输入
      ratio_map: 化学计量比映射，如 {"C":2, "O":1}
      total_atoms: 目标总原子数
      random_seed: 随机种子（可选），保证删除/添加操作可重现

    返回:
      新的 Structure 对象，元素计量与 ratio_map 和 total_atoms 对应。
    """
    # 1. 拷贝结构，避免修改原始 struct
    s = struct.copy()
    # 算出总原子数
    total_atoms = sum(constant.num_atoms)

    # 2. 计算各元素目标数目（先按比例，再四舍五入）
    target_map = {
        elem: count
        for elem, count in zip(constant.formula_list, constant.num_atoms)
    }

    # 3. 删除多余原子
    if random_seed is not None:
        np.random.seed(random_seed)
    curr_counts = Counter([site.specie.symbol for site in s.sites])
    to_remove = []
    for elem, tgt in target_map.items():
        over = curr_counts.get(elem, 0) - tgt
        if over > 0:
            # 找出所有该元素的索引，随机选 over 个删除
            idxs = [i for i, site in enumerate(s.sites) if site.specie.symbol == elem]
            rm = list(np.random.choice(idxs, size=over, replace=False))
            to_remove.extend(rm)
            curr_counts[elem] -= over
    if to_remove:
        # 必须从大到小删除以保持索引正确
        s.remove_sites(sorted(to_remove, reverse=True))

    # 4. 添加缺失原子
    curr_counts = Counter([site.specie.symbol for site in s.sites])
    for elem, tgt in target_map.items():
        deficit = tgt - curr_counts.get(elem, 0)
        for _ in range(deficit):
            # 在分数坐标系内随机生成位置
            frac = np.random.rand(3)
            # append 接受 Species 或元素符号
            s.append(Species(elem), frac, coords_are_cartesian=False)

    return s

def cross_over(parents: list, num_crossover) -> list:
    print("crossover start")
    j = 0
    for i in parents:
        parents[j] = AseAtomsAdaptor.get_structure(i.atoms)
        j += 1

    # 主循环，对所有的父本都进行交叉
    new_population = []
    for i in range(num_crossover):
        # 随机找出亲代对
        parent_part = list(random.sample(parents, 2))

        # 做晶格交叉
        lattice = lattice_crossover(parent_part[0].lattice, parents[1].lattice)

        # 获取随机切割平面
        n_vec = np.random.normal(size=3)
        n_vec /= np.linalg.norm(n_vec)
        cut_off_distance = random.random()

        # 构造子代
        child_species = []
        child_frac = []
        for j in parent_part[0]:
            if j.frac_coords.dot(n_vec) < cut_off_distance:
                child_species.append(j.species)
                child_frac.append(j.frac_coords)
        for j in parent_part[1]:
            if j.frac_coords.dot(n_vec) >= cut_off_distance:
                child_species.append(j.species)
                child_frac.append(j.frac_coords)
        # 打包 + 修复
        child = repair_structure_by_ratio(Structure(
            lattice=lattice,
            coords=child_frac,
            species=child_species,
        ))

        child = AseAtomsAdaptor.get_atoms(child)
        child = Gene(child)
        new_population.append(child)
    return new_population

def mutate(pop: list) -> list:
    mutated_pop = []
    for gene in pop:
        structure = AseAtomsAdaptor.get_structure(gene.atoms)
        mutated_structure = mutate_structure(structure)
        mutated_atoms = AseAtomsAdaptor.get_atoms(mutated_structure)
        mutated_gene = Gene(mutated_atoms, gene.energy, gene.fitness)
        mutated_pop.append(mutated_gene)
    return mutated_pop

def evolve(pop: list) -> list:
    print("evolve start")
    new_population = []
    length = len(pop)
    elitisim = pop[0: constant.num_elitism]
    new_population.extend(elitisim)
    num_crossover = int(constant.crossover_rate * length)

    if constant.low_fitness_switch:
        waiting_crossover = pop[constant.num_elitism:]
    else:
        waiting_crossover = pop[constant.num_elitism:num_crossover]
    waiting_crossover = select_parents(waiting_crossover)
    cross_over_done = cross_over(waiting_crossover, num_crossover)

    mutate_done = mutate(cross_over_done)
    new_population.extend(mutate_done)

    # 启动生产消费队列，从之前保持的线程中拿取足量的生成的结构
    if len(new_population) < length:
        num_random = length - len(new_population)
        for i in range(num_random):
            item = queue2.get()
            new_population.append(item)
    return new_population

# 在转换后将传入的pymatgen Structure进行变异操作
def mutate_structure(structure: Structure) -> Structure:
    # 结构变异操作
    mutation_rate = constant.mutate_rate
    if random.random() < mutation_rate:
        mutation_type = random.choice([
            'lattice_strain',
            'atom_displacement',
            'volume_change',
            'lattice_shear'
        ])

        new_structure = structure.copy()

        if mutation_type == 'lattice_strain':
            # 晶格应变变异 - 对晶格矩阵施加小的随机应变
            lattice_matrix = new_structure.lattice.matrix
            new_lattice_matrix = []
            for row in lattice_matrix:
                new_row = []
                for value in row:
                    new_value = value * (1 + random.uniform(-0.1, 0.1))
                    new_row.append(new_value)
                new_lattice_matrix.append(new_row)
            new_structure.lattice = new_structure.lattice.__class__(new_lattice_matrix)

        elif mutation_type == 'atom_displacement':
            # 原子位移变异 - 随机选择一些原子并对其位置进行小的随机位移
            for site in new_structure.sites:
                if random.random() < 0.3:  # 30%的原子会被位移
                    displacement = [random.uniform(-0.5, 0.5) for _ in range(3)]
                    site.frac_coords = [coord + disp for coord, disp in zip(site.frac_coords, displacement)]
                    site.frac_coords = [coord % 1 for coord in site.frac_coords]  # 确保分数坐标在 [0, 1) 范围内

        elif mutation_type == 'volume_change':
            # 体积变化变异 - 按比例缩放晶格和原子坐标
            volume_factor = random.uniform(constant.volume_range[0], constant.volume_range[1]) ** (1/3)
            lattice_matrix = new_structure.lattice.matrix
            new_lattice_matrix = [[value * volume_factor for value in row] for row in lattice_matrix]
            new_structure.lattice = new_structure.lattice.__class__(new_lattice_matrix)
            for site in new_structure.sites:
                site.frac_coords = [coord * volume_factor for coord in site.frac_coords]

        elif mutation_type == 'lattice_shear':
            # 晶格剪切变异 - 对晶格施加随机剪切变换
            shear_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            # 随机选择剪切方向
            i1, i2 = random.sample(range(3), 2)
            shear_matrix[i1][i2] = random.uniform(-0.2, 0.2)  # 剪切因子

            lattice_matrix = new_structure.lattice.matrix
            new_lattice_matrix = []
            for row in lattice_matrix:
                new_row = []
                for i in range(3):
                    new_value = sum([row[j] * shear_matrix[j][i] for j in range(3)])
                    new_row.append(new_value)
                new_lattice_matrix.append(new_row)
            new_structure.lattice = new_structure.lattice.__class__(new_lattice_matrix)

            for site in new_structure.sites:
                new_coords = []
                for i in range(3):
                    new_coord = sum([site.frac_coords[j] * shear_matrix[j][i] for j in range(3)])
                    new_coords.append(new_coord)
                site.frac_coords = new_coords

        # 确保原子间最小距离，避免不合理的结构
        atoms = AseAtomsAdaptor.get_atoms(new_structure)
        atoms = ensure_min_distances(atoms)
        new_structure = AseAtomsAdaptor.get_structure(atoms)

        return new_structure

    return structure

if __name__ == "__main__":
    # 初始化gpu
    device = torch.device("cuda", 0)  # 只指定一次 GPU
    calc = CHGNetCalculator(
        device=device,
        batch_size=constant.init_batch_size,
        sanitize=True,  # 默认就开：wrap 到原胞、剔除过近原子……
    )

    population = []
    waiting_evaluates = []
    # 创建一个进程池暴力填满种群
    candidate_group = crystal.find_candidate_groups(constant.num_atoms)
    seeds = [random.randint(0, 10**6) for _ in range(constant.population_size)]
    with ProcessPoolExecutor(max_workers=constant.max_cpu) as cpu_pool:
        # 提交,返回一个iterator
        results = [cpu_pool.submit(build.build_structure, candidate_group, s) for s in seeds]
        for fut in as_completed(results):
            try:
                waiting_evaluates.append(fut.result())
            except Exception as e:
                pass

    # 实例化
    for i in waiting_evaluates:
        i = Gene(i,)
        population.append(i)

    # 启动gpu，评估适应度,松弛结构
    original = population[:]
    idx1 = 0
    for batch_in in chunked(original):
        batch_out = structure_upgrade(batch_in)
        idx2 = 0
        for j in batch_out:
            population[idx2 + idx1] = j
            idx2 += 1
        idx1 += 1

    population = evaluate_fitness(population)
    for i in population:
        print(i.atoms, i.energy, i.fitness)

    # 同时后台单开一个进程专门用来跑结构生成
    queue2 = Queue(maxsize=constant.candidate_size * 2)

    # 启动保护式进程
    maintain = Process(target=queue_maintain, args=(queue2,candidate_group), daemon=True)
    maintain.start()

    best_e_history = []
    for i in range(constant.max_generations):
        population = evolve(population)
        # 启动gpu，评估适应度,松弛结构
        original = population[:]
        idx1 = 0
        for batch_in in chunked(original):
            batch_out = structure_upgrade(batch_in, constant.f_max2, constant.max_step2)
            idx2 = 0
            for j in batch_out:
                population[idx2 + idx1] = j
                idx2 += 1
            idx1 += 1

        population = evaluate_fitness(population)

        best_individual = population[0]
        best_energy = best_individual.energy
        best_e_history.append(best_energy)

        for g in population:
            print(g.atoms, g.energy, g.fitness)

        print(f"这是第{i + 1}代,最佳个体能量为{best_energy}，它的结构是{best_individual.atoms}")

        if len(best_e_history) >= constant.windows:
            recent = np.array(best_e_history[-constant.windows:])  # 最近 window 代能量
            gens = np.arange(constant.windows)  # 0,1,2,…,window-1
            slope = np.polyfit(gens, recent, 1)[0]  # 线性拟合斜率

            if abs(slope) < constant.slope_energy:
                print(f"[Gen {i + 1}] 收敛：最近 {constant.windows} 代斜率 ≈ {slope:.2e} (< {constant.slope_energy})")
                break  # 退出进化循环

    # 转换为 pymatgen 的 Structure 对象
    best_structure = AseAtomsAdaptor.get_structure(best_individual.atoms)
    #获取化学式
    formula_str = build.get_formula_str()
    # 生成 CIF 文件名
    cif_dir = r"D:\ra2ol\code\Material-Structure-Prediction-on-rl-ga\best_individual"
    cif_path = os.path.join(cif_dir, f"{formula_str}.cif")
    # 使用 CifWriter 保存为 CIF 文件
    writer = CifWriter(best_structure)
    writer.write_file(cif_path)
    print(f"最优个体的 CIF 文件已保存为 {cif_path}")