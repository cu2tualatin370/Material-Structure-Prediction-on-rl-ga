from pyxtal import pyxtal
from pyxtal.symmetry import get_wyckoffs
import time
import random
import numpy as np
import constant
import re
from ase import Atoms
from ase.data import covalent_radii

def find_candidate_groups(num_atoms:list):
    groups_candidates = []
    sum_all = sum(num_atoms)
    for i in range(1,231):
        multi_all = {0}
        wyckoff = get_wyckoffs(i)
        mults = [len(w) for w in wyckoff]
        for j in mults:
            multi_all1 = {k + j for k in multi_all}
            multi_all |= multi_all1
            if sum_all in multi_all:
                groups_candidates.append(i)
                break

    print(f"found {len(groups_candidates)} candidates,they are {groups_candidates}")
    return groups_candidates
def generate_crystal_structure(formula_list: list, num_atoms: list,groups: list, seed:int = int(time.time()), per_group:int = 10):
    """
        随机生成给定化学式和空间群的对称晶体结构，并转换为 pymatgen 的 Structure。

        参数
        ----------
        formula_symbols : list of str
            元素符号列表，例如 ['Ba', 'Ti', 'O']
        num_atoms : list of int
            对应每种元素在晶胞内的原子数，例如 [1, 1, 3]
        space_group : int
            国际空间群编号，例如 99（P4mm）
        seed : int, optional
            随机种子，用于结果可复现

        返回
        -------
        pymatgen.core.structure.Structure or None
            如果生成成功，返回对应的 Structure；否则返回 None。
        """
    # 2. 创建 PyXtal 对象
    xtal = pyxtal()
    # 3. 用 from_random() 随机生成符合对称要求的晶体
    #    dim=3      -> 三维结构
    #    group      -> 空间群编号
    #    species    -> 化学元素符号列表
    #    numIons    -> 每种元素的原子数列表
    #    random_state -> 随机数种子
    done = False
    i = random.randint(0,len(groups)-1)
    for group in groups:
        i += 1
        for k in range(per_group):
            xtal.from_random(
                dim=3,
                group = group,
                species=formula_list,
                numIons=num_atoms,
                random_state=seed + i
                )
            if xtal.valid :
                structure = xtal.to_ase()
                done = True
                break
        if done:
            break

    # 4. 检查生成是否成功
    #    xtal.valid == True 时，表示找到了合法的 Wyckoff 布局并生成了结构
    if not xtal.valid:
        return False
    else:
        return structure

def generate_random_structure(formula: str,
                              cell_length: float = None,
                              ) -> Atoms:
    print("crystal failure")
    """
    基于已知化学式生成纯随机结构，保证任意两原子距离 >= min_dist。

    参数
    ----
    formula : str
        化学式，如 "Al2O3" 或 "Fe3 O4"（不支持括号嵌套）。
    cell_length : float 可选
        立方晶胞边长 (Å)。若为 None，则根据经验体积自动估算：
        约 10 Å³/原子；
    min_dist : float 可选
        最小原子间距阈值 (Å)。若为 None，则使用 constant.min_distanceA。

    返回
    ----
    Atoms
        一个在立方晶胞内随机构建且满足最小距离的 ASE Atoms 对象。
    """
    # 1. 最小距离

    # 2. 解析化学式
    comps = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    symbols = []
    for sym, cnt in comps:
        n = int(cnt) if cnt else 1
        symbols += [sym] * n
    N = len(symbols)

    # 3. 晶胞体积估算
    if cell_length is None:
        # 经验：10 Å³/原子
        vol = 10.0 * N
        cell_length = vol ** (1/3)

    # 4. 随机放置并检查

    pos = np.random.rand(N, 3) * cell_length
    atoms = Atoms(symbols=symbols,
                    positions=pos,
                    cell=[cell_length]*3,
                    pbc=True)
    return atoms


def has_close_atoms_dynamic(atoms: Atoms) -> bool:
    """
    判断任意两原子是否过近（distance < rcov_i + rcov_j - tol），
    过近则立即返回 True。
    """
    #获取两两原子之间的距离
    dm = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dm, np.inf)

    #取出每个原子的公家把半径构建两两原子之间的阈值半径矩阵
    z = atoms.get_atomic_numbers()
    r = covalent_radii[z]
    cutoff = (r[:, None] + r[None, :]) * constant.scale
    return bool(np.any(dm < cutoff))

def ensure_min_distances(atoms:Atoms) -> Atoms:
    """输入一个pymatgen结构，确保原子之间的距离不小于最小距离，同时做出改变"""
    new_atoms = atoms.copy()
    pos = new_atoms.get_positions()  # shape (N,3)
    #计算出位移向量，运用广播减法，得到一个NN3的阵
    vector_distance = pos[:,None,:] - pos[None,:,:]
    #运用爱因斯坦函数，批量计算平方距离
    distance_2 = np.einsum("ijk,ijk -> ij",vector_distance,vector_distance)
    #避免自己和自己重叠使用.fill_diagonal填充对角线
    np.fill_diagonal(distance_2,1e6)
    #找到所有过近的原子对,注意这里返回的是一个布尔矩阵，以原子对为ij的索引为储存的值
    too_close = np.argwhere(distance_2 < constant.min_distance ** 2)
    if too_close.size > 0:
        idx = np.unique(too_close[:, 0])
        disp = (np.random.rand(len(idx), 3) - 0.5) * 2 * constant.max_move
        pos[idx] += disp
        new_atoms.set_positions(pos)
    return new_atoms
