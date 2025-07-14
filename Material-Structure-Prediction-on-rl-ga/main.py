import numpy as np
import torch
from torch.cuda.amp import autocast

from chgnet.model.model import CHGNet
from chgnet.model import CHGNetCalculator, CHGNet
from chgnet.data.dataset import StructureData, get_train_val_test_loader

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import BFGS ,FIRE

from concurrent.futures import ProcessPoolExecutor, as_completed ,ThreadPoolExecutor
from multiprocessing import Queue, Event, Process
import threading, queue

import os
import random
import time
from itertools import islice
from functools import partial
from copy import deepcopy

import constant
import crystal
import build

calc = None

class Gene:
    def __init__(self, atom, energy = 1e4, fitness = 0):
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
    #达到容忍能量都要
    if energy_range < constant.tolerant_energy:
        for i in new_population:
            i.fitness = 1
    else:
        #计算归一化能量
        for i in new_population:
            i.fitness = (max_energy - i.energy)/energy_range
    return new_population

def queue_maintain(stop_evt: Event)->None:
    seed = int(time.time())
    while not stop_evt.is_set():
        #阻塞式填充，满了就不会轮询，不会使cpu空转
        queue2.put(build.build_structure(seed))

if __name__ == "__main__":
    #初始化gpu
    device = torch.device("cuda", 0)  # 只指定一次 GPU
    calc = CHGNetCalculator(
        device=device,
        batch_size=constant.init_batch_size,
        sanitize=True,  # 默认就开：wrap 到原胞、剔除过近原子……
    )

    population = []
    waiting_evaluates = []
    #创建一个进程池暴力填满种群
    candidate_group = crystal.find_candidate_groups(constant.num_atoms)
    seeds = [random.randint(0, 10**6) for _ in range(constant.population_size)]
    with ProcessPoolExecutor(max_workers=constant.max_cpu) as cpu_pool:
        #提交,返回一个iterator
        results = [cpu_pool.submit(build.build_structure, candidate_group,s) for s in seeds]
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
        idx1 +=1

    population = evaluate_fitness(population)
    for i in population:
        print(i.atoms,i.energy,i.fitness)

    #同时后台单开一个进程专门用来跑结构生成
    queue2 = queue.Queue(maxsize=constant.candidate_size * 2)
    q = Queue(maxsize=constant.candidate_size)  # 受限容量的缓冲队列
    stop_evt1 = Event()

    #启动保护式进程
    maintain = Process(target=queue_maintain, args=(q, stop_evt1), daemon=True)
    maintain.start()





