import crystal
import constant
import time

# 生成化学式字符串
def get_formula_str():
    return ''.join([f'{elem}{num}' if num > 1 else elem for elem, num in zip(constant.formula_list, constant.num_atoms)])

def build_structure(candidate_groups, seed: int = int(time.time())):
    structure = crystal.generate_crystal_structure(constant.formula_list, constant.num_atoms, candidate_groups, seed=seed)
    if structure is not False:
        return structure
    else:
        # 如果不行就上自己写的，确保生成成功
        formula_str = get_formula_str()
        structure = crystal.generate_random_structure(formula_str)
        structure = crystal.ensure_min_distances(structure)
        return structure