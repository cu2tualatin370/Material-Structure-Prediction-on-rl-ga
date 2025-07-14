import crystal
import constant
import time


def build_structure(candidate_groups,seed:int = time.time(),):
    structure = crystal.generate_crystal_structure(constant.formula_list,constant.num_atoms,candidate_groups,seed=seed)
    if structure is not False:
        return structure
    else:
        #如果不行就上自己写的，确保生成成功
        structure = crystal.generate_random_structure(constant.formula_list,constant.num_atoms)
        crystal.ensure_min_distances(structure)
        return structure
