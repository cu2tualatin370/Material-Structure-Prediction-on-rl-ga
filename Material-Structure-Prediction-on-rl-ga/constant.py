
#化学式
formula_list = ['Sr', 'Fe', 'Mo', 'O']
num_atoms = [2,    1,    1,    6]


#种群大小，该数最好同时是5和16的倍数
population_size = 64
#维持的候选结构队列的最大长度
candidate_size = 40
#锦标赛法的待选个体数量,该数必须是2的倍数,同时注意不要超出待选择的个体数
rank_candidates = 32
#锦标赛法进行几轮，rank_candidates除该数不能小于2
turn_max = 2

#缩放因子，0.8较为严格，0.9较为宽松
scale = 0.9
#允许的最小半径，粗筛
min_distance = 0.8
#ensure_min_distances的最大位移幅度
max_move = 0.5

#初始化时单批次评估的结构数量
init_batch_size = 16
#初始化时最长隔多久也要评估一次
init_timeout = 0.1

#容忍能量，当一个种群中的能量差小于这个的时候，就给予他们相同的适应度
tolerant_energy = 1e-3
#初始能量的大小
init_energy = 1e4

#最大cpu进程数
max_gpu = 4
#最大gpu线程数
max_cpu = 4

#粗松弛时的最大力，迭代数
f_max1 = 0.1
max_step1 = 50
#细松弛时的最大力
f_max2 = 0.05
max_step2 = 100

#每代直接继承，不进行杂交的个体数
num_elitism = 3
#每代中进行交叉变异的比例
crossover_rate = 0.8
#是否将直接抛弃的个体也参与杂交
low_fitness_switch = False

#晶格矩阵交叉时的混合参数
eta = 2.0

#最大迭代数
max_generations = 20
#滑动窗口的大小（代）
windows = 3
#收敛能量的大小
slope_energy = 1e-3 * sum(num_atoms)

#变异率
mutate_rate = 0.1
#体积范围
volume_range = [0.9, 1.1]

