
#缩放因子，0.8较为严格，0.9较为宽松
scale = 0.9
#允许的最小半径
min_distance = 0.8
#种群大小
population_size = 60
#维持的候选结构队列的最大长度
candidate_size = 40
#化学式
num_atoms = [4,8]
formula_list = ['Si','O']
#初始化时单批次评估的结构数量
init_batch_size = 16
#初始化时最长隔多久也要评估一次
init_timeout = 0.1
#ensure_min_distances的最大位移幅度
max_move = 0.5
#容忍能量，当一个种群中的能量差小于这个的时候，就给予他们相同的适应度
tolerant_energy = 1e-3
#最大cpu进程数
max_gpu = 4
#最大gpu线程数
max_cpu = 4
#粗松弛时的最大力，迭代数
f_max1 = 0.1
max_step1 = 50
#细松弛时的最大力
f_max2 = 0.05
max_step2 = 200


