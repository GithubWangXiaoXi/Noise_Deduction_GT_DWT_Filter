import PSO
import matplotlib.pyplot as plt
import numpy as np

dim = 2   # 粒子的维度
size = 20  # 粒子个数
iter_num = 1000 # 迭代次数
x_max = 10
max_vel = 0.5 # 粒子最大速度

pso = PSO.PSO(dim, size, iter_num, x_max, max_vel)
fit_var_list, best_pos = pso.update()
print("最优位置:" + str(best_pos))
print("最优解:" + str(fit_var_list[-1]))
plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list, alpha=0.5)
plt.show()