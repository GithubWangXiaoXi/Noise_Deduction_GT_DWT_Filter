import Game
from Game import Player,GameTheroy
from utils import mydsp,OperaterUtils
import Wavelet,LMS
import pandas as pd
import numpy as np
from MyTest import testGameTheroy

if __name__ == '__main__':
    # ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
    # name = "Block"
    # ys,name = mydsp.HeavySine()
    # ys,name = mydsp.Bumps()
    ys,name = mydsp.Droppler(2000,200,amp=1)

    DWT_set = {'haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'}  # 离散小波基
    LMS_set = {0.00001, 0.00003, 0.00005, 0.00007, 0.0001}  # 学习步长

    game = GameTheroy()  # 开始博弈
    game.initRowAndCol(DWT_set, LMS_set)  # 初始化收益矩阵

    # 收益矩阵表（用DataFrame）
    print(game.payoff_Matrix)
    for basis in DWT_set:
        DWT = testGameTheroy.player1(ys, basis, 20.0)
        for step in LMS_set:
            LMS = testGameTheroy.player2(ys, step, 20.0)
            payoff = list()
            payoff.append(game.DWT_payoff(player1=DWT, player2=LMS))
            payoff.append(game.LMS_payoff(player1=LMS, player2=DWT))
            # print(game.payoff_Matrix[game.row[basis]][game.col[step]])
            game.payoff_Matrix[game.row[basis]][game.col[step]] = payoff

    row = list(DWT_set)
    col = list(LMS_set)
    df = pd.DataFrame(data=game.payoff_Matrix, columns=col, index=row)
    df.to_csv(name + "_PayoffMatrix.csv")
