'''
player1 = DWT
    strategies: 'haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'
    payoff: ​​5×(1-SNR)^2 + 2×(1-MSE)^2
player2 = LMS
    strategies: step = 0.00001,0.00003,0.00005,0.00007,0.0001
    payoff: ​​​​5×(1-MSE)^2 + 2×(1-SNR)^2

Notes:规定两个玩家处理同种信号
'''
import unittest
import Game
from Game import Player,GameTheroy
from utils import mydsp,OperaterUtils
import Wavelet,LMS
import pandas as pd
import numpy as np
from MyTest import testGameTheroy

class MyTest(unittest.TestCase):

    def testPayoffMatrix(self):
        ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
        name = "Block"
        # ys,name = mydsp.HeavySine()
        # ys,name = mydsp.Bumps()
        ys, name = mydsp.Droppler(2000,200)

        DWT_set = {'haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'}  # 离散小波基
        LMS_set = {0.00001, 0.00003, 0.00005, 0.00007, 0.0001}  # 学习步长

        game = GameTheroy()  # 开始博弈
        game.initRowAndCol(DWT_set, LMS_set)  # 初始化收益矩阵

        # 收益矩阵表（用DataFrame）
        print(game.payoff_Matrix)
        for basis in DWT_set:
            DWT = player1(ys, basis, 20.0)
            for step in LMS_set:
                LMS = player2(ys, step, 20.0)
                payoff = list()
                payoff.append(game.DWT_payoff(player1=DWT, player2=LMS))
                payoff.append(game.LMS_payoff(player1=LMS, player2=DWT))
                # print(game.payoff_Matrix[game.row[basis]][game.col[step]])
                game.payoff_Matrix[game.row[basis]][game.col[step]] = payoff

        row = list(DWT_set)
        col = list(LMS_set)
        df = pd.DataFrame(data=game.payoff_Matrix, columns=col, index=row)
        df.to_csv("../payoff/" + name + "_PayoffMatrix.csv")
        return None

    def testStrategy(self):
        df = pd.read_csv("../payoff/Bumps_PayoffMatrix.csv")
        df = df.drop('7e-05',axis=1)
        # print(df.drop(1,axis=0))
        print(df)
        # firstHand = {1e-05, 3e-05, 5e-05, 7e-05, 0.0001}  # 学习步长,先手是LMS
        # # print(df.columns)
        # DWT_set = df[df.columns[0]]
        # for s in firstHand:
        #     scoreDict = {}
        #     for i in range(0,len(DWT_set)):
        #         scoreDict[i] = str(df[str(s)][i]).split(",")[0].replace("[","")
        #     value = max(zip(scoreDict.values(), scoreDict.keys()))
        #     basisIndex = value[1]
        #     print("当LMS步长step = ",str(s),"时, DWT采用",DWT_set[basisIndex],"小波基")

    def testDenoiseWithNashEquilibrium(self):
        '''
        main test func
        :return:
        '''
        # ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
        # name = "Block"
        ys,name = mydsp.HeavySine()

        # ys,name = mydsp.Bumps()
        ys, name = mydsp.Bumps()
        df = pd.read_csv("../payoff/Bumps_PayoffMatrix.csv")

        # print(df[0:1])
        NashEquilibrium = Game.getNashEquilibrium(df)
        print(NashEquilibrium)
        firstHand = dict()  # 小波基,先手是DWT

        DWT_Series = df[df.columns[0]]
        for i in range(0,len(DWT_Series)):
            firstHand[i] = DWT_Series[i]

        LMS_set = df.columns[1:]

        print("当DWT采用",firstHand[NashEquilibrium[0][0]],"小波基","LMS步长step = ",LMS_set[NashEquilibrium[0][1]])

        LMS = player2(ys, float(LMS_set[NashEquilibrium[0][1]]), 2.0)  #
        DWT = player3(LMS.output_ys, str(firstHand[NashEquilibrium[0][0]]), LMS.SNR)  # DWT

    def testCombineDenoise(self):
        '''
        先进行LMS去噪，再进行小波去噪
        :return:
        '''
        ys,name = mydsp.Bumps()

        DWT_set = {'haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'}  # 离散小波基
        LMS_set = {0.00001, 0.00003, 0.00005, 0.00007, 0.0001}  # 学习步长

        LMS = player2(ys,0.00005,2.0) #
        DWT = player3(LMS.output_ys,'rbio1.3',LMS.SNR) #DWT


def getPayoffMatrix(ys,name):
    # ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
    #name = "Block"
    # ys,name = mydsp.HeavySine()
    # ys,name = mydsp.Bumps()

    DWT_set = {'haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'} # 离散小波基
    LMS_set = {0.00001, 0.00003, 0.00005, 0.00007, 0.0001}  # 学习步长

    game = GameTheroy()#开始博弈
    game.initRowAndCol(DWT_set,LMS_set) #初始化收益矩阵

    # 收益矩阵表（用DataFrame）
    print(game.payoff_Matrix)
    for basis in DWT_set:
        DWT = player1(ys, basis,20.0)
        for step in LMS_set:
            LMS = player2(ys, step,20.0)
            payoff = list()
            payoff.append(game.DWT_payoff(player1=DWT,player2=LMS))
            payoff.append(game.LMS_payoff(player1=LMS, player2=DWT))
            # print(game.payoff_Matrix[game.row[basis]][game.col[step]])
            game.payoff_Matrix[game.row[basis]][game.col[step]] = payoff

    row = list(DWT_set)
    col = list(LMS_set)
    df = pd.DataFrame(data=game.payoff_Matrix,columns=col,index=row)
    df.to_csv("../"+ name + "_PayoffMatrix.csv")
    return None

def player1(ys,basis,SNR):

    DWT = Player(name="DWT",basis=basis)

    # SNR = 20.0

    signal = mydsp.Signal(signal_ys=ys)
    signal.addNoise(SNR=SNR)
    # signal = mydsp.HeavySine()
    # signal.plot(title="input signal")

    order = 4  #小波分解阶数
    print(basis)
    print(type(ys))
    # threshold = round(OperaterUtils.heuristicThreshold(ys), 4)  # 采用启发式阈值方法
    # rec_a, rec_d = Wavelet.wavelet(ys=signal.input_ys, basis=basis, order=order,threshold=threshold)
    rec_a, rec_d = Wavelet.wavelet(ys=signal.input_ys, basis=basis, order=order)
    ys_o = [rec_a[i][0] for i in range(len(rec_a))]

    # ts = [i for i in range(len(ys_o))]
    signal.output_ys = ys_o
    title = "denoised singal"
    # signal.plotDenoise(title=title)

    ys_n = signal.output_ys[:len(signal.signal_ys)] - signal.signal_ys  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列
    MSE = OperaterUtils.getMSE(signal.signal_ys, signal.output_ys[:len(signal.signal_ys)])

    # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
    SNR_DWT = OperaterUtils.getSNR(signal.signal_ys, ys_n)
    print("MSE:", MSE)
    print("去噪后的SNR:", SNR_DWT)
    print("SNR提升量:", SNR_DWT - signal.SNR)
    print("SNR提升率:", (SNR_DWT - signal.SNR) / signal.SNR)

    DWT.DWT_payoff(MSE=MSE,SNR_rate=(SNR_DWT - signal.SNR) / signal.SNR)
    DWT.output_ys = ys_o
    print(DWT.payoff)
    return DWT

def player2(ys,step,SNR):

    LMS_object = Player(name="LMS", step=step)

    # SNR = 20.0

    signal = mydsp.Signal(signal_ys=ys)
    signal.addNoise(SNR=SNR)
    # signal = mydsp.HeavySine()
    # signal.plot(title="input signal")

    order = 4
    print(step)
    output_ys, MSE = LMS.LMS(ys=signal.input_ys, dn=signal.signal_ys, a=step)
    signal.output_ys = output_ys
    # signal.plotDenoise(title="LMS")

    ys_n = signal.output_ys[:len(signal.signal_ys)] - signal.signal_ys  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列

    # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
    SNR_LMS = OperaterUtils.getSNR(signal.signal_ys, ys_n)
    print("MSE:", MSE)
    print("去噪后的SNR:", SNR_LMS)
    print("SNR提升量:", SNR_LMS - signal.SNR)
    print("SNR提升率:", (SNR_LMS - signal.SNR) / signal.SNR)

    payoff_func = getattr(LMS_object, LMS_object.payoff_func)
    payoff = payoff_func(MSE=MSE, SNR_rate=(SNR_LMS - signal.SNR) / signal.SNR)

    LMS_object.output_ys = output_ys
    LMS_object.SNR = SNR_LMS
    print(LMS_object.payoff)
    return LMS_object

def player3(ys,basis,SNR):

    DWT = Player(name="DWT",basis=basis)

    signal = mydsp.Signal(signal_ys=ys)
    signal.SNR = SNR
    signal.addNoise(SNR=SNR)
    # signal = mydsp.HeavySine()
    signal.plot(title="input signal")

    order = 4  #小波分解阶数
    print(basis)
    # threshold = round(OperaterUtils.heuristicThreshold(ys), 4)  # 采用启发式阈值方法
    # rec_a, rec_d = Wavelet.wavelet(ys=signal.input_ys, basis=basis, order=order, threshold=threshold)
    rec_a, rec_d = Wavelet.wavelet(ys=signal.input_ys, basis=basis, order=order)
    ys_o = [rec_a[i][0] for i in range(len(rec_a))]

    # ts = [i for i in range(len(ys_o))]
    signal.output_ys = ys_o
    title = "denoised singal"
    signal.plotDenoise(title=title)

    ys_n = np.array(signal.output_ys[:len(signal.signal_ys)]) - np.array(signal.signal_ys)  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列
    MSE = OperaterUtils.getMSE(signal.signal_ys, signal.output_ys[:len(signal.signal_ys)])

    # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
    SNR_DWT = OperaterUtils.getSNR(signal.signal_ys, ys_n)
    print("MSE:", MSE)
    print("去噪后的SNR:", SNR_DWT)
    print("SNR提升量:", SNR_DWT - signal.SNR)
    print("SNR提升率:", (SNR_DWT - signal.SNR) / signal.SNR)

    DWT.DWT_payoff(MSE=MSE,SNR_rate=(SNR_DWT - signal.SNR) / signal.SNR)
    DWT.output_ys = ys_o
    print(DWT.payoff)
    return DWT