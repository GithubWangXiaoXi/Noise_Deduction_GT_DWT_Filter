import Game
from utils import mydsp, OperaterUtils
from MyTest import testGameTheroy
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # ys = np.array(pd.read_csv("dataset/Blocks.csv")['ys'])
    # name = "Block"
    # ys, name = mydsp.Bumps()
    # ys, name = mydsp.HeavySine()
    ys, name = mydsp.Droppler(2000,200)
    # testGameTheroy.getPayoffMatrix(ys,name)

    df = pd.read_csv("./payoff/" + name+"_PayoffMatrix.csv")

    # print(df[0:1])
    NashEquilibriumIndex,NashEquilibriumValT = Game.getNashEquilibrium(df)
    print("NashEquilibriumIndex=",NashEquilibriumIndex)
    print("NashEquilibriumVal=", NashEquilibriumValT)

    NashEquilibriumVal = list()
    token = str(NashEquilibriumValT[0]).split(",")
    NashEquilibriumVal.append(float(token[0].replace("[","")))
    NashEquilibriumVal.append(float(token[1].replace("]", "")))
    N = NashEquilibriumVal[0] + NashEquilibriumVal[1]
    NashEquilibriumVal[0] = NashEquilibriumVal[0] / N
    NashEquilibriumVal[1] = NashEquilibriumVal[1] / N

    firstHand = dict()  # 小波基,先手是DWT

    DWT_Series = df[df.columns[0]]
    for i in range(0, len(DWT_Series)):
        firstHand[i] = DWT_Series[i]

    LMS_set = df.columns[1:]

    print("当DWT采用", firstHand[NashEquilibriumIndex[0][0]], "小波基", "LMS步长step = ", LMS_set[NashEquilibriumIndex[0][1]])

    SNR = 10.0
    LMS = testGameTheroy.player2(ys, float(LMS_set[NashEquilibriumIndex[0][1]]), SNR)  #LMS
    DWT = testGameTheroy.player3(ys, str(firstHand[NashEquilibriumIndex[0][0]]), SNR)  # DWT

    '''LMS和DWT算法的融合：双方博弈值作为两种算法的权重，得到的SNR一般在两种算法的SNR之间'''
    length =len(LMS.output_ys) if(len(DWT.output_ys) > len(LMS.output_ys)) else len(DWT.output_ys)
    a = ([NashEquilibriumVal[0]] * length)
    b = ([NashEquilibriumVal[1]] * length)
    a = np.array(a)
    b = np.array(b)
    output_ys = DWT.output_ys[:length] * a +  LMS.output_ys[:length] * b

    ys_n = output_ys - ys
    SNR_mix = OperaterUtils.getSNR(ys, ys_n)
    MSE = OperaterUtils.getMSE(ys,output_ys)
    print("去噪后的MSE:", MSE)
    print("去噪后的SNR:", SNR_mix)
    ts = [i for i in range(len(output_ys))]
    plt.plot(ts,output_ys)
    plt.show()