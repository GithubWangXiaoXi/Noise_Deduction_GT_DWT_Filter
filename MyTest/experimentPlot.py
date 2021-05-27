from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import Game
import Wavelet,LMS
from utils import mydsp, OperaterUtils
from MyTest import experimentPlot, testGameTheroy

if __name__ == '__main__':

    '''原信号'''
    # experimentPlot.printOriginSignal()

    '''染噪信号'''
    experimentPlot.printSignalWithNoise()

    '''小波分解：不同小波基的去噪效果'''
    # experimentPlot.printDWT()

    '''LMS: 不同步长的去噪效果'''
    # experimentPlot.printLMS()

    '''GT:'''
    # experimentPlot.printNash()

def printOriginSignal():
    fig = plt.figure()

    ys1 = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
    name1 = "Block"
    signal1 = mydsp.Signal(signal_ys=ys1)

    plt.subplot(2, 2, 1)
    plt.title(name1,fontsize=14)
    plt.plot(signal1.ts, signal1.signal_ys)

    ys2,name2 = mydsp.Bumps()
    signal2 = mydsp.Signal(signal_ys=ys2)

    plt.subplot(2, 2, 2)
    plt.title(name2,fontsize=14)
    plt.plot(signal2.ts, signal2.signal_ys)

    ys3, name3 = mydsp.HeavySine()
    signal3 = mydsp.Signal(signal_ys=ys3)

    plt.subplot(2, 2, 3)
    plt.title(name3,fontsize=14)
    plt.plot(signal3.ts, signal3.signal_ys)

    ys4, name4 = mydsp.Droppler(2000,200)
    signal4 = mydsp.Signal(signal_ys=ys4)

    plt.subplot(2, 2, 4)
    plt.title(name4,fontsize=14)
    plt.plot(signal4.ts, signal4.signal_ys)

    plt.suptitle('四种基准信号(x:times(s),y:Amplitude)',fontsize=16)
    plt.show()

def printSignalWithNoise():
    SNR = 10.0
    ys1 = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
    name1 = "Block"
    signal1 = mydsp.Signal(signal_ys=ys1)
    signal1.addNoise(SNR=SNR)

    plt.subplot(2, 2, 1)
    plt.title(name1,fontsize=14)
    plt.plot(signal1.ts, signal1.input_ys)

    ys2, name2 = mydsp.Bumps()
    signal2 = mydsp.Signal(signal_ys=ys2)
    signal2.addNoise(SNR=SNR)

    plt.subplot(2, 2, 2)
    plt.title(name2,fontsize=14)
    plt.plot(signal2.ts, signal2.input_ys)

    ys3, name3 = mydsp.HeavySine()
    signal3 = mydsp.Signal(signal_ys=ys3)
    signal3.addNoise(SNR=SNR)

    plt.subplot(2, 2, 3)
    plt.title(name3,fontsize=14)
    plt.plot(signal3.ts, signal3.input_ys)

    ys4, name4 = mydsp.Droppler(2000, 200)
    signal4 = mydsp.Signal(signal_ys=ys4)
    signal4.addNoise(SNR=SNR)

    plt.subplot(2, 2, 4)
    plt.title(name4,fontsize=14)
    plt.plot(signal4.ts, signal4.input_ys)
    plt.suptitle('AWGN染噪信号(SNR=10, x:times(s), y:Amplitude)',fontsize=16)

    plt.show()

def printDWT():
    plt.figure(figsize=(20, 30))
    DWT_set = {'haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'}  # 离散小波基
    # ys, name = mydsp.Bumps()
    ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
    name = "Block"
    signal = mydsp.Signal(signal_ys=ys)
    signal.addNoise(SNR=2.0)
    i = 1
    order = 4
    for basis in DWT_set:
        plt.subplot(3, 2, i)
        rec_a, rec_d = Wavelet.wavelet(ys=signal.input_ys, basis=basis, order=order)
        ys_o = [rec_a[i][0] for i in range(len(rec_a))]

        # ts = [i for i in range(len(ys_o))]
        signal.output_ys = ys_o
        title = "denoised singal"
        plt.plot(signal.ts, signal.output_ys[0:len(signal.ts)])

        ys_n = signal.output_ys[:len(signal.signal_ys)] - signal.signal_ys  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列
        MSE = OperaterUtils.getMSE(signal.signal_ys, signal.output_ys[:len(signal.signal_ys)])

        # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
        SNR_DWT = OperaterUtils.getSNR(signal.signal_ys, ys_n)
        print("MSE:", MSE)
        print("去噪后的SNR:", SNR_DWT)
        print("SNR提升量:", SNR_DWT - signal.SNR)
        print("SNR提升率:", (SNR_DWT - signal.SNR) / signal.SNR)

        # DWT.DWT_payoff(MSE=MSE, SNR_rate=(SNR_DWT - signal.SNR) / signal.SNR)
        # DWT.output_ys = ys_o
        # print(DWT.payoff)
        plt.title(basis + ":SNR=" + str(round(SNR_DWT, 2)), fontsize=14)

        i = i + 1
    plt.suptitle(name+'信号(SNR=2, x:times(s), y:Amplitude)', fontsize=16)
    plt.legend()
    plt.show()

def printLMS():
    plt.figure(figsize=(20, 30))
    LMS_set = {0.00001, 0.00003, 0.00005, 0.00007, 0.0001}
    ys, name = mydsp.Bumps()
    # ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
    # name = "Block"
    signal = mydsp.Signal(signal_ys=ys)
    signal.addNoise(SNR=2.0)
    i = 1
    for step in LMS_set:
        plt.subplot(3, 2, i)
        output_ys, MSE = LMS.LMS(ys=signal.input_ys, dn=signal.signal_ys, a=step)
        signal.output_ys = output_ys
        # signal.plotDenoise(title="LMS")
        plt.plot(signal.ts, signal.output_ys[0:len(signal.ts)])

        ys_n = signal.output_ys[:len(signal.signal_ys)] - signal.signal_ys  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列

        # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
        SNR_LMS = OperaterUtils.getSNR(signal.signal_ys, ys_n)
        print("MSE:", MSE)
        print("去噪后的SNR:", SNR_LMS)
        print("SNR提升量:", SNR_LMS - signal.SNR)
        print("SNR提升率:", (SNR_LMS - signal.SNR) / signal.SNR)

        title1 = "步长为" + str(float(step)) + ":SNR=" + str(round(SNR_LMS, 2))
        plt.title(title1, fontsize=14)

        i = i + 1
    plt.suptitle('Bumps信号(SNR=2, x:times(s), y:Amplitude)', fontsize=16)
    plt.legend()
    plt.show()

def printNash():

    plt.figure(figsize=(20, 30))
    # ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
    # name = "Block"
    SNR = 2.0
    # ys, name = mydsp.Bumps()
    ys, name = mydsp.HeavySine()
    # ys, name = mydsp.Droppler(2000,200)
    # testGameTheroy.getPayoffMatrix(ys,name)
    signal = mydsp.Signal(signal_ys=ys)

    # plt.subplot(2, 1, 1)
    # plt.title("原始信号", fontsize=16)
    # plt.plot(signal.ts,signal.signal_ys)
    #
    #
    signal.addNoise(SNR = SNR)
    # plt.subplot(2, 1, 2)
    # plt.title("染噪信号", fontsize=16)
    # plt.plot(signal.ts, signal.input_ys)

    df = pd.read_csv("../payoff/" + name + "_PayoffMatrix.csv")

    # print(df[0:1])
    NashEquilibrium = Game.getNashEquilibrium(df)
    print("NashEquilibrium=", NashEquilibrium)
    firstHand = dict()  # 小波基,先手是DWT

    DWT_Series = df[df.columns[0]]
    for i in range(0, len(DWT_Series)):
        firstHand[i] = DWT_Series[i]

    LMS_set = df.columns[1:]

    print("当DWT采用", firstHand[NashEquilibrium[0][0]], "小波基", "LMS步长step = ", LMS_set[NashEquilibrium[0][1]])

    output_ys, MSE = LMS.LMS(ys=signal.input_ys, dn=signal.signal_ys, a=float(LMS_set[NashEquilibrium[0][1]]))
    ys_n = output_ys[:len(signal.signal_ys)] - signal.signal_ys  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列
    # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
    SNR_LMS = OperaterUtils.getSNR(signal.signal_ys, ys_n)
    plt.subplot(2, 1, 1)
    plt.title("步长为"+LMS_set[NashEquilibrium[0][1]] + "的LMS去噪信号,SNR="+str(SNR_LMS), fontsize=16)
    plt.plot(signal.ts, output_ys[0:len(signal.ts)])

    rec_a, rec_d = Wavelet.wavelet(output_ys, basis=firstHand[NashEquilibrium[0][0]], order=4)
    ys_o = [rec_a[i][0] for i in range(len(rec_a))]
    plt.subplot(2, 1, 2)

    ys_n = ys_o[:len(signal.signal_ys)] - signal.signal_ys  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列
    # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
    SNR_DWT = OperaterUtils.getSNR(signal.signal_ys, ys_n)
    plt.title(firstHand[NashEquilibrium[0][0]] + "小波基下的DWT去噪信号,SNR="+str(SNR_DWT), fontsize=16)
    plt.plot(signal.ts, ys_o[0:len(signal.ts)])

    plt.suptitle(name + '信号(SNR=2, x:times(s), y:Amplitude)', fontsize=16)
    plt.legend()
    plt.show()
