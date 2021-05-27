import unittest
from utils import SignalNoiseUtils,OperaterUtils
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import Wavelet


# 参考<https://jingyan.baidu.com/article/c1a3101e6b8062de656deb0b.html>
class MyTest(unittest.TestCase):

    '''函数前缀为test，则该函数为测试方法'''
    def testRayleighFChannel(self):
        sigmas = {0.5, 1, 2, 3, 4}
        for sigma in sigmas:
            x = np.linspace(0, 10, 100)
            y = SignalNoiseUtils.RayleighFChannel(x, sigma)

            label = "sigma = " + str(sigma)
            plt.plot(x, y, label=label)
            plt.xlim(left=0, right=10)
            plt.ylim(bottom=0, top=1.2)

        plt.legend()
        plt.show()

    def testBumps(self):
        ts,ys = SignalNoiseUtils.Bumps(55, 1, 0.5)
        plt.plot(ts,ys)
        plt.show()

    def testHeavySine(self):
        ts,ys = SignalNoiseUtils.HeavySine(2)
        plt.plot(ts, ys)
        plt.show()

    def testBlock(self):
        ys = pd.read_csv("../dataset/Blocks.csv")
        num = len(ys)
        #参考<https://blog.csdn.net/weixin_33841503/article/details/94723928>
        ts = [i for i in range(num)]
        print(ts)
        plt.plot(ts,ys)
        plt.show()

    def testBlockWithNoise(self):
        SNR = 4.0
        df_ys = pd.read_csv("../dataset/Blocks.csv") #pd.readcsv() 读出来的ys是个二维数组(DataFrame)，需要转化成一维的(Series)
        ys = np.array(df_ys['ys'])
        # ys = df_ys
        ts = [i for i in range(len(ys))]

        avgAmp = np.mean(ys)  #信号的振幅
        noiseAmp = avgAmp / SNR #噪声的振幅

        ys,SNR = SignalNoiseUtils.BlockWithNoise(ys, SNR)
        print("SNR:",SNR-4.5)
        plt.plot(ts,ys)
        plt.show()

    def testPrintNum(self):
        for i in range(0,80):
            print(-0.8)

    def testBlockWithWavelet(self):

        SNR = 20.0
        df_ys = pd.read_csv("../dataset/Blocks.csv")  # pd.readcsv() 读出来的ys是个二维数组(DataFrame)，需要转化成一维的(Series)
        ys = np.array(df_ys['ys'])
        ts = [i for i in range(len(ys))]

        avgAmp = np.mean(ys)  # 信号的振幅
        noiseAmp = avgAmp / SNR  # 噪声的振幅

        ys_input,SNR = SignalNoiseUtils.BlockWithNoise(ys_s=ys,SNR=SNR)
        # ys_input, SNR = SignalNoiseUtils.BumpsWithNoise(SNR=SNR)
        plt.title = "Block signal with AWGN"
        # ts = [i for i in range(len(ys_input))]
        plt.plot(ts, ys_input)
        plt.show()
        print("输入信号的SNR",SNR - 4.5)

        order = 4
        wavelist=  ['haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'] #离散小波基
        for basis in wavelist:
            print(basis)
            rec_a,rec_d = Wavelet.wavelet(ys=ys_input, basis=basis,order=order)
            ys_o = [rec_a[i][0] for i in range(len(rec_a))]

            ts = [i for i in range(len(ys_o))]
            plt.title = "denoised block singal"
            plt.plot(ts,ys_o)
            plt.show()

            # window = np.ones(10)
            # window /= sum(window)
            # ys = OperaterUtils.smooth(ys, window)  # 平滑降噪后的信号 平滑效果不是很好
            # ts = [i for i in range(len(ys))]
            # plt.title = "denoised block singal with smooth"
            # plt.plot(ts, ys)
            # plt.show()

            ys_n = [rec_d[i][0] for i in range(len(rec_d))]
            ts = [i for i in range(len(ys_n))]
            plt.title = "noise singal"
            plt.plot(ts, ys_n)
            plt.show()

            print("去噪后的SNR:", OperaterUtils.getSNR(ys_o,ys_n)-4.5)

            # print(rec_a,"-----",len(rec_a))
            # print(rec_d, "-----", len(rec_d))

    def testBumpsWithWavelet(self):

        SNR = 20.0

        ys_input, SNR = SignalNoiseUtils.BumpsWithNoise(SNR=SNR)
        plt.title = "Block signal with AWGN"
        ts = [i for i in range(len(ys_input))]
        plt.plot(ts, ys_input)
        plt.show()
        print("输入信号的SNR",SNR - 4.5)

        order = 4
        wavelist=  ['haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'] #离散小波基
        for basis in wavelist:
            print(basis)
            rec_a,rec_d = Wavelet.wavelet(ys=ys_input, basis=basis,order=order)
            ys_o = [rec_a[i][0] for i in range(len(rec_a))]

            ts = [i for i in range(len(ys_o))]
            plt.title = "denoised block singal"
            plt.plot(ts,ys_o)
            plt.show()

            # window = np.ones(10)
            # window /= sum(window)
            # ys = OperaterUtils.smooth(ys, window)  # 平滑降噪后的信号 平滑效果不是很好
            # ts = [i for i in range(len(ys))]
            # plt.title = "denoised block singal with smooth"
            # plt.plot(ts, ys)
            # plt.show()

            ys_n = [rec_d[i][0] for i in range(len(rec_d))]
            ts = [i for i in range(len(ys_n))]
            plt.title = "noise singal"
            plt.plot(ts, ys_n)
            plt.show()

            print("去噪后的SNR:", OperaterUtils.getSNR(ys_o,ys_n)-4.5)

            # print(rec_a,"-----",len(rec_a))
            # print(rec_d, "-----", len(rec_d))

    def testHeavySineWithWavelet(self):

        SNR = 20.0

        ys_input, SNR = SignalNoiseUtils.HeavySineWithNoise(SNR=SNR)
        plt.title = "Block signal with AWGN"
        ts = [i for i in range(len(ys_input))]
        plt.plot(ts, ys_input)
        plt.show()
        print("输入信号的SNR",SNR - 4.5)

        order = 4
        wavelist=  ['haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'] #离散小波基
        for basis in wavelist:
            print(basis)
            rec_a,rec_d = Wavelet.wavelet(ys=ys_input, basis=basis,order=order)
            ys_o = [rec_a[i][0] for i in range(len(rec_a))]

            ts = [i for i in range(len(ys_o))]
            plt.title = "denoised block singal"
            plt.plot(ts,ys_o)
            plt.show()

            # window = np.ones(10)
            # window /= sum(window)
            # ys = OperaterUtils.smooth(ys, window)  # 平滑降噪后的信号 平滑效果不是很好
            # ts = [i for i in range(len(ys))]
            # plt.title = "denoised block singal with smooth"
            # plt.plot(ts, ys)
            # plt.show()

            ys_n = [rec_d[i][0] for i in range(len(rec_d))]
            ts = [i for i in range(len(ys_n))]
            plt.title = "noise singal"
            plt.plot(ts, ys_n)
            plt.show()

            print("去噪后的SNR:", OperaterUtils.getSNR(ys_o,ys_n)-4.5)

            # print(rec_a,"-----",len(rec_a))
            # print(rec_d, "-----", len(rec_d))