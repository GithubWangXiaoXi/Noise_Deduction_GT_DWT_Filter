import unittest
import pandas as pd
from coreKits import thinkdsp
import LMS
from utils import mydsp, OperaterUtils
import numpy as np
import Wavelet
from matplotlib import pyplot as plt

class MyTest(unittest.TestCase):

    def testBlockSingal(self):
        df_ys = pd.read_csv("../dataset/Blocks.csv")
        ys = np.array(df_ys['ys'])
        block = mydsp.Signal(signal_ys=ys)
        block.plot(title="原始信号")

        block.addNoise(SNR=20)
        block.plot(title="含噪信号")

    def testHeavySineSingal(self):
        ys = mydsp.HeavySine()
        sine = mydsp.Signal(signal_ys=ys)
        sine.plot(title="原始信号")

        sine.addNoise(SNR=20)
        sine.plot(title="含噪信号")

    def testBumpsSingal(self):
        ys = mydsp.Bumps()
        sine = mydsp.Signal(signal_ys=ys)
        sine.plot(title="原始信号")

        sine.addNoise(SNR=20)
        sine.plot(title="含噪信号")

    def testDropplerSingal(self):
        startFreq,endFreq = 10000,200
        ys,name = mydsp.Droppler(startFreq=startFreq,endFreq=endFreq,amp=1)

        droppler = mydsp.Signal(signal_ys=ys)
        droppler.plot(title="原始信号")

        droppler.addNoise(SNR=20)
        droppler.plot(title="含噪信号")

    def testWavelet(self):
        SNR = 2.0

        # ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
        ys,name = mydsp.HeavySine()
        # ys,name = mydsp.Bumps()
        signal = mydsp.Signal(signal_ys=ys)
        signal.addNoise(SNR=SNR)
        # signal = mydsp.HeavySine()
        signal.plot(title="input signal")

        order = 4
        wavelist = ['haar', 'db1', 'sym2', 'coif1', 'bior1.3', 'rbio1.3']  # 离散小波基
        for basis in wavelist:
            print(basis)
            threshold = round(OperaterUtils.heuristicThreshold(signal.input_ys),4) # 采用启发式阈值方法
            print(threshold)
            rec_a, rec_d = Wavelet.wavelet(ys=signal.input_ys, basis=basis, order=order,threshold=threshold)
            ys_o = [rec_a[i][0] for i in range(len(rec_a))]

            # ts = [i for i in range(len(ys_o))]
            signal.output_ys = ys_o
            title = "denoised singal"
            signal.plotDenoise(title=title)

            ys_n = signal.output_ys[:len(signal.signal_ys)] - signal.signal_ys #去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列
            MSE = OperaterUtils.getMSE(signal.signal_ys,signal.output_ys[:len(signal.signal_ys)])

            # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
            SNR_DWT = OperaterUtils.getSNR(signal.signal_ys, ys_n)
            print("MSE:", MSE)
            print("去噪后的SNR:", SNR_DWT)
            print("SNR提升量:", SNR_DWT - signal.SNR)
            print("SNR提升率:", (SNR_DWT - signal.SNR)/signal.SNR)

    def testLMS(self):
        SNR = 20.0

        # ys = np.array(pd.read_csv("../dataset/Blocks.csv")['ys'])
        # ys = mydsp.HeavySine()
        # ys,name = mydsp.Bumps()
        startFreq, endFreq = 2000, 200
        ys, name = mydsp.Droppler(startFreq=startFreq, endFreq=endFreq, amp=1)
        signal = mydsp.Signal(signal_ys=ys)
        signal.plot(title="origin")

        signal.addNoise(SNR=SNR)
        # signal = mydsp.HeavySine()
        signal.plot(title="input signal")

        output_ys,MSE= LMS.LMS(ys=signal.input_ys,dn=signal.signal_ys,a=0.0001)
        signal.output_ys = output_ys
        signal.plotDenoise(title="LMS")

        ys_n = signal.output_ys[:len(signal.signal_ys)] - signal.signal_ys  # 去噪信号序列 - 原始信号序列 = 分离出来的噪声信号序列

        # 去噪后的SNR = 10log((原始信号序列^2) / (分离出来的噪声信号序列^2))
        SNR_LMS = OperaterUtils.getSNR(signal.signal_ys, ys_n)
        print("MSE:", MSE)
        print("去噪后的SNR:", SNR_LMS)
        print("SNR提升量:", SNR_LMS - signal.SNR)
        print("SNR提升率:", (SNR_LMS - signal.SNR) / signal.SNR)

    def testDWTthreshold(self):
        ys, name = mydsp.Bumps()
        # threshold = OperaterUtils.rigsureThreshold(ys)
        # print(threshold)

        # threhold = OperaterUtils.sqtwologThreshold(ys)
        # print(threhold)
        #
        # threshold1 = OperaterUtils.heuristicThreshold(ys)
        # print(threshold1)

        threhold2 = OperaterUtils.rigsureThreshold(ys)
        print(threhold2)