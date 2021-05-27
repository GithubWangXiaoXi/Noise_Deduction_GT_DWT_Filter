from coreKits import thinkdsp
from matplotlib import pyplot as plt
import numpy as np
from utils import OperaterUtils

class Signal:

    def __init__(self,signal_ys):
        '''
        :param signal_ys: 原始信号序列
        :param ts: 时序序列
        :param noise_ys: 噪声序列
        :param input_ys:混合信号序列
        :param SNR:信噪比
        :param output_ys:降噪信号序列
        '''
        self.signal_ys = signal_ys
        self.ts = [i for i in range(len(self.signal_ys))]
        self.noise_ys = None
        self.input_ys = self.signal_ys
        self.SNR = None
        self.output_ys = None

    def addNoise(self,SNR):
        '''
        默认添加高斯噪声
        :param SNR:
        :return:
        '''
        block_wave = thinkdsp.Wave(self.signal_ys)
        noise = thinkdsp.UncorrelatedGaussianNoise(ys=self.signal_ys, SNR=SNR)
        noise_wave = noise.make_wave(framerate=len(self.signal_ys))  # 根据原始信号长度，对噪声进行采样

        input = block_wave.ys + noise_wave.ys  # Signal对象有加法操作，但是Wave对象没有

        self.noise_ys = noise_wave.ys
        self.SNR = SNR
        self.input_ys = input

        SNR = OperaterUtils.getSNR(self.signal_ys, noise_wave.ys)
        print("SNR:",SNR - 4.5)

    def plot(self,title):
        ts = [i for i in range(len(self.signal_ys))]
        plt.title = title
        plt.plot(ts, self.input_ys)
        plt.show()

    def plotDenoise(self, title):
        ts = [i for i in range(len(self.output_ys))]
        plt.title = title
        plt.plot(ts, self.output_ys)
        plt.show()


def HeavySine(amp = 2):
    '''
    重正弦 Heavy sine
    '''
    sin = thinkdsp.SinSignal(freq= 1/(2 * np.pi),amp=amp)
    duration = sin.period * 2

    wave = sin.make_wave(duration=duration,framerate=2000)
    # print( 0.8 * wave.ys[period1:period2] - 0.2)
    period1 = len(wave.ys) // 4
    period2 = 3 * (len(wave.ys) // 4)
    segment1 = 0.8 * wave.ys[period1:period2] - 0.2  #切片是通过复制得到的，并非在原列表上进行操作。
    segment2 = 1.2 * wave.ys[period1:period2] + 0.1

    k = 0
    for i in range(period1,period2):
        wave.ys[i] = segment1[k]
        k = k + 1

    k = 0
    for i in range(period2, len(wave.ys)):
        wave.ys[i] = segment2[k]
        k = k + 1

    return wave.ys,"HeavySine"


# def Blocks(freq,amp = 2):
#     '''
#     方波(变形)  Blocks
#     '''
#     block = thinkdsp.SquareSignal(freq=freq, amp=amp)
#     wave = block.make_wave()
#     return wave.ts, wave.ys


def Bumps(freq = 55,amp = 2,duration = 0.5):
    '''
    碰撞信号 Bumps
    '''
    bump1 = thinkdsp.SinSignal(freq=freq, amp=amp)
    for i in range(1,3): #叠加正弦
        freqTmp = freq // (i+1)
        ampTmp = amp * (i+1)
        bump1 += thinkdsp.SinSignal(freq=freqTmp, amp=ampTmp)

    for i in range(0,3): #叠加余弦
        freqTmp = freq // (i + 1)
        ampTmp = amp * (i + 1)
        bump1 += thinkdsp.CosSignal(freq=freqTmp, amp=ampTmp)

    wave = bump1.make_wave(duration=duration)
    wave.ys[wave.ys < 0] = 0
    return wave.ys,"Bumps"


def Droppler(startFreq,endFreq,amp=1):
    '''
    衰减信号 Droppler
    '''
    signal = thinkdsp.ExpoChirp(start=startFreq, end=endFreq)
    wave = signal.make_wave(duration=0.05)
    ts = wave.ts
    # print(ts,len(ts))
    f0, f1 = np.log10(startFreq), np.log10(endFreq)
    freqs = np.logspace(f0, f1, len(ts))
    # print("---------------------------------")
    # print(freqs, len(freqs))
    dts = np.diff(ts, prepend=0)
    # print("---------------------------------")
    # print(dts,len(dts))
    dphis = thinkdsp.PI2 * freqs * dts
    phases = np.cumsum(dphis)

    phases = phases / 2 * np.pi
    N = len(phases)
    ys = [0] * N
    # temp_i = 0
    for i in range(0, N, 4):
        phase_copy = phases.copy()
        # temp_i = i
        # i = i * amp
        for j in range(0, N):  # 处理phase采样间隔：保留2个pi周期内采样
            if (phase_copy[j] < np.pi * (i + 4) and phase_copy[j] > np.pi * i):
                None
            else:
                phase_copy[j] = np.pi / 2
        amp = amp * np.power(1.1,1.3)
        ys += amp * np.cos(phase_copy)

    ys = np.array(ys)
    print(type(ys))
    return ys,"Droppler"


def GaussianNoise(amp,framerate):
    '''
    高斯噪声: 返回时间序列，振幅，数据类型均为ndarray
    '''
    guassianNoise = thinkdsp.UncorrelatedGaussianNoise(amp=amp)
    wave = guassianNoise.make_wave(framerate=framerate)
    return round(wave.ys,3)

