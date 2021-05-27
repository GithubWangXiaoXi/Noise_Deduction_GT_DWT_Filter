from coreKits import thinkdsp
from matplotlib import pyplot as plt
import numpy as np
from utils import OperaterUtils

'''高斯噪声: 返回时间序列，振幅，数据类型均为ndarray'''
def GaussianNoise(amp,framerate):
    guassianNoise = thinkdsp.UncorrelatedGaussianNoise(amp=amp)
    wave = guassianNoise.make_wave(framerate=framerate)
    return wave.ys


'''瑞丽衰落信道'''
def RayleighFChannel(x,sigma):
    # y = 1 - np.exp(-np.power(x,2)/2 * np.power(sigma,2))  #分布函数
    return x * np.exp(-np.power(x,2)/ (2 * np.power(sigma,2)))/np.power(sigma,2) #概率密度函数（运算时注意加括号，保证运算顺序）

# signal = thinkdsp.SinSignal(freq=5,amp=10)
# signal1 = thinkdsp.SinSignal(freq=5,amp=5)
# wave = signal.make_wave(duration=1)
# wave1 = signal1.make_wave(duration=1)
#
# noise1 = thinkdsp.UncorrelatedGaussianNoise(amp=1)
# noise2 = thinkdsp.PinkNoise(beta=0.5,amp=1)
#
# input = wave + wave1 + noise1.make_wave() + noise2.make_wave()
# print(input)
# input.plot()
# plt.show()

'''重正弦 Heavy sine'''
def HeavySine(amp):
    sin = thinkdsp.SinSignal(freq= 1/(360 * np.pi),amp=amp)
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

    return wave.ys

'''方波(变形)  Blocks'''
def Blocks(freq,amp):
    block = thinkdsp.SquareSignal(freq=freq, amp=amp)
    wave = block.make_wave()
    return wave.ts, wave.ys


'''碰撞信号 Bumps'''
def Bumps(freq,amp,duration):

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
    return wave.ys

'''衰减信号 Droppler'''
def Droppler(startFreq,endFreq,amp):
    droppler = thinkdsp.ExpoChirp(start=startFreq,end=endFreq,amp=amp)
    wave = droppler.make_wave()
    return wave.ts, wave.ys

'''blocks信号（含高斯噪声）'''
def BlockWithNoise(ys_s,SNR):
    """
       ys_s: 原始block信号振幅序列
       amp_n:噪声振幅（默认添加Guassian噪声）
       returns: 添加噪声后的block信号振幅序列,以及信噪比SNR
    """
    block_wave = thinkdsp.Wave(ys_s)
    noise = thinkdsp.UncorrelatedGaussianNoise(ys=ys_s,SNR=SNR)
    noise_wave = noise.make_wave(framerate=len(ys_s))  #根据原始信号长度，对噪声进行采样

    output = block_wave.ys + noise_wave.ys  #Signal对象有加法操作，但是Wave对象没有

    SNR = OperaterUtils.getSNR(ys_s,noise_wave.ys)
    # 返回信噪比
    return output,SNR

'''Bumps信号（含高斯噪声）'''
def BumpsWithNoise(SNR,freq=55,amp=5,duration = 0.5):
    """
       ys_s: 原始bumps信号振幅序列
       amp_n:噪声振幅（默认添加Guassian噪声）
       returns: 添加噪声后的bumps信号振幅序列,以及信噪比SNR
    """
    ys_s = Bumps(freq=freq,amp=amp,duration=duration)
    bumps_wave = thinkdsp.Wave(ys=ys_s)
    noise = thinkdsp.UncorrelatedGaussianNoise(ys=ys_s,SNR=SNR)
    noise_wave = noise.make_wave(framerate=len(ys_s))  #根据原始信号长度，对噪声进行采样

    output = bumps_wave.ys + noise_wave.ys  #Signal对象有加法操作，但是Wave对象没有

    SNR = OperaterUtils.getSNR(ys_s,noise_wave.ys)
    # 返回信噪比
    return output,SNR

def HeavySineWithNoise(SNR,amp=2):
    """
       ys_s: 原始bumps信号振幅序列
       amp_n:噪声振幅（默认添加Guassian噪声）
       returns: 添加噪声后的bumps信号振幅序列,以及信噪比SNR
    """
    ys_s = HeavySine(amp=amp)
    bumps_wave = thinkdsp.Wave(ys=ys_s)
    noise = thinkdsp.UncorrelatedGaussianNoise(ys=ys_s,SNR=SNR)
    noise_wave = noise.make_wave(framerate=len(ys_s))  #根据原始信号长度，对噪声进行采样

    output = bumps_wave.ys + noise_wave.ys  #Signal对象有加法操作，但是Wave对象没有

    SNR = OperaterUtils.getSNR(ys_s,noise_wave.ys)
    # 返回信噪比
    return output,SNR