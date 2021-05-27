from matplotlib import pyplot
from coreKits import thinkdsp
import numpy as np
import math

def smooth(ys, window):
    N = len(ys)
    smoothed = np.zeros(N)
    '''
    padded 是窗的一个版本，它在末尾添加0，以保持和segment.ys 的长度一致。
    '''
    padded = thinkdsp.zero_pad(window,N)
    rolled = padded
    for i in range(N):
        smoothed[i] = sum(rolled * ys)
        '''
        当将滑动后的窗口与波形矩阵相乘，我们就得到了波形矩阵下一组11个元素的均值，这11个元素从波形数组第二个开始。
        '''
        rolled = np.roll(rolled, 1)
    return smoothed

# 参考<https://blog.csdn.net/benbenls/article/details/103338153>
def getSNR(signal,noise):
    energy_o = np.sum(np.power(signal, 2))
    energy_n = np.sum(np.power(noise, 2))
    return 10 * np.log10(energy_o / energy_n)

def getMSE(signal,output):
    err = 0
    for i in range(0,len(signal)):
        err += np.power(output[i] - signal[i],2)
    return err/float(len(signal))

# 参考<https://blog.csdn.net/zhang0558/article/details/76019832>
def rigsureThreshold(ys):
    '''
    无偏风险估计阈值
    :param ys: 信号序列
    :return:
    '''
    N = len(ys)
    # ys = np.array(ys)
    ys = list(abs(ys))
    ys.sort()
    ys_square = np.power(ys,2)
    min = 100
    min_point = -1 #risk最小值位置
    for k in range(1,N):
        risk = riskFunc(k,N,ys_square)
        if risk < min:
            min = risk
            min_point = k
    return math.sqrt(ys_square[min_point])

def riskFunc(k,N,ys_square):
    risk = N - (2 * k) + (N - k) * ys_square[N-k]
    for j in range(1,N):
        risk += (ys_square[j])
    return risk // N

def sqtwologThreshold(ys):
    '''
    固定阈值
    :param ys: 信号序列
    :return:
    '''
    return math.sqrt(2*np.log(len(ys)))

def heuristicThreshold(ys):
    '''
    启发式阈值
    :param ys:
    :return:
    '''
    N = len(ys)
    ys = list(ys)
    crit = math.sqrt((1/N) * math.pow(math.log(N,math.e)/math.log(2,math.e),3))
    temp = 0
    for j in range(1,N):
        temp += math.pow(abs(ys[j]),2) - N
    eta = temp / N
    if(eta < crit):
        return sqtwologThreshold(ys)
    else:
        return min(sqtwologThreshold(ys),rigsureThreshold(ys))

def variance(ys):
    '''
    小波分解在每个尺度下高频成分的方差估计
    :param ys:高频信号序列
    :return:
    '''
    var = np.median(ys) / 0.6745
    # print("var=",var)
    # print("mean=",np.mean(ys))
    return var


