import pywt
import numpy as np
from matplotlib import pyplot as plt
from utils import OperaterUtils

def wavelet(ys,basis,order,threshold=None):
    '''

    :param ys: 含噪信号
    :param waveletBasis: 小波基名称
            ['haar']
            ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38']
            ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
            ['coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17']
            ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']
            ['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']
            ['dmey']
            ['gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8']
            ['mexh']
            ['morl']
            ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8']
            ['shan']
            ['fbsp']
            ['cmor']
    :param order: 分解层数
    :return:
    '''

    w = pywt.Wavelet(basis)
    a = ys
    ca = []  # 近似分量 低频成分
    cd = []  # 细节分量 高频成分
    # 一维小波变换（5阶,得到每一阶的高频 + 低频）
    var = 1
    for i in range(0, order):

        (a, d) = pywt.dwt(a, w) #离散小波变换，默认mode='symmetric'，即对称分解，假设原始信号长度为2000，则第一次变换会得到1000低频 + 1000高频

        if(threshold != None):
            var = OperaterUtils.variance(d)  # 对高频成分进行方差估计
            threshold = threshold * var
            a = pywt.threshold(a, threshold, mode="soft", substitute=0)  # 建议采用启发式阈值 * 各尺度下高频成分的方差估计

        ca.append(a)
        cd.append(d)

    # print("cA:", ca)  # 得到所有低频
    # print("cD:", cd)  # 得到所有高频

    rec_a = []
    rec_d = []

    # 只用近似分量（低频成分）来重构经过降噪后的原始信号，而非逐层使用低频 + 高频，重构输入信号
    for i, coeff in enumerate(ca):

        '''
         为什么要用None填充长度??
         一方面，在waverec参数中，要求coeff_list是一个二维数组；coeff是一个一维array,如果直接传参，肯定会报错。
             比如这么写：coeff_list = coeff  #会报ValueError: Expected sequence of coefficient arrays.
         另一方面，在waverec中：a, ds = coeffs[0], coeffs[1:]，ds一定要补位，
             比如这么写：coeff_list = [coeff, None]就不会报错
        '''
        coeff_list = [coeff, None] + [None] * i  #用None填充长度，也可以理解成将高频置为None。
        # coeff_list = [coeff, None]# 用None填充长度，也可以理解成将高频置为None。

        rec_a.append(pywt.waverec(coeff_list, w))  # waverec重构, wavedec分解

    rec_a = np.array(rec_a[-1])  # -1取最后一个基波
    rec_a = rec_a.reshape(-1, 1)
    # rec_a = pywt.idwt(ca[0],ca[0],wavelet=w)
    # print(rec_a)

    # 只用细节分量（高频成分）来重构，提取出来的噪声信号，而非逐层使用低频 + 高频，重构输入信号
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w)) #取出所有噪声波，1阶变换，有3个噪声
    # rec_d = np.array(rec_d)
    rec_d = np.array(rec_d[-1])  # -1取最后一个基波
    rec_d = rec_d.reshape(-1, 1)

    # print("ca", ca)
    # print("rec_a",rec_a)

    return rec_a, rec_d  # 取出趋势，噪声
