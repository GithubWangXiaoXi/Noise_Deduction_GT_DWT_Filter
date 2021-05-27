'''
   算法：最小均方算法(lms)
   均方误差：样本预测输出值与实际输出值之差平方的期望值，记为MES
   设:observed 为样本真值,predicted为样本预测值,则计算公式:
   (转换为容易书写的方式，非数学标准写法,因为数学符号在这里不好写)
   MES=[(observed[0]-pridicted[0])*(observed[0]-pridicted[0])+....
         (observed[n]-pridicted[n])*(observed[n]-pridicted[n])]/n
'''

'''
   变量约定：大写表示矩阵或数组，小写表示数字
   X：表示数组或者矩阵
   x:表示对应数组或矩阵的某个值
'''

'''
     关于学习效率（也叫步长：控制着第n次迭代中作用于权值向量的调节）。(下面的参数a)：
     学习效率过大：收敛速度提高，稳定性降低，即出结果快，但是结果准确性较差
     学习效率过小：稳定性提高，收敛速度降低，即出结果慢，准确性高，耗费资源
     对于学习效率的确定，有专门的算法，这里不做研究。仅仅按照大多数情况下的选择：折中值
'''
import numpy as np

#注意步长0<a<1，如果设置太大，w权值变化太快，误差增大
def LMS(ys,dn,a = 0.0001):
    '''

    :param ys: 输入信号
    :param dn: 期望信号
    :param a: 学习步长
    :return: 去噪信号（权值系数 * 抽头输入）,和均方误差MSE
    '''
    signal_list = list()

    M = 10 #滤波器抽头数，即阶数
    for i in range(0,M):  #经过M个延迟算子，对输入信号ys进行M步延迟处理（Z^-k[X(n)]=X(n-k)）,将信号整体向后平移,前面位置补0
        ys_delay = list()
        # leftMove(ys_delay,i)
        ys_delay.extend([0] * i)
        ys_delay.extend(ys[0:len(ys) - i])
        signal_list.append(ys_delay)

    # print(signal_list)
    # X = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0,1]])  ##输入矩阵
    # D = np.array([1, 1, 1, 1])  ##期望输出结果矩阵

    X = np.array(signal_list).T  ##输入矩阵,列向量为每一阶抽头输入信号！！！
    X = np.around(X,4)
    D = np.array(dn)  ##期望输出结果矩阵
    W = np.array([1] * M)  ##权重向量
    expect_e = 0.005  ##期望误差
    maxtrycount = 20  ##最大尝试次数

    ##修正权值
    cnt = 0
    MSE = 0
    while True:
        err = 0
        i = 0
        for xn in X:
            W, e = neww(W, D[i], xn, a)
            # if(e > 1): print("e,W,x(i),y(i),d(i) = ",e,",",W,",",xn,",",xn*W,",",D[i])
            i = i + 1
            err += round(pow(e, 2),4)  ##lms算法的核心步骤，即：MES   如果误差信号为-1072284757.0473，指数运算肯定溢出,所以记得求期望
        err /= float(i)
        MSE = err
        cnt += 1
        # print(u"第 %d 次调整后的权值：" % cnt)
        # print(W)
        # print(u"误差：%f" % err)
        if err < expect_e or cnt >= maxtrycount:
            break

    output = [0] * len(ys)
    i = 0
    for xn in X:
        output[i] = get_v(W,xn)
        i = i + 1

    print("MSE：", MSE)
    print("最后的权值：", W.T)
    return output,MSE



# 参考<https://www.cnblogs.com/ahua1188/p/6149636.html>
def reverse(l, left, right):
    for m in range((right-left)//2):
        temp = l[right-1-m]
        l[right-1-m] = l[left+m]
        l[left+m] = temp

# 列表整体向后平移step步
def leftMove(list, step):
    reverse(list, 0, len(list))
    reverse(list, 0, len(list)-step)
    reverse(list, len(list)-step, len(list))


def get_v(W, x):
    ##读取实际输出
    '''
        这里是两个向量相乘，对应的数学公式：
        a(m,n)*b(p,q)=m*p+n*q
        在下面的函数中，当循环中xn=1时(此时W=([0.1,0.1]))：
        np.dot(W.T,x)=(1,1)*(0.1,0.1)=1*0.1+1*0.1=0.2>0 ==>sgn 返回1
    '''
    y = np.dot(W.T, x)  ##dot表示两个矩阵相乘
    return np.around(y,4)


def get_e(W, x, d):
    ##读取误差值
    y = get_v(W, x)
    # print("W(M) = ",W)
    # print("y = ",y)
    # print("d = ", d)
    return round(d - y,4)

def neww(oldW, d, x, a):
    ##权重计算函数(批量修正)
    '''
      对应数学公式: w(n+1)=w(n)+a*x(n)*e
      对应下列变量的解释：
      w(n+1) <= neww 的返回值
      w(n)   <=oldw(旧的权重向量)
      a      <= a(学习率，范围：0<a<1)
      x(n)   <= x(输入值)
      e      <= 误差值或者误差信号
    '''
    e = get_e(oldW, x, d)
    return (np.around(oldW + a * x * e,4), e)