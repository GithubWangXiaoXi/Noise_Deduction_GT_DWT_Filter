import numpy as np


class GameTheroy:
    '''
    博弈论三要素：理性玩家，策略，支付函数
    '''

    def __init__(self, players=None):
        '''
        :param players: 多个理性玩家，这里为两个
        '''
        self.players = players
        self.payoff_Matrix = None
        self.row = dict() #行表示DWT的小波基basis
        self.col = dict() #列表示LMS的步长step

    def initRowAndCol(self,DWT_set,LMS_set):
        i = 0
        for basis in DWT_set:
            self.row[basis] = i
            i = i + 1

        i = 0
        for step in LMS_set:
            self.col[step] = i
            i = i + 1

        self.payoff_Matrix = list()
        for i in range(0, len(self.row)):
            self.payoff_Matrix.append([[0,0]] * len(self.col))
        print("row=",self.row)
        print("col=",self.col)

    def DWT_payoff(self, player1, player2):
        '''
        DWT的真实得分受到对手的制约
        :param player1:本玩家
        :param player2:对手
        :return:
        '''
        payoff = player1.payoff - 0.3 * player2.payoff
        # self.payoff_Matrix[rowIndex][colIndex] = payoff
        return payoff

    def LMS_payoff(self, player1, player2):
        '''
        LMS的真实得分受到对手的制约
        :param player1:本玩家
        :param player2:对手
        :return:
        '''
        payoff = player1.payoff - 0.3 * player2.payoff
        # self.payoff_Matrix[rowIndex][colIndex] = payoff
        return payoff


class Player:
    def __init__(self, name=None, basis=None,step = None):
        '''
        :param name:玩家姓名
        :param strategies:玩家可采取的策略
        :param MSE: 玩家在某策略下的MSE
        :param SNR_rate: 玩家在某策略下的SNR_rate
        :param payoff_func:玩家的支付函数名字（支付函数是相对于其他玩家的，受到对方分数的制约！！！）
        :param payoff:玩家对方采取不同策略下的支付值
        '''
        self.name = name
        self.MSE = None
        self.SNR_rate = None
        self.SNR = None
        self.basis = basis
        self.step = step
        self.payoff = None
        self.output_ys = None
        if (name == "DWT"):
            self.payoff_func = "DWT_payoff"
        elif(name == "LMS"):
            self.payoff_func = "LMS_payoff"

    def DWT_payoff(self, MSE, SNR_rate):
        '''
        DWT在每个小波基下单独的得分
        :param MSE:
        :param SNR_rate:
        :param basis:
        :return:
        '''
        # payoff = 20 * (1-SNR_rate) + 8 * (1 - MSE)  #这个支付函数存在问题：如果SNR=20，SNR_rate小概率大于1，而对于某种信号来说，1 - MSE大概率<0，导致得到的收益矩阵都是负的
        payoff = 5 * np.exp(1 - SNR_rate) + 2 * np.exp(1 - MSE)
        self.SNR_rate = SNR_rate
        self.MSE = MSE
        self.payoff = payoff
        return payoff

    def LMS_payoff(self, MSE, SNR_rate):
        '''
        LMS在每个步长下单独的得分
        :param MSE:
        :param SNR_rate:
        :param basis:
        :return:
        '''
        payoff = 2 * np.exp(1-SNR_rate) + 5 * np.exp(1 - MSE)
        self.SNR_rate = SNR_rate
        self.MSE = MSE
        self.payoff = payoff
        return payoff

def getNashEquilibrium(df):
    '''
    根据收益矩阵得到NashEquilibrium解
    1）删除严格劣策略
    2）画线法（如果在某个单元格中，两个策略均画上线，则为nash均衡解）
    :param df:收益矩阵
    :return:
    '''

    df_origin = df
    basisSet_origin = df[df.columns[0]] #原始小波基集合
    LMS_set_origin = df.columns[1:] #原始LMS步数集合
    basisSet_originN = len(df[df.columns[0]]) #小波基集合原始长度（为的是初始化DWT,LMS矩阵）
    LMS_set_originN = len(df.columns[1:])   #LMS步数集合原始长度（为的是初始化DWT,LMS矩阵）

    df = deleteBadStrategy(df) #删除严格劣策略

    # 更新df后的basisSet，LMS_set
    basisSet = df[df.columns[0]]  # 小波基集合
    LMS_set = df.columns[1:]  # LMS步数集合

    # 画线法:如果DWT_matrix,LMS_matrix在同一个位置均为1，则为nash均衡解
    DWT_matrix = list()
    LMS_matrix = list()
    for i in range(0,basisSet_originN):
        DWT_matrix.append([0] * LMS_set_originN)
        LMS_matrix.append([0] * LMS_set_originN)

    #先手:DWT,更新LMS_matrix
    for i in df.index:
        index = getBestStrategy(df,i,LMS_set)
        LMS_matrix[i][index] = 1

    #先手:LMS,更新DWT_matrix
    for i in range(0,len(LMS_set)):
        index = getBestStrategy1(df, LMS_set[i],basisSet)
        DWT_matrix[index][i] = 1

    NashEquilibriumIndex = list()
    NashEquilibriumVal = list()
    #寻找nash均衡解
    for i in range(0, len(basisSet)):
        for j in range(0, len(LMS_set)):
            if(DWT_matrix[i][j] == 1 and LMS_matrix[i][j] == 1):
                a = [i,j]
                NashEquilibriumIndex.append(a)
                # print(df_origin.at[i,LMS_set_origin[j]])
                NashEquilibriumVal.append(df_origin.at[i,LMS_set_origin[j]])

    return NashEquilibriumIndex,NashEquilibriumVal

def getBestStrategy(df,firstHand,secondSet):
    '''
    :param df:
    :param firstHand: 先手(只有一个策略)，DWT先做决策
    :param secondSet: secondSet是LMS的策略集合
    :return:
    '''

    scoreDict = {}
    # print(secondSet)
    for i in range(0, len(secondSet)):
        scoreDict[i] = str(df[secondSet[i]][firstHand]).split(",")[1].replace("]", "").lstrip()
    # print(scoreDict)
    value = max(zip(scoreDict.values(), scoreDict.keys()))
    # print(value)
    index = value[1]
    return index

def getBestStrategy1(df,firstHand,secondSet):
    '''
    :param df:
    :param firstHand: 先手(只有一个策略)，LMS先做决策
    :param secondSet: secondSet是DWT的策略集合
    :return:
    '''
    scoreDict = {}
    # print(secondSet)
    for i in df.index:
        scoreDict[i] = str(df[str(firstHand)][i]).split(",")[0].replace("[", "")
    value = max(zip(scoreDict.values(), scoreDict.keys()))
    index = value[1]
    return index

def deleteBadStrategy(df):
    '''
    删除严格劣策略
    :param df: 收益矩阵
    :return: 删除严格劣策略的收益矩阵
    '''
    DWT_bad,LMS_bad = 1, 1
    #1) 如果DWT或LMS的劣策略非空，则需要再进行一次劣策略判断
    while((DWT_bad != None or LMS_bad != None)):
        DWT_bad, LMS_bad = getBadStrategy(df)
        if(DWT_bad != None):
            print("删除basis劣策略:",df.at[DWT_bad,df.columns[0]])
            df = df.drop(DWT_bad, axis=0)
        if (LMS_bad != None):
            print("删除step劣策略:",df.columns[LMS_bad])
            df = df.drop(df.columns[LMS_bad], axis=1)
    return df

def getBadStrategy(df):

    if(len(df.columns) == 2 or len(df.index) == 1):# 判断df是否只有1行或1列
        return None,None

    #判断DWT是否存在劣策略
    DWT_basisScore = dict()
    DWT_bad = None
    for i in df.index:
        scoreList = [0] * (len(df.loc[i].values)-1)
        for j in range(1,len(df.loc[i].values)):
            scoreList[j-1] = str(df.loc[i].values[j:j+1]).split(",")[0].replace("['[","")
        DWT_basisScore[i] = scoreList

    # print(DWT_basisScore)
    row = len(DWT_basisScore)

    col = 0
    for i in range(0,len(DWT_basisScore)):
        if(DWT_basisScore.get(i) != None):
            col = len(DWT_basisScore[i])
            break

    if(len(df.index) != 1): #判断df是否只有1行
        for i in range(0,row):

            if(DWT_basisScore.get(i) == None): continue

            count = 0 #统计第i行所有值严格小于多少行，如果count == row - 1，则为严格劣策略
            for j in range(0,row):
                if(j == i):continue
                if(DWT_basisScore.get(j) == None): continue
                else:
                    flag = True  # 判断第i行的所有值相比于第j行，是不是严格最小
                    for k in range(0,col):
                        if(DWT_basisScore[i][k] > DWT_basisScore[j][k]):
                            flag = False
                            break
                    if(flag == True):
                        count = count + 1
            if(count == row - 1):
                DWT_bad = i

    #判断LMS是否存在劣策略
    LMS_stepScore = dict()
    LMS_bad = None
    # print(df.columns)
    if ((len(df.columns) - 1) != 1):  # 判断df是否只有1列
        for i in range(1,len(df.columns)):
            scoreList = [0] * (len(df.index))
            for j in range(0, len(df.index)):
                if (DWT_basisScore.get(j) == None): continue
                scoreList[j] = str(df.at[j,df.columns[i]]).split(",")[1].replace("]", "")
            LMS_stepScore[i-1] = scoreList

        # print(DWT_basisScore)
        row = len(LMS_stepScore)
        col = len(LMS_stepScore[0])
        for i in range(0, row):
            count = 0  # 统计第i行所有值严格小于多少行，如果count == row - 1，则为严格劣策略
            for j in range(0, row):
                if (j == i):
                    continue
                else:
                    flag = True  # 判断第i行的所有值相比于第j行，是不是严格最小
                    for k in range(0, col):
                        if (float(LMS_stepScore[i][k]) > float(LMS_stepScore[j][k])):
                            flag = False
                            break
                    if (flag == True):
                        count = count + 1
            if (count == row - 1):
                LMS_bad = i + 1  #df中第0列为DWT小波基列表

    return DWT_bad,LMS_bad

# if __name__ == '__main__':
    # payoff_Matrix = list()
    # print(payoff_Matrix)
    # print([[0, 0]] * 5)
    # payoff_Matrix.append([[0, 0]] * 2 for i in range(0, 2))
    # print(payoff_Matrix)

    # payoff_Matrix = list()
    # print(payoff_Matrix)
    # print([[0, 0]] * 5)
    # for i in range(0, 2):
    #     payoff_Matrix.append([[0, 0]] *2)
    # print(payoff_Matrix)