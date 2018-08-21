from numpy import *
# 这是朴素贝叶斯分类算法

def loadDataSet():
    # 这是养狗社区训练过滤歧视语言的语言集
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


# 把所有的单词汇总起来，构建一个单词唯一的集合
def createVocabList(vocabVectors):
    s = set([])
    for vec in vocabVectors:
        s = s | set(vec)
    return list(s)


# 把每条句子转换成词向量，其中单词出现的地方为1，不出现为0
def sentenceToVector(vocabList, sentence):
    vlen = len(vocabList)
    sentenceVector = [0] * vlen

    for word in sentence:
        # 改为词袋模型，即累加词语出现的次数 而不是简单置为1
        # sentenceVector[vocabList.index(word)] = 1 原来方式
        sentenceVector[vocabList.index(word)] = 1

    return sentenceVector


# 训练朴素贝叶斯分类器
# 函数返回各个类别下的各个特征的概率的向量，和总的歧视概率
def trainingNB(sentencesVectors, classVec):
    # 有歧视语句分类的总概率
    pAbusive = sum(classVec) / len(classVec)
    # 利用贝叶斯分类器对文档进行分类时，要计算多个（单词的）概率的乘积，如果其中一个为0，那么乘积为0，为降低这种影响，
    # 可以将所有词的出现数初始化为1，且将分母初始化为2——这其实是Laplace校准
    # p0Sum = zeros(sentencesVectors.shape[1]) 原来的方式
    # p1Sum = zeros(sentencesVectors.shape[1])
    # p0Demon = 0
    # p1Demon = 0

    p0Sum = ones(sentencesVectors.shape[1])
    p1Sum = ones(sentencesVectors.shape[1])
    p0Demon = 2
    p1Demon = 2
    for i in range(len(classVec)):
        if classVec[i] == 0:
            p0Sum += sentencesVectors[i, :]
            p0Demon += sum(sentencesVectors[i, :])
        else:
            p1Sum += sentencesVectors[i, :]
            p1Demon += sum(sentencesVectors[i, :])

    # 为了不让很多个小概率的乘积变为0，决定每个概率都用log包装
    # 所以Pa1 * Pa2 --> log(Pa1 * Pa2)=log(Pa1) + log(Pa2)
    # p0Vec = p0Sum / p0Demon  原来的方式
    # p1Vec = p1Sum / p1Demon

    p0Vec = log(p0Sum / p0Demon)
    p1Vec = log(p1Sum / p1Demon)

    return p0Vec, p1Vec, pAbusive


def classifyNB(sentenceVectorArr, p0Vec, p1Vec, pAbusive):
    # 为了不让很多个小概率的乘积变为0，决定每个概率都用log包装
    # 所以Pa1 * Pa2 --> log(Pa1 * Pa2)=log(Pa1) + log(Pa2)

    p0 = sum(p0Vec * sentenceVectorArr) + log(1 - pAbusive)
    p1 = sum(p1Vec * sentenceVectorArr) + log(pAbusive)

    if p0 > p1:
        return 0
    else:
        return 1


# 使用
def testing():
    postingList, classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    sentencesVectors = []
    for line in postingList:
        sentencesVectors.append(sentenceToVector(vocabList, line))

    p0Vec, p1Vec, pAbusive = trainingNB(array(sentencesVectors), classVec)
    # print('p0Vec=', p0Vec)
    # print('p1Vec=', p1Vec)

    s1 = ['love', 'my', 'dalmation']
    s2 = ['stupid', 'garbage']
    print(s1, ' classified as', classifyNB(array(sentenceToVector(vocabList, s1)), p0Vec, p1Vec, pAbusive))
    print(s2, ' classified as', classifyNB(array(sentenceToVector(vocabList, s2)), p0Vec, p1Vec, pAbusive))

#testing()