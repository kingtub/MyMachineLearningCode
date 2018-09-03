from numpy import *
import matplotlib.pyplot as plt
# 这是K-均值聚类算法


def loadData(fileName):
    dataMat = []
    with open(fileName) as f:
        for line in f.readlines():
            ss = line.strip().split('\t')
            dataMat.append([float(ss[0]), float(ss[1])])

    return array(dataMat)


# 计算2个特征向量的距离
def calDistance(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 产生K个随机向量，用于初始化
def randCent(dataMat, k):
    # 产生的特征向量的值必须在已有的数据范围内
    mins = dataMat.min(axis=0)
    maxs = dataMat.max(axis=0)
    rans = maxs - mins
    kCenters = zeros((k, dataMat.shape[1]))

    for i in range(k):
        r = random.rand()
        kCenters[i, :] = r * rans + mins
    return kCenters


def kMeans(dataMat, k):
    kCenters = randCent(dataMat, k)
    assignMat = zeros((dataMat.shape[0], 2))
    changed = True
    # 遍历每个特征向量，通过计算其与每个中心的距离，把它归到最小距离的中心；
    # 然后取每个聚类群的均值作为新的中心。重复这个过程，直到每个点的分配不再改变
    while changed:
        changed = False
        for i in range(dataMat.shape[0]):
            vec = dataMat[i, :]
            minDistance = 1e100
            ci = -1
            for j in range(kCenters.shape[0]):
                d = calDistance(vec, kCenters[j, :])
                if d < minDistance:
                    minDistance = d
                    ci = j

            oldCi = assignMat[i, 0]
            if oldCi != ci:
                changed = True
            assignMat[i, 0] = ci
            assignMat[i, 1] = minDistance ** 2

        if changed:
            kmeanSum = zeros((kCenters.shape))
            count = zeros(kCenters.shape[0])
            for i in range(assignMat.shape[0]):
                kmeanSum[int(assignMat[i, 0]), :] += dataMat[i, :]
                count[int(assignMat[i, 0])] += 1
            for j in range(kCenters.shape[0]):
                kCenters[j, :] = kmeanSum[j, :]/ count[j]
            #print(kCenters)

    return kCenters, assignMat


def run():
    dataMat = loadData('testSet.txt')
    # print(dataMat)

    kCenters, assignMat = kMeans(dataMat, 4)

    x1 = [e[0] for e in dataMat]
    x2 = [e[1] for e in dataMat]
    plt.scatter(x1, x2)

    c1 = [e[0] for e in kCenters]
    c2 = [e[1] for e in kCenters]
    plt.scatter(c1, c2, c='r')
    plt.show()




run()