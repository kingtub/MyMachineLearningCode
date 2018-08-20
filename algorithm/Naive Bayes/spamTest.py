from numpy import *
import naive_bayes as nb

# 过滤垃圾邮件的例子
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(0)

    vocabList = nb.createVocabList(docList)

    trainingSet = [i for i in range(50)]
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    print(trainingSet)
    print(testSet)

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(nb.sentenceToVector(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])


    p0Vec, p1Vec, pSpam = nb.trainingNB(array(trainMat), trainClasses)

    errorCount = 0
    for docIndex in testSet:
        c = nb.classifyNB(array(nb.sentenceToVector(vocabList, docList[docIndex])), p0Vec, p1Vec, pSpam)
        if c != classList[docIndex]:
            errorCount += 1
            print(docList[docIndex])
    print('Error rate is ', errorCount / len(testSet))


spamTest()