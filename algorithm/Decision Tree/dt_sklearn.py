from sklearn.tree import DecisionTreeClassifier


def test1():
    X = [[1, 1],
         [1, 1],
         [1, 0],
         [0, 1],
         [0, 1]]
    y = [['yes'],
         ['yes'],
         ['no'],
         ['no'],
         ['no']]
    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4)
    # 拟合模型
    clf.fit(X, y)
    print(clf.predict([[1, 1], [1, 0], [0, 1]]))


def test2():
    X = []
    y = []
    with open('lenses.txt') as f:
        for line in f.readlines():
            ss = line.strip().split('\t')
            y.append(ss[-1])
            del ss[-1]
            X.append(ss)
    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4, )
    # 拟合模型
    clf.fit(X, y)
    # soft   no lenses   hard
    print(clf.predict([['young', 'myope', 'no', 'normal'], ['young', 'myope', 'no', 'reduced'], ['pre', 'myope', 'yes', 'normal']]))


#test1()
test2()
