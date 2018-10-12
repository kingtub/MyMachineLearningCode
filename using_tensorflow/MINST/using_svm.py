from sklearn import svm
import time
# 运行太久了，2个多小时都没跑完，在我的ThinkPad笔记本上
import minst_data


# 5 -> [0 0 0 0 0 1 0 0 0 0]
def transform_labels(labels):
    m = len(labels)
    result = []
    for i in range(m):
        item = [0] * 10
        item[labels[i]] = 1
        result.append(item)
    return result


def train_and_test():
    print("开始时间为 :", time.localtime(time.time()))

    train_images, train_labels, test_images, test_labels = minst_data.load_mnist()
    # train_labels = transform_labels(train_labels)
    # test_labels = transform_labels(test_labels)

    clf = svm.SVC()  # class
    clf.fit(train_images, train_labels)  # training the svc model

    test_result = clf.predict(test_images)
    errorCount = 0
    for i in range(len(test_result)):
        ti = test_result[i]
        labelsI = test_labels[i]
        if ti != labelsI:
            errorCount += 1

    print('ErrorRate: ', errorCount/len(test_result))
    print("结束时间为 :", time.localtime(time.time()))


train_and_test()