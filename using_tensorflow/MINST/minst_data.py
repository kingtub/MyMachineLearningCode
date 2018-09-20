import os
import struct
import numpy as np


def load_mnist():
    """Load MNIST data from `path`"""
    path = 'C:\D\Code\Python\machine_learning_data'
    train_labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    train_images_path = os.path.join(path, 'train-images.idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    with open(train_labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        train_labels = np.fromfile(lbpath,
                                   dtype=np.uint8)

    with open(train_images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        train_images = np.fromfile(imgpath,
                                   dtype=np.uint8).reshape(len(train_labels), 784)

    with open(test_labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        test_labels = np.fromfile(lbpath,
                                  dtype=np.uint8)

    with open(test_images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        test_images = np.fromfile(imgpath,
                                  dtype=np.uint8).reshape(len(test_labels), 784)

    return train_images, train_labels, test_images, test_labels


def show_some_image(train_images, train_labels, test_images, test_labels):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(10):
        img = train_images[train_labels == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


# 5 -> [0 0 0 0 0 1 0 0 0 0]
def transform_labels(labels):
    m = len(labels)
    a = np.zeros((10, m))
    for i in range(m):
        a[labels[i], i] = 1
    return a


class Data:
    def __init__(self, batch_size=100):
        train_images, train_labels, test_images, test_labels = load_mnist()
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        self.batch_size = batch_size
        self.init_indexs()

    def init_indexs(self):
        m_train_images = len(self.train_images)
        # m_train_labels = len(self.train_labels)
        # m_test_images = len(self.test_images)
        # m_test_labels= len(self.test_labels)

        self.train_images_indexs = list(range(m_train_images))
        # self.train_labels_indexs = list(range(m_train_labels))
        # self.test_images_indexs = list(range(m_test_images))
        # self.test_labels_indexs = list(range(m_test_labels))

    def re_init(self):
        self.init_indexs()

    def test_data(self):
        return np.mat(self.test_images, dtype=np.float32), np.mat(transform_labels(self.test_labels), dtype=np.float32)

    def next_train_batch(self):
        if 0 == len(self.train_images_indexs):
            self.re_init()

        imgs = []
        labels = []
        # 从列表中选择不重复元素
        a = np.random.choice(self.train_images_indexs, size=self.batch_size, replace=False)

        for i in a:
            imgs.append(self.train_images[i])
            labels.append(self.train_labels[i])
            self.train_images_indexs.remove(i)

        return np.mat(imgs), np.mat(transform_labels(labels))


def testing():
    train_images, train_labels, test_images, test_labels = load_mnist()
    # show_some_image(train_images, train_labels, test_images, test_labels)

    #print(train_images[0])
    #print(train_labels[0])
    print(train_images.shape)
    print(train_labels.shape)


#testing()


