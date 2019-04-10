from PIL import Image
from numpy import *
import os


def get_imlist(path):
    """ 返回目录中所有JPG 图像的文件名列表"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    """ 使用PIL 对象重新定义图像数组的大小"""
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


def compute_average(imlist):
    """ 计算图像列表的平均图像"""
    # 打开第一幅图像，将其存储在浮点型数组中
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)
    # 返回uint8 类型的平均图像
    return array(averageim, 'uint8')
