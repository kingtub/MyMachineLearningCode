from PIL import Image
from numpy import *
import os

# 图像二值化处理

dirs = ['D:', 'li', 'tempImages']
dirStr = os.path.sep.join(dirs)

# 读取出图像Image对象
pil_im = Image.open(os.path.join(dirStr, 'screen.bmp'))
# 转换成numpy数组
im = array(pil_im)

# 灰度处理后转换成numpy数组
imL = array(pil_im.convert('L'))
# # 显示灰度图像
# pil_im.convert('L').show()

# 二值化处理
a = 255 * (imL > 128)
# 从numpy数组创建图像
aImg = Image.fromarray(a)
# aImg.save('my.png')
aImg.show()

# 对上述二值图像反转
Image.fromarray(255 - a).show()
# 对灰度图像反转
Image.fromarray(255 - imL).show()

print(im.shape)
print(im.dtype)

print(imL.shape)
print(imL.dtype)
