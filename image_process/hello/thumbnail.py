from PIL import Image
import os

'''创建缩略图'''

dirs = ['D:', 'li', 'tempImages']
dirStr = os.path.sep.join(dirs)

for f in os.listdir(dirStr):
    img = Image.open(os.path.join(dirStr, f))
    # 创建图像的缩略图, 修改img对象了，参数元组表示不能超过这么长（像素）
    img.thumbnail((128, 128))
    img.show()