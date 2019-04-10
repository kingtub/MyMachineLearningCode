from PIL import Image
import os

'''复制和粘贴图像区域'''

dirs = ['D:', 'li', 'tempImages']
dirStr = os.path.sep.join(dirs)

# 使用crop() 方法可以从一幅图像中裁剪指定区域：
# box = (100,100,400,400)
# region = pil_im.crop(box)
# 该区域使用四元组来指定。四元组的坐标依次是（左，上，右，下）

pil_im = Image.open(os.path.join(dirStr, 'screen.bmp'))
box = (100, 100, 400, 400)
region = pil_im.crop(box)
region.show()

# 我们可以旋转上面代码中获取的区域，然后使用
# paste() 方法将该区域放回去，具体实现如下:
region = region.transpose(Image.ROTATE_180)
pil_im.paste(region, box)
pil_im.show()