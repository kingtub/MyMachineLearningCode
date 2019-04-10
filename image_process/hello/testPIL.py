from PIL import Image
import os
dirs = ['D:', 'li', 'tempImages']
dirStr = os.path.sep.join(dirs)

pil_im = Image.open(dirStr + os.path.sep + 'screen.bmp')
#pil_im.show()

# 将其转换成灰度图像
l = pil_im.convert('L')
l.show()
# 保存成bmp格式
l.save(dirStr + os.path.sep + 'screenLLL.bmp')
# 保存成jpeg格式
l.save(dirStr + os.path.sep + 'screenLLL.jpg')

