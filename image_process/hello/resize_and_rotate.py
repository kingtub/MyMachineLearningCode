from PIL import Image
import os

'''调整尺寸和旋转'''

dirs = ['D:', 'li', 'tempImages']
dirStr = os.path.sep.join(dirs)

# 要调整一幅图像的尺寸，我们可以调用resize() 方法。该方法的参数是一个元组，
# 用来指定新图像的大小：

pil_im = Image.open(os.path.join(dirStr, 'screen.bmp'))
out = pil_im.resize((128, 128))
out.show()

# 要旋转一幅图像，可以使用逆时针方式表示旋转角度，然后调用rotate() 方法：
pil_im.rotate(45).show()