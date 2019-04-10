from PIL import Image
from numpy import *
import os

dirs = ['D:', 'li', 'tempImages']
dirStr = os.path.sep.join(dirs)


pil_im = Image.open(os.path.join(dirStr, 'screen.bmp'))
im = array(pil_im)
print(im.shape)
print(im.dtype)

imL = array(pil_im.convert('L'))
print(imL.shape)
print(imL.dtype)
