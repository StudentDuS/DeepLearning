import os
from PIL import Image
import glob

paths = glob.glob('./*')
# 输出所有文件和文件夹
for file in paths:
    fp = open(file, 'rb')
    img = Image.open(fp)
    fp.close()
    width = img.size[0]
    height = img.size[1]
    if (width <= 224) or (height <= 224):
        os.remove(file)
