# -*- coding: UTF-8 -*- 
# !/usr/bin/env python
import os.path

from PIL import Image

im_num = []
for line in open("/data1/wanglu/datasets/plad/plad73/VOC2012/ImageSets/main/train.txt", "r"):
    im_num.append(line)
# print(im_num)

for a in im_num:
    im_name = '/data1/wanglu/datasets/plad/plad73/VOC2012/JPEGImages/{}'.format(a[:-1]) + '.jpg'  #原始路径
    if not os.path.exists(im_name):
        im_name = '/data1/wanglu/datasets/plad/plad73/VOC2012/JPEGImages/{}'.format(a[:-1]) + '.JPG'  # 原始路径
    print(im_name)
    im = Image.open(im_name)  # 打开指定路径下的图像

    tar_name = '/data1/wanglu/datasets/plad/plad73/VOC2012/ImageSets/main/trainimgs/{}'.format(a[:-1]) + '.jpg'  #移动后的路径
    print(tar_name)
    im.save(tar_name)  # 另存
    im.close()
