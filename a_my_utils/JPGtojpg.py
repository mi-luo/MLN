import pdb

import shutil
import os
import xml.etree.ElementTree as ET

# python 3.2

#dir = "/home/Desktop/workpy";
dir = "/data/pth/wl-pth/firstsandxuexiao/VOC2007";

my_files = os.listdir(dir + "/JPEGImages/")
os.mkdir(dir + "/JPEGImages_jpg/")
for files in my_files:
    if "JPG" in files:
        shutil.copyfile(dir + "/JPEGImages/" + files, dir + "/JPEGImages_jpg/" + files[:-4] + ".jpg")
