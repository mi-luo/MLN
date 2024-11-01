import os
import shutil

strleft = r"/data1/wanglu/datasets/insulator_haze_oneclass/nohaze/v5/val/images"
strRight = r"/data1/wanglu/datasets/insulator_haze_oneclass/nohaze/VOC2012/Annotations"

strDst = r"/data1/wanglu/datasets/insulator_haze_oneclass/nohaze/VOC2012/valxml"  # 目标文件

lsLeft = os.listdir(strleft)
lsRight = os.listdir(strRight)

lsLeftFileName = []
lsRightFileName = []

for i in lsLeft:
    lsLeftFileName.append(i.split('.')[0])

for i in lsRight:
    lsRightFileName.append(i.split('.')[0])

for i in lsRightFileName:
    if i in lsLeftFileName:  # 用于选择出两个文件夹中存在的文件名相同的文件
        shutil.copy(strRight + '/' + i + '.xml', strDst + '/' + i + '.xml')  # 文件类型改为.xml