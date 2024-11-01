import os
import random

trainval_percent = 1.0
train_percent = 0.8
#xmlfilepath = 'Annotations'
xmlfilepath = '/data1/wanglu/datasets/insulator_defect/defect1twoclass/VOC2007/Annotations'
            #txtsavepath = 'ImageSets\Main'
txtsavepath = '/data1/wanglu/datasets/insulator_defect/defect1twoclass/VOC2007/ImagesSets/main'
total_xml = os.listdir(xmlfilepath)
#total_xml = os.listdir('D:\mmdetection2.6\data\VOCdevkit\VOC2007\Annotations')

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

#ftrainval = open('D:/mmdetection2.6/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
#ftest = open('D:/mmdetection2.6/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
#ftrain = open('D:/mmdetection2.6/data/VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
#fval = open('D:/mmdetection2.6/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')
ftrainval = open('/data1/wanglu/datasets/insulator_defect/defect1twoclass/VOC2007/ImagesSets/main/trainval.txt', 'w')
ftest = open('/data1/wanglu/datasets/insulator_defect/defect1twoclass/VOC2007/ImagesSets/main/test.txt', 'w')
ftrain = open('/data1/wanglu/datasets/insulator_defect/defect1twoclass/VOC2007/ImagesSets/main/train.txt', 'w')
fval = open('/data1/wanglu/datasets/insulator_defect/defect1twoclass/VOC2007/ImagesSets/main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()