#修改xml文件中的目标的名字，
import os, sys
import glob
from xml.etree import ElementTree as ET

# 批量读取Annotations下的xml文件
# per=ET.parse(r'C:\Users\rockhuang\Desktop\Annotations\000003.xml')
#xml_dir = r'/data/pth/wl-pth/firstsandxuexiaoandnet8to2/VOC2007/annos'
xml_dir = r'/data1/wanglu/datasets/insulator_defect/defect2/annotations'
xml_list = glob.glob(xml_dir + '/*.xml')
for xml in xml_list:
    print(xml)
    per = ET.parse(xml)
    p = per.findall('/object')

    for oneper in p:  # 找出person节点
        #child = oneper.getchildren()[0]  # 找出person节点的子节点
        child = list(oneper)[0]
        if child.text == 'defect':   #需要修改的名字
            child.text = 'defective_insulator'    #修改成什么名字

    per.write(xml)
    print(child.tag, ':', child.text)