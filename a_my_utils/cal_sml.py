"""

read xml file and statistics the number of small &middle &large object

"""

import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse

# file road
xml_ano = r'/data1/wanglu/datasets/insulator_haze_oneclass/nohaze/VOC2007/Annotations'
xml_list = os.listdir(xml_ano)
num_1 = 0
num_2 = 0
num_3 = 0
num_4 = 0
num_5 = 0
num_6 = 0
num_7 = 0

for xml_pa in xml_list:
    xml_path = xml_ano + '/' + xml_pa
    domTree = parse(xml_path)
    rootNode = domTree.documentElement
    # print(rootNode.nodeName)

    # get the name of object content
    nodes = rootNode.getElementsByTagName("object")
    for node in nodes:
        # enter the name is bndbox content
        bndbox = node.getElementsByTagName("bndbox")[0]

        index = bndbox.getElementsByTagName("xmin")[0]
        xmin = index.childNodes[0].data

        index = bndbox.getElementsByTagName("ymin")[0]
        ymin = index.childNodes[0].data

        index = bndbox.getElementsByTagName("xmax")[0]
        xmax = index.childNodes[0].data

        index = bndbox.getElementsByTagName("ymax")[0]
        ymax = index.childNodes[0].data

        mult = (int(xmax) - int(xmin)) * (int(ymax) - int(ymin))
        # print(mult)

        # you can change the request
        if mult < 256:
            num_1 = num_1 + 1
        elif 256 <= mult <= 1024:
            num_2 = num_2 + 1
        elif 1024 <= mult <= 4096:
            num_3 = num_3 + 1
        elif 4096 <= mult <= 9216:
            num_4 = num_4 + 1
        elif 9216 <= mult <= 16384:
            num_5 = num_5 + 1
        elif 16384 <= mult <= 65536:
            num_6 = num_6 + 1
        else:
            num_7 = num_7 + 1

print("num 0-16 :", num_1)
print("num 16-32:", num_2)
print("num 32-64:", num_3)
print("num 64-96:", num_4)
print("num 96-128:", num_5)
print("num 128-256:", num_6)
print("num 256-512:", num_7)
print("num total:", num_1+num_2+num_3+num_4+num_5+num_6+num_7)

