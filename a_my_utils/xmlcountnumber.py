"""

read xml file and statistics the number of small &middle &large object

"""

import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse

# file road
xml_ano = r'/data1/wanglu/datasets/plad/plad73/VOC2012/Annotations'
xml_list = os.listdir(xml_ano)
num_s = 0
num_m = 0
num_l = 0

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
        if mult < 21292.6464:
            num_s = num_s + 1
        elif 1362729.37<= mult <=5450917.48:
            num_m = num_m + 1
        else:
            num_l = num_l + 1

print("small package num :", num_s)
print("middle package num :", num_m)
print("large package num :", num_l)
