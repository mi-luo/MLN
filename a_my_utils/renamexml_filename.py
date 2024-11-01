# import os
# from xml.etree.ElementTree import parse, Element
# out_dir = r'F:\AllNet\test_data\newanno/'  ##这里是保存的目标文件夹
# b = os.listdir(r'F:\AllNet\test_data\anno/')
# for i in  b:
#     print(i)
#     dom = parse(r'F:\AllNet\test_data\anno/'+i)
#     root = dom.getroot()
#     print(root)
#     for obj in root.iter('anno'):
#         obj.find('filename').text = i.rstrip(".xml")+".JPG"
#         name1 = obj.find('filename').text
#         print(name1)
#     dom.write(out_dir + i, xml_declaration=True)
#      ##“Gray是我自己定义的，可以改为任意值，i为原来的名字，也可以直接修改成想要的名字”

import xml.dom.minidom
import os

path = r'/home/wanglu/datasets/insulator_all/chengjiao3twoclasses/VOC2007/Annotations'  # xml文件存放路径
sv_path = r'/home/wanglu/datasets/insulator_all/chengjiao3twoclasses/VOC2007/Annotationsnew'  # 修改后的xml文件存放路径
files = os.listdir(path)

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    names = root.getElementsByTagName('filename')
    a, b = os.path.splitext(xmlFile)  # 分离出文件名a
    for n in names:
        n.firstChild.data = a + '.jpg'
    with open(os.path.join(sv_path, xmlFile), 'w', encoding='utf-8') as fh:
        dom.writexml(fh)

