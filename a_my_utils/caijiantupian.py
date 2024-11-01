import cv2
import os
from xml.dom.minidom import Document
import xml.etree.ElementTree as et

category_set = ['insulator', 'spacer', 'tower', 'plate', 'damper']

path = '/data1/wanglu/datasets/plad/pladtogather481/VOC2007/cutimages/image/'  # 图像数据集的路径
#path_txt = 'labels/'  # 原数据集标签文件
path_txt = '/data1/wanglu/datasets/plad/pladtogather481/VOC2007/cutimages/annaotation/'
path_out = '/data1/wanglu/datasets/plad/pladtogather481/VOC2007/cutimages/cut/'  # 切割后数据集的标签文件存放路径

z=800
#size_w=800
ims_list=os.listdir(path)  # 所有图像文件名集合
print("ims_list: ", ims_list)
for im_list in ims_list:
    img = cv2.imread(path + '/' + im_list)
    img_size = img.shape
    name_list = []
    print(name_list)
    name = im_list[:-4]
    name_list = name.split('_')
    print(name_list)
    print(len(name_list))  # 3
    if len(name) < 3:
        continue  # 结束本次循环
    print("name_list: ", name_list)
    # y = int(name_list[1])
    print("原点纵轴坐标为:", y)
    x = int(name_list[2])
    #print("原点横轴坐标为:", x)
    xmlpath = path_txt + name_list[0] + '.xml'
    #print('原xml文件名为：', xmlpath)
    xml_outpath = path_out + name + '.xml'
    #print('分割后xml文件名为：', xml_outpath)

    # 获取原xml文件信息
    tree = et.parse(xmlpath)
    root = tree.getroot()  # 获取根节点
    pngname = root.find('filename').text

    origin_width = root.find('size').find('width').text
    origin_height = root.find('size').find('height').text
    origin_depth = root.find('size').find('depth').text
    # 创建新的xml文件
    # 1. 创建document对象
    doc = Document()
    # 2.创建annotation节点
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)  # 将annotation添加到doc对象中
    # 3. 创建folder节点，添加至annotation中
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("test")
    folder.appendChild(folder_txt)
    # 4. 创建filename节点，添加至annotation中
    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(pngname)
    filename.appendChild(filename_txt)
    # 5. 创建size节点，添加到annotation中
    size = doc.createElement('size')
    annotation.appendChild(size)
    # 6.在size节点中，添加width，height，depth属性
    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(img_size[0]))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(img_size[1]))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(origin_depth))
    depth.appendChild(depth_txt)
    with open(xml_outpath, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))


    for Object in root.findall('object'):
        label_name = Object.find('name').text
        # print(name)
        bndbox = Object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin, ymin, xmax, ymax)
        if x <= xmin <= x + z and x <= xmax <= x + z and y <= ymin <= y + z and y <= ymax <= y + z:
            print(Object.find('name').text)
            print('原标签的xmin, xmax, ymin，ymax分别为：', xmin, xmax, ymin, ymax)
            new_xmin = xmin - x
            new_xmax = xmax - x
            new_ymin = ymin - y
            new_ymax = ymax - y
            print('分割后的new_xmin, new_xmax, new_ymin，new_ymax分别为：', new_xmin, new_xmax, new_ymin, new_ymax)

            # 在新的xml文件中创建object节点
            object_new = doc.createElement("object")
            annotation.appendChild(object_new)

            # 在object节点中添加name，pose，truncated，difficult节点
            name = doc.createElement('name')
            object_new.appendChild(name)
            name_txt = doc.createTextNode(str(label_name))
            name.appendChild(name_txt)

            pose = doc.createElement('pose')
            object_new.appendChild(pose)
            pose_txt = doc.createTextNode("Unspecified")
            pose.appendChild(pose_txt)

            truncated = doc.createElement('truncated')
            object_new.appendChild(truncated)
            truncated_txt = doc.createTextNode("0")
            truncated.appendChild(truncated_txt)

            difficult = doc.createElement('difficult')
            object_new.appendChild(difficult)
            difficult_txt = doc.createTextNode("0")
            difficult.appendChild(difficult_txt)
            # 在object节点中添加bndbox节点
            bndbox = doc.createElement('bndbox')
            object_new.appendChild(bndbox)

            xmin = doc.createElement('xmin')
            bndbox.appendChild(xmin)
            xmin_txt = doc.createTextNode(str(new_xmin))
            xmin.appendChild(xmin_txt)

            ymin = doc.createElement('ymin')
            bndbox.appendChild(ymin)
            ymin_txt = doc.createTextNode(str(new_ymin))
            ymin.appendChild(ymin_txt)

            xmax = doc.createElement('xmax')
            bndbox.appendChild(xmax)
            xmax_txt = doc.createTextNode(str(new_xmax))
            xmax.appendChild(xmax_txt)

            ymax = doc.createElement('ymax')
            bndbox.appendChild(ymax)
            ymax_txt = doc.createTextNode(str(new_ymax))
            ymax.appendChild(ymax_txt)
            with open(xml_outpath, 'wb') as f:
                f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
            f.close()
