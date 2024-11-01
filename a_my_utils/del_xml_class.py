import os
import xml.etree.ElementTree as ET
import tqdm


def del_delete_eq_1(xml_path):
    # 从xml文件中读取，使用getroot()获取根节点，得到的是一个Element对象
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for object in root.findall('object'):
        deleted = str(object.find('name').text)

        if (deleted in ['plate']):
            root.remove(object)

    tree.write(xml_path)


def main():
    root_dir = "/data/pth/wl-pth/plad/VOC2007/annotations_gai/"
    xml_path_list = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]

    # 使用tqdm显示进程
    for xml in tqdm.tqdm(xml_path_list):
        del_delete_eq_1(xml)


if __name__ == '__main__':
    main()

