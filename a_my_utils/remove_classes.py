import os
from xml.etree import ElementTree as ET

def remove_cls(root, remove_cls):
    obj_list = root.findall("object")
    for obj in obj_list:
        if obj.find("name").text in remove_cls:
            root.remove(obj)

def check_obj(root):
    if root.findall('object') is None:
        return False
    else:
        return True

def get_all_cls(root):
    cls_list = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        cls_list.append(cls)
    return cls_list

if __name__ == "__main__":
    src_dir = r"/data1/wanglu/datasets/insulator_defect/defect2twoclass/VOC2007/Annotations"         # 需要修改的标签文件夹
    dst_dir = r"/data1/wanglu/datasets/insulator_defect/defect2twoclass/VOC2007/Annotationsnew" # 修改后的标签保存的文件夹

    remove_classes = {'part_defective_insulator'} # 要删掉的类别名

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_list = os.listdir(src_dir)

    cls_set = set()
    for i, src_file in enumerate(src_list):
        print("NO.{} 正在统计{}文件中的类别。".format(i, src_file))
        src_path = os.path.join(src_dir, src_file)
        src_xml = ET.parse(src_path)
        root = src_xml.getroot()

        cls_list = get_all_cls(root)

        cls_set.update(cls_list)
    print("当前数据集中有{}类: {}".format(len(cls_set), cls_set))

    print("将要删去{}类: {}".format(len(remove_classes), remove_classes))

    for i, src_file in enumerate(src_list):
        print("NO.{} 正在处理{}文件。".format(i, src_file))
        src_path = os.path.join(src_dir, src_file)
        src_xml = ET.parse(src_path)
        root = src_xml.getroot()

        remove_cls(root, remove_cls=remove_classes)
        dst_path = os.path.join(dst_dir, src_file)
        if check_obj(root) is True:
            src_xml.write(dst_path)