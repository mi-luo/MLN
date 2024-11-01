import cv2
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring, ElementTree

def xml_resize(xml_string, scale, img_name, lists=None, threshold=None):
    root = ET.fromstring(xml_string)
    changed_flag = False # 用于标记image_size是否修改过
    for obj in root.findall('object'):
        category = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text) - 1
        xmax = int(bndbox.find('xmax').text) - 1
        ymin = int(bndbox.find('ymin').text) - 1
        ymax = int(bndbox.find('ymax').text) - 1
        area = (xmax-xmin)*(ymax-ymin)

        if not changed_flag:
            pre,app = img_name.split(".")
            root.find("filename").text = "{}_r.{}".format(pre, app)
            width = root.find('size/width')
            height = root.find('size/height')
            width.text = str(int(width.text)//scale)
            height.text = str(int(height.text)//scale)
            changed_flag = True


        bndbox.find('xmin').text = str(xmin//scale + 1)
        bndbox.find('xmax').text = str(xmax//scale + 1)
        bndbox.find('ymin').text = str(ymin//scale + 1)
        bndbox.find('ymax').text = str(ymax//scale + 1)
    return ET.tostring(root)

def get_template():
    obj_string = '<object><name>sicktree2</name>' + \
    '<pose>Unspecified</pose><truncated>0</truncated>' + \
    '<difficult>0</difficult><bndbox><xmin>834</xmin>' + \
    '<ymin>763</ymin><xmax>868</xmax><ymax>795</ymax></bndbox></object>'
    
    root_string = '<annotation><folder>yilingtest</folder>' + \
    '<filename>p_yl_0001.jpg</filename><path>D:\\&#27979;&#35797;&#22270;' + \
    '\\yilingtest\\p_yl_0001.jpg</path><source><database>Unknown</database>' + \
    '</source><size><width>1000</width><height>1000</height><depth>3</depth></size>' + \
    '<segmented>0</segmented></annotation>'
    return obj_string, root_string


def check_bound(value, little, large):
    if value < little:
        return little, "little"
    elif value > large:
        return large, 'large'
    else:
        return value, 'equal'


def xml_split(xml_string, scale, img_name, lists=None, threshold=None):
    """
    对于在网格边界上的物体，中心点位于那个网格，则将该物体划入该网格中，
    其余网格不再考虑该物体。
    
    """
    xmls = {}
    
    key_format = "{row}_{col}"
    
    root = ET.fromstring(xml_string)
    width_t = int(root.find('size/width').text)
    height_t = int(root.find('size/height').text)
    width_s = width_t // scale
    height_s = height_t // scale
    
    obj_template, root_template = get_template()
    
    for obj in root.findall('object'):
        # 对小物体进行split操作
        category = obj.find('name').text
        xmin = obj.find('bndbox/xmin')
        xmax = obj.find('bndbox/xmax')
        ymin = obj.find('bndbox/ymin')
        ymax = obj.find('bndbox/ymax')
        
        xmin_v = int(xmin.text) - 1
        xmax_v = int(xmax.text) - 1
        ymin_v = int(ymin.text) - 1
        ymax_v = int(ymax.text) - 1
        
        x_center = (xmin_v + xmax_v)//2
        y_center = (ymin_v + ymax_v)//2
        
        area = (xmax_v - xmin_v) * (ymax_v - ymin_v)

        col = x_center // width_s
        row = y_center // height_s

        xmin_shift = xmin_v-col*width_s if xmin_v-col*width_s>=0 else 0
        xmax_shift = xmax_v-col*width_s if xmax_v-col*width_s<=width_s-1 else width_s-1
        ymin_shift = ymin_v-row*height_s if ymin_v-row*height_s>=0 else 0
        ymax_shift = ymax_v-row*height_s if ymax_v-row*height_s<=height_s-1 else height_s-1

        obj_create = ET.fromstring(obj_template)
        obj_create.find('name').text = category
        obj_create.find('bndbox/xmin').text = str(xmin_shift + 1)
        obj_create.find('bndbox/xmax').text = str(xmax_shift + 1)
        obj_create.find('bndbox/ymin').text = str(ymin_shift + 1)
        obj_create.find('bndbox/ymax').text = str(ymax_shift + 1)

        key = key_format.format(row=row, col=col)
        if xmls.get(key, None) is None:
            xml_create = ET.fromstring(root_template)
            xml_create.find('size/width').text = str(width_s)
            xml_create.find('size/height').text = str(height_s)
            pre, app = img_name.split(".")
            xml_create.find('filename').text = "{}_{}.{}".format(pre, key, app)
            xmls[key] = xml_create
        xmls[key].append(obj_create)
    for k, v in xmls.items():
        ET.indent(v)
        xmls[k] = ET.tostring(v)
    return xmls

def image_split(img, scale):
    imgs = {}
    key_format = "{row}_{col}"
    height_t, width_t = img.shape[:2]
    height_s = height_t // scale
    width_s = width_t // scale
    for row in range(scale):
        for col in range(scale):
            key = key_format.format(row=row, col=col)
            imgs[key] = img[row*height_s:(row+1)*height_s, col*width_s:(col+1)*width_s]
    return imgs

def get_prename(xml_string):
    root = ET.fromstring(xml_string)
    img_name = root.find('filename').text
    return img_name.split(".")

def is_small_object(category=None, area=None, lists=None, threshold=None):
    if lists is None and threshold is None:
        threshold = 32**2
    if lists is None:
        if area <= threshold:
            return True
        else:
            return False
    
    if category in lists:
        return True
    else:
        return False

def object_exists(xml_string):
    root = ET.fromstring(xml_string)
    if root.find('object') is None:
        return False
    else:
        return True

def split(img, xml_string, scale, img_name, imgs_to, xmls_to, lists=None, threshold=None):
    prename, appendix = get_prename(xml_string)
    img_path_format = prename + "_{key}." + appendix
    xml_path_format = prename + "_{key}.xml"
    xmls = xml_split(xml_string, scale, img_name, lists=lists, threshold=threshold)
    imgs = image_split(img, scale)
    for k, v in xmls.items():
        if object_exists(v):
            img_path = os.path.join(imgs_to, img_path_format.format(key=k))
            xml_path = os.path.join(xmls_to, xml_path_format.format(key=k))
            save_image(imgs[k], img_path)
            save_xml(v, xml_path)

def resize(img, xml_string, scale, img_name, img_to, xml_to, lists=None, threshold=None):
    img_r = cv2.resize(img, None, fx=1/scale, fy=1/scale)
    root = ET.fromstring(xml_string)
    xml_r = xml_resize(xml_string, scale, img_name, lists=lists, threshold=threshold)
    
    if object_exists(xml_r):
        save_image(img_r, img_to)
        save_xml(xml_r, xml_to)

def save_xml(root_string, filepath):
    with open(filepath, "wb") as f:
        f.write(root_string)

def save_image(img, filepath):
    cv2.imwrite(filepath, img)

def make_diretory(save_dir=None):
    if save_dir is None:
        save_dir = '/data1/wanglu/datasets/plad/plad73new/train'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    type_list = ['Resize', 'Split']
    file_type = ['Annotations', 'JPEGImages']
    for type_ in type_list:
        type_dir = os.path.join(save_dir, type_)
        if not os.path.exists(type_dir):
            os.mkdir(type_dir)
        for file_ in file_type:
            dir_ = os.path.join(type_dir, file_)
            if not os.path.exists(dir_):
                os.mkdir(dir_)
    return save_dir

if __name__ == "__main__":
    # save_dir = None
    # lists = []
    # threshold = 32**2
    
    scale_r = 2 # resize的尺度
    scale_s = 4 # split的尺度
    small_object_list = ['damper']
    # small_object_list = None
    # root = r"E:\python_workspace\data_test"
    root = r'/data1/wanglu/datasets/plad/plad73/VOC2012/ImageSets/main'
    xml_path = os.path.join(root, "trainxml")
    img_path = os.path.join(root, "trainjpg")
    #imgs_list = os.listdir(img_path)
    imgs_list = os.listdir(img_path)
    
    save_dir = make_diretory()
    
    imgs_to = os.path.join(save_dir, "Split", "JPEGImages")
    xmls_to = os.path.join(save_dir, "Split", "Annotations")
    
    img_to_pre = os.path.join(save_dir, "Resize", "JPEGImages")
    xml_to_pre = os.path.join(save_dir, "Resize", "Annotations")
    
    for img_file in imgs_list:
        prename, appendix = img_file.split(".")
        xml_file = prename + ".xml"
        xml_filename = os.path.join(xml_path, xml_file)
        img_filename = os.path.join(img_path, img_file)
        
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        root_string = ET.tostring(root)
        
        img = cv2.imread(img_filename)
        
        print("Dividing img: {}".format(img_file))
        split(img, root_string, scale_s, img_file, imgs_to=imgs_to, xmls_to=xmls_to, lists=small_object_list)
        
        img_to = os.path.join(img_to_pre, prename+"_r."+appendix)
        xml_to = os.path.join(xml_to_pre, prename+"_r.xml")
        print("Resizing img: {}".format(img_file))
        resize(img, root_string, scale_r, img_file, img_to=img_to, xml_to=xml_to, lists=small_object_list)
    print("Finished!")

