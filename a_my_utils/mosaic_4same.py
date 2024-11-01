import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from xml.etree import ElementTree as ET
import cv2
import random


def load_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img

def save_image(img, path):
    img = Image.fromarray(img)
    img.save(path)

def load_label(path):
    doc = ET.parse(path)
    root = doc.getroot()
    objs = []
    for obj in root.findall("object"):
        category = obj.find("name").text
        xmin = obj.find("bndbox/xmin").text
        xmin = int(xmin)
        xmax = obj.find("bndbox/xmax").text
        xmax = int(xmax)
        ymin = obj.find("bndbox/ymin").text
        ymin = int(ymin)
        ymax = obj.find("bndbox/ymax").text
        ymax = int(ymax)
        objs.append((category, (xmin, ymin, xmax, ymax)))
    return objs

def save_label(labels, path):
    """
    labels = [
    (category, bndbox),
    (category, bndbox),
    (category, bndbox),
    ...
    ]
    """
    template, obj_template = get_template()
    root = ET.fromstring(template)
    doc = ET.ElementTree(root)
    for category, (xmin, ymin, xmax, ymax) in labels:
        obj = ET.fromstring(obj_template)
        obj.find('name').text = category
        obj.find("bndbox/xmin").text = str(xmin)
        obj.find("bndbox/xmax").text = str(xmax)
        obj.find("bndbox/ymin").text = str(ymin)
        obj.find("bndbox/ymax").text = str(ymax)
        root.append(obj)
    doc.write(path)

def get_template():
    string1 = """<annotation>
	<folder>WH_data</folder>
	<filename>DJI_0001_r.jpg</filename>
	<source>
		<database>WH Data</database>
		<annotation>WH</annotation>
		<image>flickr</image>
		<flickrid>NULL</flickrid>
	</source>
	<owner>
		<flickrid>NULL</flickrid>
		<name>WH</name>
	</owner>
	<size>
		<width>2736</width>
		<height>1824</height>
		<depth>3</depth>
	</size>
		<segmented>0</segmented>
	</annotation>
    """
    string2 = """<object>
		<name>spacer</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>311</xmin>
			<ymin>1430</ymin>
			<xmax>459</xmax>
			<ymax>1530</ymax>
		</bndbox>
	</object>
    """
    return string1, string2

def choose3patches(img, labels, prior_classes):
    h, w, _ = img.shape
    choose_num = 0
    choose_tar = []
    obj_num = len(labels)
    labels_copy = []
    if obj_num >= 3:
        for category, bndbox in labels:
            if category in prior_classes:
                choose_num += 1
                choose_tar.append(choose1patch(img, (category, bndbox)))
            else:
                labels_copy.append((category, bndbox))
        if choose_num < 3:
            random.shuffle(labels_copy)
            short = 3 - choose_num
            for i in range(short):
                choose_tar.append(choose1patch(img, labels_copy[i]))
    else:
        short = 3 - obj_num
        for i in range(short):
            pos = []
            for coor in [h, w]:
                pos.append(random.randint(0, coor//2-1))
            h_start, w_start = pos
            w_end = w_start + w//2
            h_end = h_start + h//2
            choose_tar.append((img[h_start:h_end, w_start:w_end, :] ,None, (h_start, w_start, h_end, w_end)))
        for i in range(obj_num):
            choose_tar.append(choose1patch(img, label[i]))
    choose_tar.append((img[::2, ::2, :], None, None))
    return choose_tar

def choose1patch(img, label):
    h, w, _ = img.shape
    category, bndbox = label
    xmin, ymin, xmax, ymax = bndbox
    x_center = (xmin+xmax)//2
    y_center = (ymin+ymax)//2
    x_start = x_center - w//4
    x_end = x_center + w//4
    y_start = y_center - h//4
    y_end = y_center + h//4
    if x_start < 0:
        x_end -= x_start
        x_start = 0
    if y_start < 0:
        y_end -= y_start
        y_start = 0
    if x_end > w:
        x_start = x_start - (x_end-w)
        x_end = w
    if y_end >= h:
        y_start = y_start - (y_end-h)
        y_end = h
        
    xmin = 0 if xmin-x_start < 0 else xmin-x_start
    ymin = 0 if ymin-y_start < 0 else ymin-y_start
    
    return (img[y_start:y_end, x_start:x_end, :], category, (xmin, ymin, xmax-x_start, ymax-y_start))

def merge_img(tars, h, w, c=3):
    imgs = [x[0] for x in tars]
    for index, img in enumerate(imgs):
        imgs[index] = cv2.resize(img, (w//2, h//2))
    row1 = np.concatenate(imgs[:2], axis=1)
    row2 = np.concatenate(imgs[2:], axis=1)
    new_img = np.concatenate((row1, row2), axis=0)
    return new_img

def merge_objs(tars, objs, h, w):
    new_objs = []
    for i, (img, category, bndbox) in enumerate(tars):
        if category is not None:
            if i == 0:
                new_objs.append((category, bndbox))
            elif i == 1:
                xmin, ymin, xmax, ymax = bndbox
                xmin += w//2
                xmax += w//2
                new_objs.append((category, (xmin, ymin, xmax, ymax)))
            elif i == 2:
                xmin, ymin, xmax, ymax = bndbox
                ymin += h//2
                ymax += h//2
                new_objs.append((category, (xmin, ymin, xmax, ymax)))
    for category, bndbox in objs:
        xmin, ymin, xmax, ymax = bndbox
        xmin = xmin // 2
        xmax = xmax // 2
        ymin = ymin // 2
        ymax = ymax // 2
        xmin += w//2
        xmax += w//2
        ymin += h//2
        ymax += h//2
        new_objs.append((category, (xmin, ymin, xmax, ymax)))
    return new_objs

if __name__ == "__main__":
    img_dir = r"F:\dataset\wls_data\images"
    xml_dir = r"F:\dataset\wls_data\xml"
    prior_classes = ['plate']
    img_list = os.listdir(img_dir)
    for index, img_path in enumerate(img_list):
        img = load_image(os.path.join(img_dir, img_path))
        h, w, _ = img.shape
        xml_path = os.path.join(xml_dir, img_path.replace(".jpg", ".xml"))
        objs = load_label(xml_path)
        tars = choose3patches(img, objs, prior_classes)
        new_img = merge_img(tars, h, w)
        new_objs = merge_objs(tars, objs, h, w)
        save_image(new_img, "{}.jpg".format(index))
        save_label(new_objs, "{}.xml".format(index))