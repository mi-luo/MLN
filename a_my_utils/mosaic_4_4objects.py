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
    # path = r'/data1/wanglu/datasets/plad/plad481mosiac4insulator/images/{}'.format(path)
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
    # path = r'/data1/wanglu/datasets/plad/plad481mosiac4insulator/xml/{}'.format(path)
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

    patche_num = 4

    if obj_num >= patche_num:
        for category, bndbox in labels:
            if category in prior_classes:
                print('ori plate', bndbox)
                choose_num += 1
                choose_tar.append(choose1patch(img, (category, bndbox), labels))
            else:
                labels_copy.append((category, bndbox))
        if choose_num < patche_num:
            random.shuffle(labels_copy)
            short = patche_num - choose_num
            for i in range(short):
                choose_tar.append(choose1patch(img, labels_copy[i], labels))
    else:
        short = patche_num - obj_num
        for i in range(short):
            pos = []
            for coor in [h, w]:
                pos.append(random.randint(0, coor // 2 - 1))
            h_start, w_start = pos
            w_end = w_start + w // 2
            h_end = h_start + h // 2
            choose_tar.append((img[h_start:h_end, w_start:w_end, :], None, (h_start, w_start, h_end, w_end)))
        for i in range(obj_num):
            choose_tar.append(choose1patch(img, labels[i], labels))
    # choose_tar.append((img[::2, ::2, :], None, None))
    return choose_tar


def box_in_box(o_bndbox, t_bndbox):
    o_xmin, o_ymin, o_xmax, o_ymax = o_bndbox
    t_xmin, t_ymin, t_xmax, t_ymax = t_bndbox
    if t_xmin < o_xmin and o_xmax < t_xmax and t_ymin < o_ymin and o_ymax < t_ymax:
        return True
    else:
        return False


def xy_in_box(o_bndbox, t_bndbox):
    o_xmin, o_ymin, o_xmax, o_ymax = o_bndbox
    ox_center = (o_xmin + o_xmax) / 2
    oy_center = (o_ymin + o_ymax) / 2
    t_xmin, t_ymin, t_xmax, t_ymax = t_bndbox
    if t_xmin < ox_center and ox_center < t_xmax and t_ymin < oy_center and oy_center < t_ymax:
        return True
    else:
        return False


def clap(a, s, e):
    if a < s:
        a = s
    if a > e:
        a = e
    return a


def choose1patch(img, label, all_labels):
    all_labels = all_labels.copy()
    all_labels.remove(label)

    # all_labels = [('plate', (1367, 940, 1443, 972)), ('tower', (1332, 704, 1432, 856))]
    # all_labels = [('tower', (1332, 704, 1432, 856))]
    choose_labels = []

    h, w, _ = img.shape
    t_category, t_bndbox = label
    t_xmin, t_ymin, t_xmax, t_ymax = t_bndbox

    t_x_center = (t_xmin + t_xmax) // 2
    t_y_center = (t_ymin + t_ymax) // 2
    t_x_start = t_x_center - w // 4
    t_x_end = t_x_center + w // 4
    t_y_start = t_y_center - h // 4
    t_y_end = t_y_center + h // 4
    if t_x_start < 0:
        t_x_end -= t_x_start
        t_x_start = 0
    if t_y_start < 0:
        t_y_end -= t_y_start
        t_y_start = 0
    if t_x_end > w:
        t_x_start = t_x_start - (t_x_end - w)
        t_x_end = w
    if t_y_end >= h:
        t_y_start = t_y_start - (t_y_end - h)
        t_y_end = h

    t_xmin = 0 if t_xmin - t_x_start < 0 else t_xmin - t_x_start
    t_ymin = 0 if t_ymin - t_y_start < 0 else t_ymin - t_y_start

    choose_labels.append((t_category, (t_xmin, t_ymin, t_xmax - t_x_start, t_ymax - t_y_start)))

    print('1111111', t_y_start, t_y_end, t_x_start, t_x_end)
    print(t_x_start, t_y_start, t_x_end, t_y_end)

    # 其他标签
    for category, bndbox in all_labels:

        xmin, ymin, xmax, ymax = bndbox
        new_xmin = xmin - t_x_start
        new_xmax = xmax - t_x_start
        new_ymin = ymin - t_y_start
        new_ymax = ymax - t_y_start

        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2
        # # 目標在 patch 的范围内
        print(new_xmin, new_ymin, new_xmax, new_ymax)
        # if box_in_box(bndbox, (t_x_start, t_y_start, t_x_end, t_y_end)):
        if xy_in_box(bndbox, (t_x_start, t_y_start, t_x_end, t_y_end)):
            print('222222222222', xmin, ymin, xmax, ymax)
            new_xmin = clap(new_xmin, 0, t_x_end - t_x_start)
            new_xmax = clap(new_xmax, 0, t_x_end - t_x_start)
            new_ymin = clap(new_ymin, 0, t_y_end - t_y_start)
            new_ymax = clap(new_ymax, 0, t_y_end - t_y_start)
            choose_labels.append((category, (new_xmin, new_ymin, new_xmax, new_ymax)))

        #     new_xmin = 0 if xmin - t_x_start < 0 else xmin - t_x_start
        #     new_ymin = 0 if t_ymin - t_y_start < 0 else t_ymin - t_y_start
        #     choose_labels.append((category, (new_xmin, new_ymin, xmax - t_x_start, t_ymax - t_y_start)))
        #     # choose_labels.append((t_category, (t_xmin, t_ymin, t_xmax - t_x_start, t_ymax - t_y_start)))

    print('\nall_labels', len(all_labels), all_labels)

    print('\nchoose_labels', len(choose_labels), choose_labels)
    # print(choose_labels[0])
    # print(choose_labels[1])
    # print(choose_labels[2])
    # print(choose_labels[3])
    # print(choose_labels[4])
    # _img = img[t_y_start:t_y_end, t_x_start:t_x_end, :]
    # for c, box in choose_labels:
    #     xmin, ymin, xmax, ymax = box
    #     print(c, xmin, ymin, xmax, ymax)
    #
    #     cv2.rectangle(_img, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)

    # cv2.rectangle(_img, (646, 440),(722, 472), (255, 0, 255), 2)

    # cv2.imshow('img', _img)
    # cv2.waitKey(0)
    # exit()
    # cv2.imshow()
    # return (img[y_start:y_end, x_start:x_end, :], category, (xmin, ymin, xmax - x_start, ymax - y_start))
    return (img[t_y_start:t_y_end, t_x_start:t_x_end, :], choose_labels)


def merge_img(tars, h, w, c=3):
    imgs = [x[0] for x in tars]
    for index, img in enumerate(imgs):
        imgs[index] = cv2.resize(img, (w // 2, h // 2))
    row1 = np.concatenate(imgs[:2], axis=1)
    row2 = np.concatenate(imgs[2:], axis=1)
    new_img = np.concatenate((row1, row2), axis=0)
    return new_img


def merge_objs(tars, objs, h, w):
    new_objs = []
    # for i, (img, category, bndbox) in enumerate(tars):
    #     if category is not None:
    #         if i == 0:
    #             new_objs.append((category, bndbox))
    #         elif i == 1:
    #             xmin, ymin, xmax, ymax = bndbox
    #             xmin += w // 2
    #             xmax += w // 2
    #             new_objs.append((category, (xmin, ymin, xmax, ymax)))
    #         elif i == 2:
    #             xmin, ymin, xmax, ymax = bndbox
    #             ymin += h // 2
    #             ymax += h // 2
    #             new_objs.append((category, (xmin, ymin, xmax, ymax)))
    for i, _tars in enumerate(tars):

        if _tars[1] is None:
            continue
        img, labels = _tars

        for category, bndbox in labels:

            if category is not None:
                if i == 0:
                    new_objs.append((category, bndbox))
                elif i == 1:
                    xmin, ymin, xmax, ymax = bndbox
                    xmin += w // 2
                    xmax += w // 2
                    new_objs.append((category, (xmin, ymin, xmax, ymax)))
                elif i == 2:
                    xmin, ymin, xmax, ymax = bndbox
                    ymin += h // 2
                    ymax += h // 2
                    new_objs.append((category, (xmin, ymin, xmax, ymax)))
    # print(objs)
    # print(len(objs))
    for category, bndbox in objs:
        xmin, ymin, xmax, ymax = bndbox
        xmin = xmin // 2
        xmax = xmax // 2
        ymin = ymin // 2
        ymax = ymax // 2
        xmin += w // 2
        xmax += w // 2
        ymin += h // 2
        ymax += h // 2
        new_objs.append((category, (xmin, ymin, xmax, ymax)))
    return new_objs


if __name__ == "__main__":
    #pass
    # img_dir = r"/data1/wanglu/datasets/plad/new/images"
    # xml_dir = r"/data1/wanglu/datasets/plad/new/labels"
    # prior_classes = ['insulator']
    # img_list = os.listdir(img_dir)
    # for index, img_path in enumerate(img_list):
    #     img = load_image(os.path.join(img_dir, img_path))
    #     h, w, _ = img.shape
    #     xml_path = os.path.join(xml_dir, img_path.replace(".jpg", ".xml"))
    #     objs = load_label(xml_path)
    #     tars = choose3patches(img, objs, prior_classes)
    #     new_img = merge_img(tars, h, w)
    #     new_objs = merge_objs(tars, objs, h, w)
    #     save_image(new_img, "{}.jpg".format(index))
    #     save_label(new_objs, "{}.xml".format(index))

    index = 'test'
    img_path = 'DJI_0001_r.jpg'
    xml_path = 'DJI_0001_r.xml'
    prior_classes = ['plate']

    img = load_image(img_path)
    h, w, _ = img.shape

    objs = load_label(xml_path)
    print(objs)
    tars = choose3patches(img, objs, prior_classes)
    print(len(tars))
    print(tars[0][0].shape)
    # print(tars[0])
    print(tars[0][1])
    # exit()

    new_img = merge_img(tars, h, w)
    new_objs = merge_objs(tars, objs, h, w)
    save_image(new_img, "{}.jpg".format(index))
    save_label(new_objs, "{}.xml".format(index))
