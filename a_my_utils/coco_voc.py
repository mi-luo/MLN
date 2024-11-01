# -*- coding: utf-8 -*-
"""
Created on 2022/02/25 10:00
@author: TFX
"""
import os
import json

path = r"/home/wanglu/anaconda/data/anno/annotations.json"
file = open(path, "rb")
data = json.load(file)
img_list = data["images"]
annotations_list = data["annotations"]

new_xml = r"Annotations"
if not os.path.isdir(new_xml):
    os.makedirs(new_xml)

for i in img_list:
    img_name = i["file_name"].split("/")[1]
    width, height = i["width"], i["height"]

    xml_name = img_name.split('.')[0]
    xml_file = open((new_xml + '\\' + xml_name + '.xml'), 'w')

    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>citysperson</folder>\n')
    xml_file.write('    <filename>' + str(img_name) + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    for j in annotations_list:
        if i['id'] == j['image_id']:
            x, y, w, h = j['bbox']

            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + 'person' + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(x) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(y) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(x + w) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(y + h) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
    xml_file.write('</annotation>\n')
