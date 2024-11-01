# # -- coding: utf-8 --
# # 生成train.txt文件
# import os
#
#
# def file_name(file_dir):
#     L = []
#     for root, dirs, files in os.walk(file_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == '.jpg':
#                 # L.append(os.path.join(root, file))
#                 file_name = file[0:-4]  #+ '.jpg'  # 去掉.txt
#                 L.append(file_name)
#     return L
#
#
# label_folder = '/data1/wanglu/datasets/plad/plad73/VOC2012/ImageSets/main/trainxml'
# val_file = '/data1/wanglu/datasets/plad/plad73/VOC2012/ImageSets/main/train.txt'
#
# txt_name = file_name(label_folder)
#
# with open(val_file, 'w') as f:
#     for i in txt_name:
#         f.write('{}\n'.format(i))
# f.close()

# -*- coding:utf-8 -*-
import os
# 图片地址
data_base_dir = r'/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/imageval'
file_list = [] #建立列表，用于保存图片信息
# txt文件地址
write_file_name = r'/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/ImageSets/val.txt'
write_file = open(write_file_name, "w") #以只写方式打开write_file_name文件
for file in os.listdir(data_base_dir): #file为current_dir当前目录下图片名
    if file.endswith(".jpg"): #如果file以jpg结尾
        write_name = file #图片路径 + 图片名 + 标签

        write_pre = write_name.split(".")[0]

        # file_list.append(write_name) #将write_name添加到file_list列表最后
        file_list.append(write_pre)
        # sorted(file_list) #将列表中所有元素随机排列
        number_of_lines = len(file_list) #列表中元素个数
#将图片信息写入txt文件中
sorted(file_list)
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')
#关闭文件
write_file.close()




