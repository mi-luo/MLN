import os
import shutil
import numpy as np
import cv2
from PIL import Image

# json文件展开（转换）
json_folder = '/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/json/'
json_name = os.listdir(json_folder)
os.system("activate labelme")
# for i in range(len(json_name)):
#     if (os.path.splitext(json_name[i])[1] == ".json"):
#         json_path = json_folder + json_name[i]
#         print(json_path)
        #os.system("labelme_json_to_dataset " + json_path)


# json到png的批量命名/转移
png_folder = '/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/png/'
os.makedirs(png_folder, exist_ok=True)
i = 1
for name in json_name:
    if os.path.isdir(json_folder + name):
        # if i < 10:  # 0-9
        #     newname = "00" + str(i) + '.png'
        # elif i > 9 and i < 100:  # 10-100
        #     newname = "0" + str(i) + '.png'
        # elif i > 99 and i < 1000:  # 100-1000
        #     newname = str(i) + '.png'

        # i += 1
        # os.chdir(json_folder + name)
        # old_name = 'label.png'
        # os.rename(old_name, newname)
        print(name[:-5])

        old_name = os.path.join(json_folder, name, 'label.png')
        new_name = os.path.join(png_folder, name[:-5]+'.png')
        #os.rename(old_name, newname)
        #shutil.move(json_folder + name + "/" + newname, png_folder)
        shutil.copy(old_name, new_name)
        #exit()
# 改像素值
# png_name = os.listdir(png_folder)
# for name in png_name:
#     data_source = cv2.imread(png_folder + '/' + name)
#     data = np.array(data_source)
#     img_path = png_folder + '/' + name
#     for i in range(data[:, :, 0].shape[0]):
#         for j in range(data[:, :, 0].shape[1]):
#             if data[:, :, 2][i][j] > 0:
#                 data[:, :, 2][i][j] = 255  # Red
#                 data[:, :, 1][i][j] = 255  # Green
#                 data[:, :, 0][i][j] = 255  # Blue
#     cv2.imwrite(img_path, data)
#     png = Image.open(img_path).convert('L').save(img_path)