import os
import random
import shutil

# 图片库所在的目录
image_directory = "/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/image"

# 存放选取图片的目标文件夹
target_directory_1 = "/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/imagetrain"
target_directory_2 = "/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/imageval"

# 获取图片库中的所有图片文件
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if
               file.endswith(('.png', '.jpg', '.jpeg'))]

# 计算需要选取的图片数量
num_images_to_select = int(len(image_files) * 0.8)
num_images_to_select_2 = len(image_files) - num_images_to_select

# 随机选取指定数量的图片
selected_images = random.sample(image_files, num_images_to_select)
selected_images_2 = list(set(image_files) - set(selected_images))

# 创建目标文件夹（如果不存在）
os.makedirs(target_directory_1, exist_ok=True)
os.makedirs(target_directory_2, exist_ok=True)

# 将选取的图片复制到目标文件夹
for image_file in selected_images:
    shutil.copy2(image_file, target_directory_1)

for image_file in selected_images_2:
    shutil.copy2(image_file, target_directory_2)