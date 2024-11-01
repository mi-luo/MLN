# import os
# import shutil
#
# src_folder = "/data1/wanglu/datasets/cityscapes/train_source_real/images/train"  # 替换为源文件夹路径
# dst_folder = "/data1/wanglu/datasets/cityscapes/train_source_real/images/train_new"  # 替换为目标文件夹路径
#
# # 遍历源文件夹中的所有文件
# for filename in os.listdir(src_folder):
#     # 仅处理Word文件
#     if filename.endswith(".json") :
#         # 获取不带"_leftImg8bit"的文件名
#         new_filename = filename.replace("_gtFine_polygons", "")
#         # 构建源文件路径和目标文件路径
#         src_filepath = os.path.join(src_folder, filename)
#         dst_filepath = os.path.join(dst_folder, new_filename)
#         # 重命名并移动文件到目标文件夹
#         shutil.move(src_filepath, dst_filepath)

import os

folder_path = "/data1/wanglu/datasets/cityscapes/train_target_fake/labels/val005"  # 替换为您的文件夹路径

for filename in os.listdir(folder_path):
    if filename.endswith(".png") :  # 仅处理Word文件
        new_filename = filename.replace("_leftImg8bit_foggy_beta_0.005", "")
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(os.path.join(folder_path, filename), new_filepath)