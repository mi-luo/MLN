import os
import glob

# 定义文件夹路径
folder_path = '/data1/wanglu/datasets/cityscapes/train_target_fake/labels/val_1_5'

# 定义需要删除的文件名模式
pattern = '*_beta_0.005.png'

# 使用glob库找到所有匹配的文件
matches = glob.glob(os.path.join(folder_path, pattern))

# 遍历匹配的文件并删除它们
for match in matches:
    try:
        os.remove(match)
        print(f"成功删除文件: {match}")
    except OSError as e:
        print(f"删除文件时出错: {match} - {e.strerror}")