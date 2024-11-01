import os
import glob
import json

# 设定文件夹路径
folder_path = '/data1/wanglu/datasets/insulator_haze_oneclass/Haze/v5/train/labels'

# 遍历文件夹寻找所有的txt文件
txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

# 遍历所有的txt文件，读取并存储内容
txt_contents = []
for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        txt_contents.append(json.dumps({os.path.basename(txt_file): f.read()}))  # 将文件名和内容作为一个字典存储

# 将内容列表转换为json格式，并写入到一个新的json文件中
with open('train.json', 'w') as f:
    json.dump(txt_contents, f)