import os
 
# 定义需要重命名的文件夹路径和统一的前缀名
folder_path = "/data1/wanglu/datasets/insulator_defect/defect1/Normal_Insulators/labels"
prefix = "D12_"
 
# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)
 
# 遍历所有文件，进行重命名
for file_name in file_names:
    # 获取文件路径和扩展名
    file_path = os.path.join(folder_path, file_name)
    ext = os.path.splitext(file_name)[1]
 
    # 新文件名为前缀名 + 原文件名
    new_file_name = prefix + file_name
 
    # 重命名文件
    os.rename(file_path, os.path.join(folder_path, new_file_name))