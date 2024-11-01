import os

# 定义要修改图片名称的文件夹路径
folder_path = '/data1/wanglu/datasets/insulator_haze_oneclass/Haze/VOC2012/Annotations'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 如果文件是图片文件（以.jpg、.jpeg、.png等图片格式结尾）
    if any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.xml']):
        # 获取不带扩展名的文件名（即图片名称）
        image_name = os.path.splitext(filename)[0]
        # 将指定字符替换为下划线（_）
        image_name = image_name.replace('(', '_')
        image_name = image_name.replace(')', '_')
        # 重命名图片文件
        os.rename(os.path.join(folder_path, filename),
                  os.path.join(folder_path, image_name + os.path.splitext(filename)[1]))