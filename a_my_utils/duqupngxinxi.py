from PIL import Image

# 打开图像文件
img = Image.open('/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/labeltrain/04_3420.png')

# 获取图像中的文本信息
text = img.text

# 输出文本信息
print(text)



