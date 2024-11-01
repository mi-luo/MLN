import os
from PIL import Image


def delete_images_of_size(directory, size):
    #print("*************")
    for filename in os.listdir(directory):
        #print(filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 你可以根据需要添加更多的文件类型
            file_path = os.path.join(directory, filename)

            # 打开图像并检查其大小
            img = Image.open(file_path)
            #print(img.size)
            #exit()
            if img.size == size:
                os.remove(file_path)  # 如果图像大小匹配，删除文件
                print(f"Deleted: {file_path}")

    print("Finished processing directory.")

def filter_images(imgdir, labeldir, size):

    imglist = os.listdir(imgdir)
    labellist = os.listdir(labeldir)
    namelist = []
    count = 0
    for filename in labellist:
        #print(filename)
        name = filename.split('.')[0]
        namelist.append(name)
        #print(name)
    #print(namelist)
    #exit()
    for img in imglist:
        image = img.split('.')[0]
        #print(image)
        for i in namelist:
            #print(i)
            if i == image:
                count = 1
        #print(count)
        if count == 0:  # 你可以根据需要添加更多的文件类型
            file_path = os.path.join(imgdir, img)

            os.remove(file_path)  # 如果图像大小匹配，删除文件
            print(f"Deleted: {file_path}")
        count = 0
    print("Finished processing directory.")
# 使用函数删除尺寸为1024x1024的图片
#delete_images_of_size(r"/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/data/trainannot", (1024, 1024))
filter_images(r"/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/data/train", r"/data1/wanglu/datasets/ttpla_data_original_size_v1/data_original_size/data/trainannot", (1024, 1024))