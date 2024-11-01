import os

#dirName = "/data/pth/wl-pth/dataset/olddata/VOC2007/JPEGImages/"        # 图片所在路径
dirName = "/data/pth/wl-pth/dataset1/VOC2007/JPEGImages/"

total_img = os.listdir(dirName)

for filename in total_img:                  # 索引所有图像
    newname = filename
    newname = newname.split(".")            # 分离后缀
    if newname[-1] == "JPG":                # 假定原后缀为.png格式
        newname[-1] = "jpg"                 # 目标后缀为.jpg格式
        newname = str.join(".", newname)    # '.'与后缀名称相连接
        filename = dirName + filename       # 原始图片路径+名称
        newname = dirName + newname         # 新的图像路径+名称
        os.rename(filename, newname)        # 重命名
        print(newname, "图片格式修改成功")

