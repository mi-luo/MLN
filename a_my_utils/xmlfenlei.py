import os
import shutil


if __name__ == '__main__':
    f = open("/data/pth/wl-pth/plad/VOC2007/ImageSets/main/train.txt","r")   #存放有XML文件名字的txt

    line = f.readline()
    line = line[:-1]

    while line:
        line = f.readline()
        line = line.strip('\n')
        print(line)
        path = os.getcwd()
        new_path = "/data/pth/wl-pth/plad/VOC2007/Annotations"+line   #路径为保存XML文件的文件夹
        print(new_path)
        try:
            shutil.move(new_path, '/data/pth/wl-pth/plad/VOC2007/trainxml')   #提取后保存的位置
        except:
            print("Not find error.")
        # print(path)
    f.close()
