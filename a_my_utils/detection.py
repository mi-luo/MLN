from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  # , show_result_pyplot
import cv2


def show_result_pyplot(model, img, result, score_thr=0.3):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    return img
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()


def main():
    # config文件  .py
    config_file = '/home/wanglu/mmdetection-2.17.0/a_myrunresults/faster220701/faster_rcnn_r50_fpn_1x_coco.py'
    # 训练好的模型  .pth
    checkpoint_file = '/home/wanglu/mmdetection-2.17.0/a_myrunresults/faster220701//epoch_11.pth'

    # model = init_detector(config_file, checkpoint_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 图片路径
    img_dir = '/data/pth/wl-pth/xuexiaofirst/JPEGImages/'
    # 检测后存放图片路径
    out_dir = '/data/pth/wl-pth/xuexiaofirst/JPEGImagesout_xuexiao/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 测试集的图片名称txt
    test_path = '/data/pth/wl-pth/xuexiaofirst/xuexiaofangzhen.txt'
    fp = open(test_path, 'r')
    test_list = fp.readlines()

    count = 0
    imgs = []
    for test in test_list:
        test = test.replace('\n', '')
        name = img_dir + test + '.jpg'

        count += 1
        print('model is processing the {}/{} images.'.format(count, len(test_list)))
        # result = inference_detector(model, name)
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')
        result = inference_detector(model, name)
        img = show_result_pyplot(model, name, result, score_thr=0.25)
        cv2.imwrite("{}/{}.jpg".format(out_dir, test), img)


if __name__ == '__main__':
    main()