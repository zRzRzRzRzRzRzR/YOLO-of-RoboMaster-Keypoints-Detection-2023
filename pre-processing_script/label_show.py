'''
使用前必看:
1. 此代码的作用是将数据集中的每一行数据通过描点的方式可视化在图像上。
2. 前面5位数是labelimg数据集，将会通过随机颜色的方框展现在图像上，后8位是labelimg的关键点数据集，将会通过绿色点展示在图像上。
3. 样例图label_show_sample.jpg 是当你正确标注后，展现在图像上的效果。
'''

import argparse
import cv2
import random


@staticmethod
def plot_one_box_landmarks(x, y, img, color=None, label=None, landmarks=[], line_thickness=3):
    # Plots one bounding box on image img
    h, w, c = img.shape
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(y[0])), (int(x[3]), int(y[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl // 3, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=max(tl // 4, 0.5), thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    landmarkspoints = []

    # print('\n')
    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)

        # print('第{}个点的坐标:({},{})'.format(i, point_x, point_y), end='\n')
        point_final = (point_x, point_y)

        print("第{}个点的坐标是：{}".format(i,point_final),end='\n')
        cv2.circle(img, point_final, tl, (0, 255, 0), -1)
        cv2.putText(img, str(i), point_final, 0, tl / 5, [0, 24, 26], thickness=tf, lineType=cv2.LINE_AA)
        landmarkspoints.append([point_x, point_y])


@staticmethod
def xywh2xyxy(img, xywh):
    h, w, c = img.shape
    x = [0] * 13
    y = [0] * 13
    x[0] = xywh[1] * w - 0.5 * xywh[3] * w
    y[0] = xywh[2] * h - 0.5 * xywh[4] * h
    x[1] = xywh[1] * w - 0.5 * xywh[3] * w
    y[1] = xywh[2] * h + 0.5 * xywh[4] * h
    x[2] = xywh[1] * w + 0.5 * xywh[3] * w
    y[2] = xywh[2] * h - 0.5 * xywh[4] * h
    x[3] = xywh[1] * w + 0.5 * xywh[3] * w
    y[3] = xywh[2] * h + 0.5 * xywh[4] * h
    landmarks = xywh[5:]
    names = ['B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BS', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RS']
    label = f'{names[int(xywh[0])]}'
    plot_one_box_landmarks(x, y, img, color=None, label=label, landmarks=landmarks)
    return img


@staticmethod
def read_txt(txt_path):
    with open(str(txt_path), 'r', encoding='utf-8') as f:
        data = list(map(lambda x: x.rstrip('\n'), f))
    return data


def deal_txt(pic1, txt_path):
    pic = cv2.imread(pic1)
    label_list = read_txt(txt_path)
    for line in label_list:
        for i, one_line in enumerate([line]):
            txt_info = one_line.split(' ')
        txt_info = [float(i) for i in txt_info]
        pic = xywh2xyxy(pic, txt_info)
    cv2.imwrite('label_show_sample.jpg', pic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, default='', help='image dir')
    parser.add_argument('txt_path', type=str, default='', help='txt dir')
    args = parser.parse_args()
    deal_txt(args.img_path, args.txt_path)
