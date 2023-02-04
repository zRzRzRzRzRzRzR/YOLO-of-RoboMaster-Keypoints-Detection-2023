'''
使用前必看:
1. 本代码用来将labelme标注的json文件转换为txt文件,转换后的txt每一行的数据共有9个，分别为： 标签 + 四个点的归一化坐标（x.y）（左上，左下，右下，右上）。
2. 图像的大小和训练文件中的参数意义不用，这里的图像大小第一位是图像的宽，第二位是高度。例如一张图像为1280 x 768。那么在这里就是填写 [1280，768]。
3. 标签列表没有在参数的设定里面，如果需要更改，请更改label_list变量。
'''

import os
import json
import numpy as np
import argparse

label_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BS', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RS']  # 标签列表


@staticmethod
def run(path_json, path_txt, img_size):
    if not os.path.exists(path_txt):
        os.makedirs(path_txt)
    list_json = os.listdir(path_json)
    for cnt, json_name in enumerate(list_json):
        change_json(path_json + json_name, path_txt + json_name.replace('.json', '.txt'), img_size)


@staticmethod
def change_json(path_json, path_txt, img_size):
    rows = img_size[1]
    cols = img_size[0]
    with open(path_json, 'r', encoding='gb18030') as path_json:
        jsonx = json.load(path_json)
        with open(path_txt, 'w+') as ftxt:
            for shape in jsonx['shapes']:
                xy = np.array(shape['points'])
                label = str(label_list.index(shape['label']))
                strxy = label
                for m, n in xy:
                    strxy += ' ' + str(m / cols) + ' ' + str(n / rows)
                ftxt.writelines(strxy + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', type=str, default='', help='Location of the original labelme json datasets')
    parser.add_argument('txt_dir', type=str, default='', help='Output location of the changed txt dataset')
    parser.add_argument('img_size', nargs='+', type=int, default=[1280, 768], help='image sizes')
    args = parser.parse_args()
    run(args.json_dir, args.txt_dir, args.img_size)
    print("Finish Change format json2yolo!")
