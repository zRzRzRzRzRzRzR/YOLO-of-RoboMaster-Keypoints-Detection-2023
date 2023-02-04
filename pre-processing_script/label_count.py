'''
使用前必看:
1. 本代码将所有的标签列出，输入参数为所有标签的label文件地址
'''

import os

import argparse


def label_count(txt_path):
    class_name = ['B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BS', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RS']
    class_num = 14  # 样本类别数
    class_list = [i for i in range(class_num)]
    class_num_list = [0 for i in range(class_num)]
    labels_list = os.listdir(txt_path)
    for i in labels_list:
        file_path = os.path.join(txt_path, i)
        file = open(file_path, 'r')
        file_data = file.readlines()
        for every_row in file_data:
            class_val = every_row.split(' ')[0]
            class_ind = class_list.index(int(class_val))
            class_num_list[class_ind] += 1
        file.close()
    for i in range(class_num):
        print(class_name[i], class_num_list[i])
    print('total:', sum(class_num_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dir', type=str, default='', help='Location of path')
    args = parser.parse_args()
    label_count(args.path_dir)
    print('Finish count labels！')
