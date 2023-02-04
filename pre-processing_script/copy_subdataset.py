'''
使用前必看:
1. 三个参数分别是：总数据集位置 复制后新数据集的位置 提取的编号
编号的对应关系：
1-5 分别对应1-5红蓝装甲板
6 对应红蓝前哨站
7 对应红蓝哨兵
'''
import os
import argparse
import shutil
import warnings


def change_label_name(path, copy_path, id):
    if (int(id) > 7 or int(id) < 1):
        warnings.warn(("please write in format!"))
        print("Fail!")
        return
    labels_path = path + '/labels/'
    images_path = path + '/images/'
    labels_copy_path = copy_path + '/labels/'
    images_copy_path = copy_path + '/images/'
    if not os.path.exists(copy_path):
        warnings.warn("Could no find saving_path,Creat one!")
    if not os.path.exists(labels_copy_path):
        os.makedirs(labels_copy_path)
    if not os.path.exists(images_copy_path):
        os.makedirs(images_copy_path)

    labels_txt = os.listdir(labels_path)
    for file in labels_txt:
        name = file[:-4]
        file_name = labels_path + '/' + file
        file = open(file_name, 'r')
        lines = file.readlines()
        for index, line in enumerate(lines):
            t = lines[index]  # 读取当前行的内容
            num = int(t[0:2])
            if num == (id - 1) or num == (id - 1 + 7):  # 选择要复制的文件序号
                shutil.copy(labels_path + name + '.txt', labels_copy_path)
                shutil.copy(images_path + name + '.jpg', images_copy_path)
    print("Finish creat subdataset!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dir', type=str, default='', help='Location of path')
    parser.add_argument('dis_dir', type=str, default='', help='Location of copy path')
    parser.add_argument('id', type=int, default=1, help='copy with label id')
    args = parser.parse_args()
    change_label_name(args.path_dir, args.dis_dir, args.id)
