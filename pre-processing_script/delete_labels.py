'''
使用前必看:
1. 如果你是按照READ.md文档的顺序标注的，那么正常你是用不到这个文件的。
2. 如果某一类的兵种图纸发生更改，原来的数据集关于这类兵种已经无效了，请使用这个脚本批量删除所有这一个类的数据。
2.使用本脚本前请备份你的标注文件。
'''
import os
import argparse


def change_label_name(path):
    total_txt = os.listdir(path)
    for file in total_txt:
        file_name = path + '/' + file
        file = open(file_name, 'r')
        lines = file.readlines()
        for index, line in enumerate(lines):
            t = lines[index]  # 读取当前行的内容
            num = int(t[0:2])
            if num == 6 or num == 13:  # 选择要删除数据的序号
                t = ''
                lines[index] = t
        file.close()
        t = "".join(lines)
        file = open(file_name, 'w')
        file.write(t)
        file.close()
        if os.path.getsize(file.name)==0:
            os.remove(file.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dir', type=str, default='', help='Location of path')
    args = parser.parse_args()
    change_label_name(args.path_dir)
    print("Finish delete")
