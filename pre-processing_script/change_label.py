'''
使用前必看:
1. 如果你是按照READ.md文档的顺序标注的，那么正常你是用不到这个文件的。如果你是从其他开源站获得的数据，由于标签不对需要更改标签，那么你可以使用这
个脚本帮助修改。
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

            # # 在这里修改你想txt操作的内容
            # if num > 7 and num < 10:
            #     t = str(num - 1) + t[1:]  # 改成2加字符第二位往后
            #     lines[index] = t  # 改写lines中的内容
            if num == 1:
                t = str(0) + t[1:]  # 改成2加字符第二位往后
                lines[index] = t  # 改写lines中的内容
            if num == 8:
                t = str(1) + t[1:]  # 改成2加字符第二位往后
                lines[index] = t  # 改写lines中的内容
            #
            # if num != 15:  # 切片判断前两个字符
            #     t = ''
            # lines[index] = t
            #
            # if num > 10:  # 切片判断前两个字符
            #     t = str(num - 1) + t[2:]
            #     lines[index] = t

            # if num !=1 and num!=8:
            #     t = ''
            #     lines[index] = t

        file.close()
        t = "".join(lines)
        file = open(file_name, 'w')
        file.write(t)
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dir', type=str, default='', help='Location of path')
    args = parser.parse_args()
    change_label_name(args.path_dir)
    print("Finish change_label_id!")