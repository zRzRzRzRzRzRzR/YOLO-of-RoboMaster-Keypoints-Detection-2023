'''
使用前必看:
1. 如果你是按照READ.md文档的顺序标注的，那么正常你是用不到这个文件的，如果你很不幸，文件标注的点顺序错误了（比如标注成了右上，右下，左下，左上）。
那么你将无法合并到大数据集中，你需要在执行完 change_json 文件后执行这个文件，转换为正确的顺序。
2. 本代码一次只能更改两个点的坐标，请输入你想更换的坐标，对应的数据如下：
    左上 0 左下 1 右下 2 右上 3
    例如，你想更换左下点和右下点的坐标，那么你应该输入 2 3 或者 3 2
3. 使用该脚本之前，请备份你的标注文件，以防操作失误带来的后果。
4. 如果你的txt文件已经和labelimg的标注文件合并，请添加 --labelimg_before 参数
'''

import os
import re
import argparse


@staticmethod
def change_order(path, point, labelimg_before):
    total_txt = os.listdir(path)
    for file in total_txt:
        file_name = path + '/' + file
        file = open(file_name, 'r')
        lines = file.readlines()
        for index, line in enumerate(lines):
            t = re.findall(r"\d+\.?\d*", lines[index])
            tmp1 = t[4 * labelimg_before + 1 + 2 * point[0]]
            tmp2 = t[4 * labelimg_before + 1 + 2 * point[0] + 1]
            t[4 * labelimg_before + 1 + 2 * point[0]] = t[4 * labelimg_before + 1 + 2 * point[1]]
            t[4 * labelimg_before + 1 + 2 * point[0] + 1] = t[4 * labelimg_before + 1 + 2 * point[1] + 1]
            t[4 * labelimg_before + 1 + 2 * point[1]] = tmp1
            t[4 * labelimg_before + 1 + 2 * point[1] + 1] = tmp2
            lines[index] = t
        file.close()
        # lines列表转换为字符串放在T中
        t = "".join('%s' % id for id in lines)
        t = re.sub('[’!"#$%&\'()*:;<=>?@，,。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', "", t)

        # 打开文件写入模式，把更新后的lines写进txt文件中
        file = open(file_name, 'w')
        file.write(t)
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='', help='Location of the txt datasets')
    parser.add_argument('labelimg_before', action='store_true', default=' ', help='if labelimg before')
    parser.add_argument('points', nargs='+', type=int, default=[2, 3], help='Swift points')
    args = parser.parse_args()
    change_order(args.path, args.points, args.labelimg_before)
    print("Finish Change point order!")
