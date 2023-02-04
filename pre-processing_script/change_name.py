'''
使用前必看:
1. 本代码用来重新批量命名图片和标签。默认状态下是根据命名方式为四位数的文件进行命名，例如将9000.png更改为1.png。
2. 如果你的原始图片不符合（1）中的命名方式，可以更改 file_list中的key参数进行修改。
3. 本代码一次只能处理一个文件夹。
'''

import os
import re
import sys
import argparse


@staticmethod
def rename(path, start_id):
    file_list = sorted(os.listdir(path), key=lambda x: int(x[:-4]))  # 在这里修改排序规则
    os.chdir(path)
    for fileName in file_list:
        pat = ".+\.(jpg|png|gif|py|txt)"
        pattern = re.findall(pat, fileName)
        os.rename(fileName, (str(start_id) + '.' + pattern[0]))
        start_id = start_id + 1
    os.chdir(os.getcwd())
    sys.stdin.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dir', type=str, default='', help='Location of path')
    parser.add_argument('--start_id', type=int, default=1, help='start sort with the id')
    args = parser.parse_args()
    rename(args.path_dir, args.start_id)
    print("Finish Rename!")
