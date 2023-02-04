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
def delete_useless(path, start_id, end_id):
    path1 = '/media/zr/Data/RoboMaster_data/dataset/XJTLU_2022_radar_ALL/labels'
    path2 = '/media/zr/Data/RoboMaster_data/dataset/XJTLU_2022_radar_ALL/images'
    start_id = 1
    end_id = 6774
    while start_id <= end_id:
        file_checktxt = str(path1) + '/' + str(start_id) + '.txt'
        file_checkimage = str(path2) + '/' + str(start_id) + '.jpg'
        if os.path.exists(file_checktxt):
            if os.path.getsize(file_checktxt):
                pass
            else:
                pass
                # print(start_id, "Empty!")
        else:
            if os.path.exists(file_checkimage):
                print(start_id, "Not exit!")
                os.remove(file_checkimage)
        start_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type=str, default='', help='Location of path')
    parser.add_argument('--start_id', type=int, default=1, help='Location of path')
    parser.add_argument('--end_id', type=int, default=1, help='Location of path')
    args = parser.parse_args()
    delete_useless(args.path_dir, args.start_id, args.end_id)
    print("Finish delete!")
