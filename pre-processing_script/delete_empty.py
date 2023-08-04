'''
使用前必看:
1. 本代码用来删除部分空白txt和对应的图片，保证数据集中所有的图片都是有至少一个数据
'''
import os

# 设置labels和images文件夹路径
labels_path = ''
images_path = ''

# 遍历labels文件夹中的所有txt文件
for file in os.listdir(labels_path):
    if file.endswith('.txt'):
        file_path = os.path.join(labels_path, file)
        # 检查txt文件是否为空
        if os.path.getsize(file_path) == 0:
            # 删除空的txt文件
            os.remove(file_path)
            print(f'Deleted: {file_path}')

            # 删除对应的jpg文件
            image_file = os.path.splitext(file)[0] + '.jpg'
            image_path = os.path.join(images_path, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f'Deleted: {image_path}')
            else:
                print(f'Image not found: {image_path}')
