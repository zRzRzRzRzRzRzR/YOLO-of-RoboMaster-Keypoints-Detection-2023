'''
使用前必看:
下面展示了如何将LabelStudio导出的json数据集格式转换为我们需要的yolo格式。
这套脚本主要是配合有时候LabelStudio导出的yolo格式不正常，我们需要用其yolo格式导出的images，并用这个脚本生成labels文件夹
'''

import json
import os
import argparse


def convert_labels(json_file_path, output_folder_path):
    # 定义类别名称和编号
    class_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BS', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RS', 'BB', 'RB']
    class_ids = {class_name: i for i, class_name in enumerate(class_names)}

    # 读取JSON文件
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

        # 遍历每个标注框
        for annotation in data:
            image_filename = annotation['file_upload']
            try:
                image_width = annotation['annotations'][0]['result'][0]['original_width']
                image_height = annotation['annotations'][0]['result'][0]['original_height']
            except:
                continue  # 空标签
            # 创建保存标签的文件夹
            os.makedirs(output_folder_path, exist_ok=True)

            # 创建txt文件路径
            txt_filename = os.path.splitext(image_filename)[0] + '.txt'
            txt_filepath = os.path.join(output_folder_path, txt_filename)

            # 打开txt文件以写入标签
            with open(txt_filepath, 'w') as txt_file:
                # 遍历每个标注框的结果
                for result in annotation['annotations'][0]['result']:
                    x = result['value']['x']
                    y = result['value']['y']
                    width = result['value']['width']
                    height = result['value']['height']
                    class_name = result['value']['rectanglelabels'][0]
                    class_id = class_ids[class_name]

                    # 计算标签框的中心坐标和归一化尺寸
                    x_center = x + width / 2
                    y_center = y + height / 2
                    x_normalized = x_center / image_width
                    y_normalized = y_center / image_height
                    width_normalized = width / image_width
                    height_normalized = height / image_height
                    # 将标签写入txt文件
                    txt_file.write(f"{class_id} {x_normalized} {y_normalized} {width_normalized} {height_normalized}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON labels to YOLO format")
    parser.add_argument("json_file_path", type=str, help="Path to the JSON file")
    parser.add_argument("output_folder_path", type=str, help="Path to the output folder")
    args = parser.parse_args()

    convert_labels(args.json_file_path, args.output_folder_path)
