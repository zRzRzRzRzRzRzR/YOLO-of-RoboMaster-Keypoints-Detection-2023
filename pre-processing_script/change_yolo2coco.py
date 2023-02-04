'''
使用前必看:
1. 本脚本实现将yolo.txt文件转换为coco文件。其中，0-4位分别是labelimg的数据，5位之后是labelme数据，转为关键点数据。
2. 如果你的yolo数据中有test文件，请将mode_list中设置为 'train,val,test' ，若无则设置为'train,val'。
3. 你的yolo文件夹需要满足以下格式和文件要求：

.
├── classes.txt //YOLO类别标签（每个类一行）
├── images
│   ├── test //测试图像（可选）
│   ├── train //训练图像
│   └── val //验证图像
├── labels
│   ├── test //测试标签 （可选）
│   ├── train //训练图像
│   └── val //验证图像
├── test.txt  //测试文件路径 （可选）
├── train.txt //训练文件路径
└── val.txt //验证文件路径

一份训练文件路径应该如下：
.
├── train.txt
│   ├──images/train/1.jpg
│   ├──images/train/2.jpg
│   ├──images/train/3.jpg
│   ├── ...
│   └── images/train/10000.jpg
└── val.txt //验证文件也是如此

4.生成的keypoints 为labelme对应的8个点还原归一化的值，segmentation为bbox的四个角点。
'''

import argparse
import json
import shutil
import time
import warnings
import cv2

from tqdm import tqdm
from pathlib import Path


class YOLOToCOCO(object):
    def __init__(self, data_dir):
        self.raw_data_dir = Path(data_dir)

        self.verify_exists(self.raw_data_dir / 'images')
        self.verify_exists(self.raw_data_dir / 'labels')

        save_dir_name = f'{Path(self.raw_data_dir).name}_COCO'
        self.output_dir = self.raw_data_dir.parent / save_dir_name
        self.mkdir(self.output_dir)

        self._init_json()

    def __call__(self, mode_list: list):
        if not mode_list:
            return ValueError('mode_list is empty!!')

        for mode in mode_list:
            # Read the image txt.
            txt_path = self.raw_data_dir / f'{mode}.txt'
            self.verify_exists(txt_path)
            img_list = self.read_txt(txt_path)
            if mode == 'train':
                img_list = self.append_bg_img(img_list)

            # Create the directory of saving the new image.
            save_img_dir = self.output_dir / f'{mode}2017'
            self.mkdir(save_img_dir)

            # Generate json file.
            anno_dir = self.output_dir / "annotations"
            self.mkdir(anno_dir)

            save_json_path = anno_dir / f'instances_{mode}2017.json'
            json_data = self.convert(img_list, save_img_dir, mode)

            self.write_json(save_json_path, json_data)

    def _init_json(self):
        classes_path = self.raw_data_dir / 'classes.txt'
        self.verify_exists(classes_path)
        self.categories = self._get_category(classes_path)

        self.type = 'instances'
        self.annotation_id = 1

        self.cur_year = time.strftime('%Y', time.localtime(time.time()))
        self.info = {
            'year': int(self.cur_year),
            'version': '1.0',
            'description': 'For object detection',
            'date_created': self.cur_year,
        }

        self.licenses = [{
            'id': 1,
            'name': 'Apache License v2.0',
            'url': 'https://github.com/RapidAI/YOLO2COCO/LICENSE',
        }]

    def append_bg_img(self, img_list):
        bg_dir = self.raw_data_dir / 'background_images'
        if bg_dir.exists():
            bg_img_list = list(bg_dir.iterdir())
            for bg_img_path in bg_img_list:
                img_list.append(str(bg_img_path))
        return img_list

    def _get_category(self, classes_path):
        class_list = self.read_txt(classes_path)
        categories = []
        for i, category in enumerate(class_list):
            categories.append({
                'supercategory': category,
                'id': i,
                'name': category,
            })
        return categories

    def convert(self, img_list, save_img_dir, mode):
        images, annotations = [], []
        for img_id, img_path in enumerate(tqdm(img_list, desc=mode)):
            image_dict = self.get_image_info(img_path, img_id, save_img_dir)
            images.append(image_dict)

            label_path = self.raw_data_dir / 'labels' / mode / f'{Path(img_path).stem}.txt'
            annotation = self.get_annotation(label_path,
                                             img_id,
                                             image_dict['height'],
                                             image_dict['width'])
            annotations.extend(annotation)

        json_data = {
            'info': self.info,
            'images': images,
            'licenses': self.licenses,
            'type': self.type,
            'annotations': annotations,
            'categories': self.categories,
        }
        return json_data

    def get_image_info(self, img_path, img_id, save_img_dir):
        img_path = Path(args.path + '/' + img_path)
        self.verify_exists(img_path)

        new_img_name = f'{img_id:012d}.jpg'
        save_img_path = save_img_dir / new_img_name
        img_src = cv2.imread(str(img_path))
        if img_path.suffix.lower() == ".jpg":
            shutil.copyfile(img_path, save_img_path)
        else:
            print(save_img_path)
            cv2.imwrite(str(save_img_path), img_src)

        height, width = img_src.shape[:2]
        image_info = {
            'date_captured': self.cur_year,
            'file_name': new_img_name,
            'id': img_id,
            'height': height,
            'width': width,
        }
        return image_info

    def get_annotation(self, label_path: Path, img_id, height, width):
        def get_box_info(vertex_info, height, width, keypoints_info):
            cx, cy, w, h = [float(i) for i in vertex_info]
            keypoints = [float(i) for i in keypoints_info]

            cx = cx * width
            cy = cy * height
            box_w = w * width
            box_h = h * height

            # left top
            x0 = max(cx - box_w / 2, 0)
            y0 = max(cy - box_h / 2, 0)

            # right bottom
            x1 = min(x0 + box_w, width)
            y1 = min(y0 + box_h, height)

            segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            bbox = [x0, y0, box_w, box_h]
            area = box_w * box_h

            # keypoints
            for i in range(int(len(keypoints) / 2)):
                keypoints[2 * i] *= width
                keypoints[2 * i + 1] *= height
            return segmentation, bbox, area, keypoints

        if not label_path.exists():
            annotation = [{
                'segmentation': [],
                'area': 0,
                'iscrowd': 0,
                'keypoints': [],
                'image_id': img_id,
                'bbox': [],
                'category_id': -1,
                'id': self.annotation_id,
            }]
            self.annotation_id += 1
            return annotation

        annotation = []
        label_list = self.read_txt(str(label_path))
        for i, one_line in enumerate(label_list):
            label_info = one_line.split(' ')
            if len(label_info) < 5:
                warnings.warn(
                    f'The {i + 1} line of the {label_path} has been corrupted.')
                continue

            category_id, vertex_info, keypoints_info = label_info[0], label_info[1:5], label_info[5:]
            segmentation, bbox, area, keypoints = get_box_info(vertex_info, height, width, keypoints_info)
            annotation.append({
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
                'keypoints': keypoints,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': int(category_id) + 1,
                'id': self.annotation_id,
            })
            self.annotation_id += 1
        return annotation

    @staticmethod
    def read_txt(txt_path):
        with open(str(txt_path), 'r', encoding='utf-8') as f:
            data = list(map(lambda x: x.rstrip('\n'), f))
        return data

    @staticmethod
    def mkdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def verify_exists(file_path):
        file_path = Path(file_path)
        # print(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'The {file_path} is not exists!!!')

    @staticmethod
    def write_json(json_path, content: dict):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Datasets converter from YOLO to COCO')
    parser.add_argument('path', type=str, default='', help='Dataset root path')
    parser.add_argument('--mode_list', type=str, default='train,val,test', help='generate which mode')
    args = parser.parse_args()
    converter = YOLOToCOCO(args.path)
    converter(mode_list=args.mode_list.split(','))
    print("Finish Change format yolo2coco!")
