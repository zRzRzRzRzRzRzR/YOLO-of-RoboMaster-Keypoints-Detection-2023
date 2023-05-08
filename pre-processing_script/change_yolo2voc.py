import os
import cv2
import xml.etree.ElementTree as ET
from lxml import etree
import shutil
import random
import argparse


def create_voc_xml(filename, img_shape, bboxes, labels):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC2007"
    ET.SubElement(root, "filename").text = filename

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2])

    ET.SubElement(root, "segmented").text = "0"

    for bbox, label in zip(bboxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(bbox[0])
        ET.SubElement(bb, "ymin").text = str(bbox[1])
        ET.SubElement(bb, "xmax").text = str(bbox[2])
        ET.SubElement(bb, "ymax").text = str(bbox[3])

    return root


def yolo_to_voc(input_dir, output_dir, class_list, train_ratio, val_ratio):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_path = os.path.join(input_dir, "images")
    labels_path = os.path.join(input_dir, "labels")
    voc_images_path = os.path.join(output_dir, "JPEGImages")
    voc_ann_path = os.path.join(output_dir, "Annotations")

    if not os.path.exists(voc_images_path):
        os.makedirs(voc_images_path)

    if not os.path.exists(voc_ann_path):
        os.makedirs(voc_ann_path)

    for filename in os.listdir(images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(images_path, filename)
            img = cv2.imread(img_path)
            img_shape = img.shape

            yolo_label_path = os.path.join(labels_path, filename.split('.')[0] + '.txt')
            with open(yolo_label_path, 'r') as f:
                yolo_labels = f.readlines()

            bboxes = []
            labels = []

            for yolo_label in yolo_labels:
                yolo_label = yolo_label.strip().split(' ')
                class_id = int(yolo_label[0])
                label = class_list[class_id]

                x_center = float(yolo_label[1]) * img_shape[1]
                y_center = float(yolo_label[2]) * img_shape[0]
                width = float(yolo_label[3]) * img_shape[1]
                height = float(yolo_label[4]) * img_shape[0]

                xmin = int(x_center - width / 2)
                ymin = int(y_center - height / 2)
                xmax = int(x_center + width / 2)
                ymax = int(y_center + height / 2)

                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

            voc_xml = create_voc_xml(filename, img_shape, bboxes, labels)
            voc_xml_str = ET.tostring(voc_xml, encoding='unicode', method='xml')

            voc_ann_filename = os.path.join(voc_ann_path, filename.split('.')[0] + '.xml')

            with open(voc_ann_filename, 'w') as f:
                f.write(voc_xml_str)

            shutil.copy(img_path, os.path.join(voc_images_path, filename))

    image_sets_path = os.path.join(output_dir, "ImageSets")
    main_path = os.path.join(image_sets_path, "Main")

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    all_images = [filename for filename in os.listdir(images_path) if
                  filename.endswith(".jpg") or filename.endswith(".png")]

    random.shuffle(all_images)

    n_train = int(len(all_images) * train_ratio)
    n_val = int(len(all_images) * val_ratio)
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]

    with open(os.path.join(main_path, "train.txt"), "w") as f:
        f.writelines([img.split('.')[0] + '\n' for img in train_images])

    with open(os.path.join(main_path, "val.txt"), "w") as f:
        f.writelines([img.split('.')[0] + '\n' for img in val_images])

    with open(os.path.join(main_path, "trainval.txt"), "w") as f:
        f.writelines([img.split('.')[0] + '\n' for img in train_images + val_images])

    with open(os.path.join(main_path, "test.txt"), "w") as f:
        f.writelines([img.split('.')[0] + '\n' for img in test_images])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/media/zr/Data/RoboMaster_data/Smart_Car", help="Path to YOLO dataset")
    parser.add_argument("--output_dir", type=str, default="/media/zr/Data/RoboMaster_data/Smart_Car_voc", help="Path to output VOC2007")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")

    class_list = ['Plant_Corn_1', 'Plant_Corn_2', 'Plant_Corn_3', 'Plant_Corn_4',
                  'Plant_Cucumber_1', 'Plant_Cucumber_2', 'Plant_Cucumber_3', 'Plant_Cucumber_4',
                  'Plant_Rice_1', 'Plant_Rice_2', 'Plant_Rice_3', 'Plant_Rice_4',
                  'Plant_Wheat_1', 'Plant_Wheat_2', 'Plant_Wheat_3', 'Plant_Wheat_4',
                  'Fruit_Corn_1', 'Fruit_Corn_2', 'Fruit_Corn_3', 'Fruit_Corn_4',
                  'Fruit_Cucumber_1', 'Fruit_Cucumber_2', 'Fruit_Cucumber_3', 'Fruit_Cucumber_4',
                  'Fruit_Watermelon_1', 'Fruit_Watermelon_2', 'Fruit_Watermelon_3', 'Fruit_Watermelon_4']

    args = parser.parse_args()

    yolo_to_voc(args.input_dir, args.output_dir, class_list, args.train_ratio, args.val_ratio)
