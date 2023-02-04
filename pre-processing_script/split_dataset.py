'''
使用前必看:
1. 本代码是将所有的数据集进行分割。
2. train，val文件必须分割，如果你需要test文件，需要添加 --test_mode 参数 并且修改train 和 val 的比例，满足
    test_present = 1 - train_present - val_present
'''
import shutil
import random
import os
import argparse
import warnings


def split_label(dataset_all_path, dataset_split_path, train_percent=0.85, val_percent=0.15, test_mode=1):
    image_original_path = dataset_all_path + '/images/'
    label_original_path = dataset_all_path + '/labels/'
    train_image_path = dataset_split_path + '/images/train/'
    train_label_path = dataset_split_path + '/labels/train/'
    txt_train_file = dataset_split_path + '/train.txt'
    val_image_path = dataset_split_path + '/images/val/'
    val_label_path = dataset_split_path + '/labels/val/'
    txt_val_file = dataset_split_path + '/val.txt'

    if test_mode:
        warnings.warn("You choose creat test dataset, if you have a small dataset, we do not recommend you to do this!")
        test_image_path = dataset_split_path + '/images/test/'
        test_label_path = dataset_split_path + '/labels/test/'
        txt_test_file = dataset_split_path + '/test.txt'

    if not os.path.exists(dataset_split_path):
        warnings.warn("Could no find saving_path,Creat one!")
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
        file = open(txt_train_file, 'w')
        file.write("")
        file.close()

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)

    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)
        file = open(txt_val_file, 'w')
        file.write("")
        file.close()

    if test_mode:
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        if not os.path.exists(test_label_path):
            os.makedirs(test_label_path)
            file = open(txt_test_file, 'w')
            file.write("")
            file.close()

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)
    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    train = random.sample(list_all_txt, num_train)
    val_test = [i for i in list_all_txt if not i in train]
    if test_mode:
        val = random.sample(val_test, num_val)
    else:
        val = val_test
    file_train = open(txt_train_file, 'w')
    file_val = open(txt_val_file, 'w')
    if test_mode:
        file_test = open(txt_test_file, 'w')
    if test_mode:
        print("train:{}, val:{}, test:{}".format(len(train), len(val), len(val_test) - len(val)))
    else:
        print("train:{}, val:{}".format(len(train), len(val)))
    for i in list_all_txt:
        name = total_txt[i][:-4]
        srcImage = image_original_path + name + '.jpg'
        srcLabel = label_original_path + name + '.txt'
        if i in train:
            dst_train_Image = train_image_path + name + '.jpg'
            dst_train_Label = train_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
            file_train.write(str('images/train/' + name + '.jpg' + '\n'))
        elif i in val:
            dst_val_Image = val_image_path + name + '.jpg'
            dst_val_Label = val_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
            file_val.write(str('images/val/' + name + '.jpg' + '\n'))
        else:
            dst_test_Image = test_image_path + name + '.jpg'
            dst_test_Label = test_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)
            file_test.write(str('images/test/' + name + '.jpg' + '\n'))

    file_train.close()
    file_val.close()
    if test_mode:
        file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_dir', type=str, default='', help='Location of the original all datasets')
    parser.add_argument('--split_dir', type=str, default='', help='Output location of the split dataset')
    parser.add_argument('--train_percent', type=float, default=0.85, help='train dataset percent')
    parser.add_argument('--val_percent', type=float, default=0.15, help='val dataset percent')
    parser.add_argument('--test_mode', action='store_true', help='split test dataset')
    args = parser.parse_args()
    split_label(args.all_dir, args.split_dir, args.train_percent, args.val_percent, args.test_mode)
    print("Finish Split dataset!")
