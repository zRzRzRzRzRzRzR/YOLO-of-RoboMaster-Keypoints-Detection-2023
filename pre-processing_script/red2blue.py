'''
使用前必看:
1. 本代码用来将能量机关变颜色（红色变蓝色）
2. 本代码目前鲁棒性极差，不建议使用
'''
import cv2
import os
import argparse
from tqdm import tqdm
def change_red_to_blue(image_path, output_path, target_blue=(250, 110, 20), threshold=50):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 红色HSV范围
    lower_red1 = (0, 50, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 50, 50)
    upper_red2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.add(mask1, mask2)

    # 二值化处理
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    # 将二值化部分更改为目标蓝色
    blue_image = image.copy()
    for i in range(3):
        blue_image[..., i] = (image[..., i] * (1 - binary_mask / 255.0) + target_blue[i] * (binary_mask / 255.0)).astype('uint8')

    cv2.imwrite(output_path, blue_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将红色标志替换为蓝色标志')
    parser.add_argument('--input_dir', type=str, default="/media/zr/Data/RoboMaster_data/dataset/win_dataset/images",
                        help='输入图片所在文件夹')
    parser.add_argument('--output_dir', type=str,
                        default="/media/zr/Data/RoboMaster_data/dataset/win_dataset/images_convert",
                        help='输出图片所在文件夹')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]

    for image_path in tqdm(image_paths):
        output_path = os.path.join(args.output_dir, os.path.basename(image_path))
        change_red_to_blue(image_path, output_path)
