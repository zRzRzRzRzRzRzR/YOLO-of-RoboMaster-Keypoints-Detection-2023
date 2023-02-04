'''
使用前必看:
1. 本代码是将labelme和labelimg的数据进行拼接，拼接以后生成的即为最终的标注文件。
2. 使用该脚本之前，请备份你的标注文件，以防操作失误带来的后果。
3. 请确保labelme和labelimg中的文件数量一样且标注的数量一样（txt中的行数一致）。
4. 本脚本没有排序功能，当且仅当每个标注文件内只有一行数据的时候，能够保证数据不出错。
'''
import argparse
import os
import warnings


@staticmethod
def combine_label(labelme_path, labelimg_path, label_save_path):
    try:
        assert (os.path.exists(labelme_path))
    except:
        warnings.warn("Could no find labelme_path, Combine Fail！")
        return
    try:
        assert (os.path.exists(labelimg_path))
    except:
        warnings.warn("Could no find labelimg_path，Combine Fail！")
        return
    try:
        assert (os.path.exists(label_save_path))
    except:
        warnings.warn("Could no find saving_path,Creat one!")
        os.mkdir(label_save_path)

    total_txt = os.listdir(labelme_path)
    for file in total_txt:
        labelme_file = labelme_path + '/' + file
        labelimg_file = labelimg_path + '/' + file
        label_save_file = label_save_path + '/' + file
        file_me = open(labelme_file, 'r')
        file_img = open(labelimg_file, 'r')
        lines_img = file_img.readlines()
        lines_me = file_me.readlines()
        file_w = open(label_save_file, 'w')
        for index, line in enumerate(lines_me):
            t_me = lines_me[index]
            t_img = lines_img[index]
            if int(t_me[0:2]) < 10:
                w = t_img[:-1] + t_me[1:]
            else:
                w = t_img[:-1] + t_me[2:]
            file_w.write(w)
        file_w.close()
    print("Finish Combine Labels!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labelme_path', type=str, default='', help='labelme dir')
    parser.add_argument('labelimg_path', type=str, default='', help='labelimg dir')
    parser.add_argument('combine_path', type=str, default='', help='combine and save path')
    args = parser.parse_args()
    combine_label(args.labelme_path, args.labelimg_path, args.combine_path)
