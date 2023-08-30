'''
使用前必看:
1. 如果你是按照READ.md文档的顺序标注的，那么正常你是用不到这个文件的。如果你是从其他开源站获得的数据，由于标签不对需要更改标签，那么你可以使用这
个脚本帮助修改。
2.使用本脚本前请备份你的标注文件。下面的样例是将以YOLO格式导出的LabelStudio数据集转换为西浦GMaster训练的数据集
'''
import os
import argparse


def update_labels(folder_path):
    # 你的旧的和新的类别顺序
    old_order = ["B1", "B2", "B3", "B4", "B5", "BB", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RB", "RO", "RS"]
    new_order = ["B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS", "BB", "RB"]

    # 创建一个字典来映射旧的id到新的id
    id_mapping = {old_order.index(name): new_order.index(name) for name in old_order}

    # 遍历标签文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):  # 假设标签文件是.txt格式
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()

                # 修改每一行的标签id
                for i in range(len(lines)):
                    parts = lines[i].split(" ")
                    parts[0] = str(id_mapping[int(parts[0])])  # 更新id
                    lines[i] = " ".join(parts)

                # 将修改后的行写回文件
                with open(os.path.join(root, file), "w") as f:
                    f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update label files")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing label files")
    args = parser.parse_args()

    update_labels(args.folder_path)
