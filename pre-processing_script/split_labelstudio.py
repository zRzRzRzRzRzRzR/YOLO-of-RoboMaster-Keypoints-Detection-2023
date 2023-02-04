from pathlib import Path
import re
import os

path = Path("/media/zr/Data/RoboMaster_data/dataset/XJTLU_2022_radar_ALL/labels_orgin")
files = path.glob("*.txt")
for file in files:
    if "cut3" in file.name:
        # print(file.name)
        telephon1 = re.findall('-(\d+)', file.name)
        telephon1[0] = str(int(telephon1[0])+301)
        file.replace(telephon1[0]+'.txt')  # 对文件进行改名
