from pathlib import Path
import re
import os

path = Path("D:\images")
files = path.glob("*.jpg")
for file in files:
        # print(file.name)
        telephon1 = re.findall('-(\d+)', file.name)

        telephon1[0] = str(int(telephon1[0]))
        file.replace(telephon1[0]+'.jpg')  # 对文件进行改名
