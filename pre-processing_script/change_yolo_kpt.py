'''
使用前必看:
下面展示了如何将LabelStudio导出的json数据集格式转换为我们需要的yolo格式。
由于是关键点模型，LabelStudio必然无法正常导出yolo格式，需要使用这个脚本将json格式进行转换。我们需要用其yolo格式导出的images，并用这个脚本生成labels文件夹。
'''

import json
import os
import argparse

def json_to_yolo(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for json_file in os.listdir(input_folder):
        print(json_file)
        if json_file.endswith(".json"):
            with open(os.path.join(input_folder, json_file), "r") as f:
                data = json.load(f)
            yolo_data = []

            rect_result = data["step_1"]["result"]
            point_result = data["step_3"]["result"]

            for rect in rect_result:
                rect_id = rect["id"]
                rect_label = rect["attribute"]
                rect_x = (rect["x"] + rect["width"] / 2) / data["width"]
                rect_y = (rect["y"] + rect["height"] / 2) / data["height"]
                rect_w = rect["width"] / data["width"]
                rect_h = rect["height"] / data["height"]

                points = [f"{rect_label} {rect_x} {rect_y} {rect_w} {rect_h}"]

                sorted_points = sorted([point for point in point_result if point["sourceID"] == rect_id], key=lambda x: x["label"])
                for point in sorted_points:
                    point_x = point["x"] / data["width"]
                    point_y = point["y"] / data["height"]
                    points.append(f"{point_x} {point_y}")

                yolo_data.append(" ".join(points))

            txt_file = json_file.replace(".jpg.json", ".txt")

            with open(os.path.join(output_folder, txt_file), "w") as f:
                f.write("\n".join(yolo_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON labels to YOLO format")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing JSON files")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for YOLO labels")
    args = parser.parse_args()

    json_to_yolo(args.input_folder, args.output_folder)