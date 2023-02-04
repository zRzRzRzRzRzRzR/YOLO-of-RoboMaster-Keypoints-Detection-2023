'''
使用前必看:
1. 此代码的出现是因为目前我们无法将labelme和labelimg中的数据进行匹配，例如，labelimg中的多个标签，在labelme中的顺序并不相同
如果直接鲁莽的讲每行接上，将导致数据出错，此代码的作用在于将labelme正确的衔接到对饮的labelimg数据后。
2. 运行此代码之前，你需要准备好labelme标注的json文件的文件夹，labelimg标注的txt文件夹，无需labelme转txt的文件夹。
3. 如果你需要将labelme的数据转为txt格式（主要是用来方便与其他学校交流），请运行change_json2yolo.py
'''

import os
import json
import asyncio
import argparse
from os import PathLike
from typing import Union

import aiofiles
from loguru import logger

from typing import List, Tuple
from shapely.geometry import Polygon


def if_intersect(data1: List, data2: List) -> float:
    """
    Calculation of the intersection area of any two figures
    :param data1, data2:Coordinates of two plane figures
    :return: The area intersection of the current object and the object to be compared
    """

    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    return inter_area


def xywh2two_coordinate(row, width, height) -> Tuple:
    """

    :param row: <Label serial number> <x> <y> <w> <h>
    :param width: the width of the image
    :param height: the height of the image
    :return: Tuple of two coordinates
    """
    len = 0
    for r in row:
        row[len] = float(r)
        len += 1
    x_min = min(max(0.0, row[1] - row[3] / 2), 1.0) * width
    y_min = min(max(0.0, row[2] - row[4] / 2), 1.0) * height
    x_max = min(max(0.0, row[1] + row[3] / 2), 1.0) * width
    y_max = min(max(0.0, row[2] + row[4] / 2), 1.0) * height
    return x_min, y_min, x_max, y_max


def xywh2four_coordinate(row: List, width: int, height: int) -> List[Tuple]:
    """

    :param row: <Label serial number> <x> <y> <w> <h>
    :param width: the width of the image
    :param height: the height of the image
    :return: List of four coordinates
    """
    x_min, y_min, x_max, y_max = xywh2two_coordinate(row, width, height)
    return [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (x_max, y_min)
    ]


def coordinate2normalized4(points: List, width: int, height: int) -> List:
    """
    The four points have ordinary and normal coordinate composition and are converted into normalized four coordinate points.
    :param points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    :param width: the width of the image
    :param height: the height of the image
    :return: a list of eight points, specific numbers
    """
    t_x = lambda x: x[0] / width
    t_y = lambda x: x[1] / height
    return [t_x(points[0]), t_y(points[0]),
            t_x(points[1]), t_y(points[1]),
            t_x(points[2]), t_y(points[2]),
            t_x(points[3]), t_y(points[3])]


def normalized8point2coordinate4(points: List, width: int, height: int) -> List[Tuple]:
    """
    :param points: a list of eight points, specific numbers
    :param width: the width of the image
    :param height: the height of the image
    :return: four points have ordinary and normal coordinate composition
    """
    t_x = lambda x: x * width
    t_y = lambda x: x * height
    return [
        (t_x(points[0]), t_y(points[1])),
        (t_x(points[2]), t_y(points[3])),
        (t_x(points[4]), t_y(points[5])),
        (t_x(points[6]), t_y(points[7]))
    ]


class Points:

    def __init__(self, me_data: Union[str, PathLike, bytes], img_data: Union[str, PathLike, bytes]) -> None:
        self.path = me_data
        self.height: int
        self.width: int
        self.me_points: List
        self.img_points_origin: List = []
        self.img_points: List = []
        if isinstance(me_data, str):
            try:
                with open(me_data) as fp:
                    points_data = json.loads(fp.read())
            except FileNotFoundError as e:
                logger.error("No labelme data.")
                raise e
            except (json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error("There is some encoding error about labelme's json file.")
                raise e
            finally:
                self.height = points_data["imageHeight"]
                self.width = points_data["imageWidth"]
                self.me_points = points_data["shapes"]
        if isinstance(img_data, str):
            try:
                with open(img_data) as f:
                    for line in f.readlines():
                        self.img_points_origin.append(line.split())
            except FileNotFoundError as e:
                logger.error(f"{img_data}The corresponding labelimg data is missing")
                raise e
        for points in self.img_points_origin:
            self.img_points.append(xywh2four_coordinate(points, self.width, self.height))

    def get_correspond(self, t_data: List) -> List:
        """
        for the label data of labelimg, find the correct serial number in origin list.
        :param t_data:
        :return:
        """
        sn: int = self.img_points.index(t_data)
        return self.img_points_origin[sn]


async def read_classes(class_data: Union[str, PathLike, bytes]):
    label_list: List = []
    if isinstance(class_data, str):
        try:
            async with aiofiles.open(class_data) as f:
                for line in await f.readlines():
                    label_list.append(*line.split())
        except (OSError, FileNotFoundError) as e:
            logger.error("unable to open classes.txt")
            raise e
    return label_list


async def output(label: int, img_point: List, me_point: List, final_path) -> None:
    out1 = ""
    out2 = ""
    for x in img_point[1:]:
        out1 = out1 + f" {x}"
    for x in me_point:
        out2 = out2 + f" {x}"
    out = str(label) + out1 + out2
    async with aiofiles.open(final_path, 'at') as f:
        await f.writelines(out + "\n")


async def points_callback(label_data: Points, label_list, final_path):
    tasks = []
    num = 0
    logger.info(f"There are {len(label_data.me_points)} points in the labelme")
    logger.info(f"There are {len(label_data.img_points)} points in the labelimg")
    logger.info(f"Worst case will be {len(label_data.me_points) * len(label_data.img_points)} operations")
    for me_point in label_data.me_points:
        for img_point in label_data.img_points:
            num += 1
            area = if_intersect(me_point["points"], img_point)
            logger.info(f"The operation has been carried out {num} times")
            logger.debug(f"labelme points: {me_point}")
            logger.debug(f"labelimg points: {img_point}")
            if area > 0:
                label = label_list.index(me_point["label"])
                tasks.append(output(label, label_data.get_correspond(img_point),
                                    coordinate2normalized4(me_point["points"], label_data.width,
                                                           label_data.height), final_path))
                logger.info(f"{label_data.path} match success.")
            else:
                logger.info(f"{label_data.path} match error.")
    await asyncio.gather(*tasks)


@logger.catch
async def main(labelme_dir, labelimg_dir, final_dir, classes_path) -> None:
    label_list = await read_classes(classes_path)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    list_labelme = os.listdir(labelme_dir)
    file_num = len(list_labelme)
    logger.info(f"There are {file_num} files.")
    for cnt, json_name in enumerate(list_labelme):
        labelme_path = labelme_dir + json_name
        if ".json" not in json_name:
            continue
        labelimg_path = labelimg_dir + json_name.replace('.json', '.txt')
        final_path = final_dir + json_name.replace('.json', '.txt')
        label_data = Points(labelme_path, labelimg_path)
        await points_callback(label_data, label_list, final_path)
    await asyncio.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelme_dir', type=str, default='', help='Location of the original labelme json datasets')
    parser.add_argument('--labelimg_dir', type=str, default='', help='Location of the original labelimg txt datasets')
    parser.add_argument('--output_dir', type=str, default='', help='Output location of the changed txt dataset')
    parser.add_argument('--classes_file', type=str, default='', help='Path to the file where the list of tags is saved')
    parser.add_argument('--logs_file', type=str, default='label-match_{time:YYYY-MM-DD_HH:mm:ss}',
                        help='Path to the file where log will be saved')
    args = parser.parse_args()

    logger.remove(handler_id=None)
    # logger.add(sink=sys.stderr,
    #            level="DEBUG",
    #            colorize=True,
    #            format="{time} | {level} | {message}")

    logger.add(sink='label-match_{time}.log',
               level="DEBUG",
               colorize=True,
               format="{time:YYYY-MM-DD at HH:mm:ss} | {level: <8} | {name: ^15} | {function: ^15} | {line: >3} | {message}")
    try:
        asyncio.run(
            main(args.labelme_dir, args.labelimg_dir, args.output_dir, args.classes_file)
        )
    except OSError as e:
        print("Some thing wrong with OS system,maybe your path or file have something wrong.")
        print("Check your file and path.The programme will be exited.")
        exit()
    except RuntimeWarning as e:
        print("Some thing wrong with Coroutine system.")
        print("Please make a issue in Github with your ERROR code.")
        raise e
    finally:
        print("Finish")
