**2023 RoboMaster XJTLU Armor Keypoints Detection**
=

### **Team: 动云科技GMaster战队 <br>**

#### **Author: *视觉组 张昱轩 zR***

## 背景介绍:

动云科技GMaster战队2023赛季 YOLO 目标检测 装甲板四点模型 能量籍贯五点模型以及区域赛视觉识别板检测模型训练代码。</br>
本仓库包含从数据集制作到推理部署全套代码。<br>

**```注意```**

+ 1.如果你需要下载最新版本的代码，请克隆```dev```分支，最新的代码无法保证性能稳定。<br>

## 环境配置

我们团队的训练配置和推理配置如下
***
|硬件设备| 训练设备|推理设备|
| - | - | - |
| CPU | 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz | Inter NUC 11th 11350H |
| GPU | NVIDA A5000 x 8 | Inter NUC 11th Xe核显 |
| Memory| 80GB | 8GB |

| 环境配置 |训练设备 |推理设备 |
| - | - | - |
| OS | Ubuntu 20.04.5 | Ubuntu 20.04.5 |
| CUDA | 11.3 | \ |
| OpenVINO | \ | 2022.2 |
|Kernel | 5.15.0 | 5.15.0 |
|gcc/g++ | 9.4.0 | 9.4.0 |
|cmake | \ | 3.16.3 |
|Python | 3.8.6 | 3.10.8 |
|ONNX | 1.13.0 | 1.15.0 |
|Pytorch | 1.11.0 | \ |

**```注意```**

+ 更详细的推理设备配置文件以及注意事项，请查看对应功能的```README.md```文件。

## 数据集格式和标注:<br>

### 关键点检测(装甲板四点模型，能量机关五点模型)数据集格式说明:

对于每一份数据，包含两份文件，分别是:

+ 一份图片放置在 ```images``` 目录下
+ 一份对应的txt文件 1.txt 共一行，包含13个数据， 放置在 ```labels``` 目录下。其中每个数据的含义是:<br>
  第0位: 装甲板的类别<br>
  第1-4位: 装甲板标注锚框,用于目标检测。四个数的顺序是x,y,w,h，其中x,y,是锚框中心点的归一化坐标，w,h 是归一化相对宽高。<br>
  第5-14位: 装甲板灯条的边界点，从前到后分别是:<br>

* 左上角归一化坐标x1,y1<br>
* 左下角归一化坐标x2,y2<br>
* 右下角归一化坐标x3,y3<br>
* 右上角归一化坐标x4,y4<br>
* R点归一化坐标x5,y5(能量机关五点模型特有)<br>
  **```注意```** 图片和标注的文件名必须完全相同

### 样例解释:

下图为一份数据集样例展示：
#### 使用上交开源标注工具
+ 1.jpg<br>
  ![](https://github.com/zRzRzRzRzRzRzR/YOLO-of-RoboMaster-Keypoints-Detection-2023/blob/main/show_pic/label_1.jpg)<br>
+ 1.txt<br> *2 0.498437 0.260417 0.328125 0.241667 0.357313 0.220757 0.361594 0.336893 0.659389 0.319064 0.651708
  0.202737*
#### 使用Labelme标注工具
+ 1.jpg<br>
  ![](https://github.com/zRzRzRzRzRzRzR/YOLO-of-RoboMaster-Keypoints-Detection-2023/blob/main/show_pic/label_2.jpg)<br>
+ 1.txt<br> *2 0.498437 0.260417 0.328125 0.241667 0.357313 0.220757 0.361594 0.336893 0.659389 0.319064 0.651708
  0.202737*
### 数据集标注方法:

前五位 (class,x,y,w,h):<br>

* 使用 [Labelimg](https://github.com/heartexlabs/labelImg) 标注工具,使用该工具标注的结果能直接得出0-4位的值，无需调整，并将所有标注文件放在一个文件夹。

* 5-14位:<br>

+ 方法一 (快但不保证全对):
    + 使用 [RoboMaster智能数据集标注工具](https://github.com/xinyang-go/LabelRoboMaster)
      ，如果您使用的是这个工具标注，那么接下来的操作是:<br>
        + 将所有标注文件放在一个文件夹
        + 使用 *pre-processing_script/move_label.py* 并链接两份同名的标注文件，得到最终的标注文件

+ 方法二 (慢但细心标能全对):
    + 使用 [Labelme](https://github.com/wkentaro/labelme) 标注工具，使用该工具，你需要经历以下步骤:
        + 标注角点时，请按照5-12位的顺序标注，一定要标准
        + 将所有标注好的json文件放置于同一文件夹
        + 执行 ```pre-processing_script/change_json.py``` 并获得转换后的txt文件
        + 执行 ```pre-processing_script/move_label.py``` 并链接两份同名的标注文件，得到最终的标注文件

### 目标检测(视觉识别板)数据集格式说明:

我们团队使用 [labelimg](https://github.com/heartexlabs/labelImg) 工具进行标注并保存为yolo格式。<br>
本代码基于 [YoloV7](https://github.com/WongKinYiu/yolov7) 框架开发，理论上支持COCO类型数据集。

### 数据集信息

#### 数据集来源:<br>

##### 装甲板四点模型

+ 西交利物浦大学RM2023赛季场地数据集数据集录制 2500张。
+ RM视觉开源站数据集 3000张。
+ 共计 5500张。

##### 能量机关五点模型

+ 西交利物浦大学RM2023赛季场地数据集数据集录制 1000张。
+ 共计 1000张。
  <br><br>

##### 视觉识别板检测模型

+ 西交利物浦大学RM2023赛季场地数据集数据集录制 2000张。
+ 共计 2000张。

#### 数据集分配:<br>

+ 对于装甲板四点模型检测，数据共有14类，分别为:<br>

***
|编号 | 含义 | 序号 |编号 | 含义 | 序号 |
|-|-|-|-|-|-|
| B1 | 蓝方一号装甲板 | 0 | R1 | 红方一号装甲板 | 7 |
| B2 | 蓝方二号装甲板 | 1 | R2 | 红方二号装甲板 | 8 |
| B3 | 蓝方三号装甲板 | 2 | R3 | 红方三号装甲板 | 9 |
| B4 | 蓝方四号装甲板 | 3 | R4 | 红方四号装甲板 | 10 |
| B5 | 蓝方五号装甲板 | 4 | R5 | 红方五号装甲板 | 11 |
| BO | 蓝方前哨站装甲板 | 5 | RO| 红方前哨站装甲板 | 12 |
| BS | 蓝方哨兵装甲板 | 6 | RS | 红方哨兵装甲板 | 13 |

+ 对于能量机关五点模型检测，数据共有4类，分别为:<br>

***
|编号 | 含义 | 序号 |编号 | 含义 | 序号 |
|-|-|-|-|-|-|
| RR | 红方待打击扇叶 | 0 | BR | 蓝方待打击扇叶 | 2 |
| RW | 红方已打击扇叶 | 1 | BW | 蓝方已打击扇叶 | 3 |

+ 对于视觉识别板目标检测，数据共有4类，分别为:<br>

***
|编号 | 含义 | 序号 |编号 | 含义 | 序号 |
|-|-|-|-|-|-|
| BA | 蓝方A识别板 | 0 | RA | 红方A识别板 | 2 |
| BD | 蓝方D识别板 | 1 | RD | 红方D识别板 | 3 |

**```注意```** 图片和标注的文件名必须完全相同。

+ 如果该数据集类别顺序与你的不相符， 你可以使用```pre-processing_script/change_change_label.py```
  脚本批量修改你的标签。或者修改代码成符合你的数据集顺序。
+ 数据集按照 8:1:1的比例分配为训练集，验证集和测试集。
  超参数设置:<br>
+ 训练超参数未对yolov7原本的超参数进行过多的调整，主要调整了数据集增强的部分。关闭了上下，左右的反转。同时修改了一些其他数据预处理参数。

## 训练流程<br>

### Train:

#### Single GPU:

```python train.py --workers 8 --device 0 --batch-size 32 --data data/armor/armor_detect.yaml --img 640 640 --cfg cfg/armor/yolov8-0.5-SimAM-armor.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.armor.yaml ```

#### Muti GPUs:

``` python -m torch.distributed.launch --nproc_per_node 8 train.py --workers 96 --batch-size 256 --data data/armor/armor_detect.yaml --img 640 640 -cfg cfg/armor/yolov8-0.5-SimAM-armor.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.armor.yaml```

### Inference:

#### On Videos and Image:

```python detect.py --weights yolov7_armor.pt --conf 0.25 --img-size 640 --source yourvideo.mp4```

### 效果展示:

+ inference.jpg<br>
  ![](https://github.com/zRzRzRzRzRzRzR/YOLO-of-RoboMaster-Keypoints-Detection-2023/blob/main/show_pic/result_pytorch.jpg)

### Export:

- 导出ONNX文件:<br>
  ```python export.py --weights yolov7_armor.pt --simplify --iou-thres 0.65 --conf-thres 0.7 --img-size 416 416 --max-wh 416 ```
- 如果你需要将onnx文件转换为Openvino文件,你还需要执行以下步骤:<br>
    - 安装
      OpenVINO推理引擎，安装教程请查看[OpenVINO安装指南](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html)
    - 在安装的Python环境下，执行<br>
      ```mo --input_model /path/to/your_models.onnx --output_dir /path/to/out_dir```
    - 将转换得到的xml, bin, mapping 文件放置在一个文件夹，便接入C++推理
- 目前支持onnx12,13,14,15版本的导出

### Inference in Openvino (x86)

+ Openvino推理代码位于文件夹 ```C++_inference_openvino_kpt``` 中，仅做简单推理，请根据需求自行接入工程。
+ 如果推理图像修改，工程中的anchor和图像大小也需要修改，anchor修改请执行```pre-processing_script/change_anchor.py```

## 总结

+ 该项目针对RMUC2023赛季，如果你有好的建议，欢迎给我留言哦。
+ 如果你觉得项目对你有帮助，请给个star吧~
+ 如果你有更好的建议，欢迎提出PR，或者直接联系我哦，大家一起学习！

## 联系方式

+ 作者微信: zR_ZYX
+ 作者邮箱: Yuxuan.Zhang2104@student.xjtlu.edu.cn
+ 团队邮箱: TeamGMaster@xjtlu.edu.cn