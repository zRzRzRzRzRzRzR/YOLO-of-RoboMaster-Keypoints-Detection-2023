**2023 RoboMaster XJTLU Armor Keypoints Detection(C++ Openvino inference)**
=

### **Team: 动云科技GMaster战队 <br>**

#### **Author: *视觉组 张昱轩 zR***

## 功能介绍

本代码用于将区域赛视觉标定板目标检测和装甲板四点关键点检测神经网络模型权重文件部署至 C++ Openvino环境。<br>

## DEMO演示代码架构

├── CMakeLists.txt<br>
├── demo_kpt.cpp <br>
├── yolov7_kpt.cpp <br>
├── yolov7_kpt.h <br>
└── demo_weight <br>

## 代码解释和使用说明：

### demo_kpt.cpp:<br>

主代码，将你测试的视频的绝对路径填写至 ```VIDEO_PATH``` 即可。<br>

### yolov7_kpt.h：<br>

包含了头文件和所有的宏定义函数，其中对应的含义如下：<br>

+ ```VIDEOS``` 是否显示推理视频，默认不启动。如果使用则取消注释。
+ ```IMG_SIZE``` 图像推理大小，取决与你的模型，并需要修改ANCHOR的具体值。默认值为```416```。
+ ```NMS_THRESHOLD``` nms阈值，默认值 ```0.2```。
+ ```CONF_THRESHOLD``` 置信度阈值，默认值 ```0.7```。
+ ```CONF_REMAIN``` 置信度保留度，默认值 ```0.25```。
+ ```ANCHOR``` ANCHOR的数量，默认值为```3```。不建议修改，如果修改，需要对应的修改推理的结构。
+ ```DEVICE``` 推理设备，默认值为```CPU```。
+ ```DETECT_MODE``` 检测模式选择，有三种模式可选，分别是:<br>
    + __ARMOR__ 装甲板四点模型检测，请将```DETECT_MODE```设置为```0```。
    + __WIN__ 能量机关五点模型检测，请将```DETECT_MODE```设置为```1```。
    + __BOARD__ 视觉识别板目标检测，请将```DETECT_MODE```设置为```2```。<br>
      上述不同模式对应的应用场景，因此，不同类别中的```CLS_NUM```,```KPT_NUM```值(类别数量和关键点数量)不同。
      不同模式所对应的权重文件不同。```MODEL_PATH```是推理网络的模型位置，在这里放入 ```.xml``` 或 ```.onnx```量化权重文件。

### yolov7_kpt.cpp：<br>

包含了所有的推理代码，部分函数的解释如下:<br>
```vector<Object_result> work``` 推理主函数。

#### API接口:<br>

返回值类型为自定义的```Object_result```结构体，包含如下信息:<br>
struct Object_result<br>
├── ```<int> label``` 识别的标签类别。<br>
├── ```<float> prob```  识别的置信度。<br>
├── ```<rect> bbox``` 识别的bbox信息，包含左上角的坐标和宽高<br>
└── ```vector<float> kpt``` 识别的关键点信息，长度为8，分别是左上，左下，右下，右上四个关键点的坐标。<br>

#### 类接口:<br>

类名: ```yolo_kpt```

### demo_weight：<br>

包含一份针对2023年官方能量机关红色待激活识别的代码。

## 推理样例展示

图片展示:<br>
![](https://github.com/zRzRzRzRzRzRzR/YOLO-of-RoboMaster-Keypoints-Detection-2023/blob/main/show_pic/result_openvino.jpg)
输出展示：<br>
若选择的推理模式为 __ARMOR__ 类别，将会包含 __标签，BBOX，关键点坐标__ 数据。输出的图片将包含 ```KPT_NUM```
个的绿色关键点。样例如下:

```
label:10
bbox:[104 x 81 from (132, 355)]
kpt:[154.106, 380.611] [156.97, 401.911] [223.415, 397.036] [220.275, 376.454]
```

若选择的推理模式为 __BOARD__ 类别，将会包含 __标签，BBOX__ 数据。同时，输出的图片将不会有关键点。样例如下:

```
label:1
bbox:[122 x 433 from (192, 155)]
```

若选择的推理模式为 __WIN__ 类别，将会在 __ARMOR__ 的基础上额外输出一个打击板中心坐标，逻辑如下
+ 如果能量机关装甲板四个点都检测到，直接取四个点中点
+ 如果左上，右下 / 左下， 有右上两个点检测到，另外两个点缺失一个，则取这两个检测到的点中点。
+ 如果不满足上面两个情况，取bbox中点。
经过测试，该方案能保证，只要能检测到目标板，一定能精确的检测到打击到目标板中心。


## 推理代码测试平台

```
硬件设备:
CPU: Inter NUC 11th 11350H
GPU: Inter NUC 核显 
Memory: 8GB
```

```
系统配置:
OS: Ubuntu 20.04.5
Kernel: 5.15.0
Openvino: 2022.3
gcc/g++ : 9.4.0
cmake : 3.16.3
Python : 3.8.5
ONNX : 15
OpenCV : 4.6.0
```

__注意: 该代码在上述指定版本以下的版本可能无法运行。 例如在```OpenCV 4.2.2```下没有```cv::dnn::SoftNMS```
方法,请更换为```cv::dnn::NMS```。__
