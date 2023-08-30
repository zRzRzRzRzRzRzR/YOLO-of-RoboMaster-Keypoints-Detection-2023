'''
使用前必看:
1.本脚本将yolov7训练的聚类环节单独抽出，方便在更改训练和检测数据集大小时修改anchoer的数据，本脚本采用的是yolo自带的k-means聚类方法。
2.具体参数如下：
    dataset: 数据的yaml路径
    n: 类簇的个数
    img_size: 训练过程中的图片尺寸（32的倍数）
    thr: anchor的长宽比阈值，将长宽比限制在此阈值之内
    gen: k - means算法最大迭代次数（不理解的可以去看k - means算法）
    verbose: 打印参数
3.输出anchor后，需要将其修改至 data 文件夹下对应的训练cfg文件才能生效。
'''
import sys
sys.path.append('./')
import utils.autoanchor as autoAC

new_anchors = autoAC.kmean_anchors('/root/Yolov7_radar/data/armor-2.yaml', 24, 1280, 5.0, 1000, True)
print("生成的anchor如下:")
print(new_anchors)
