import argparse
import os

from openvino.tools import mo
import torch
from PIL import Image
from utils.plots import plot_one_box
import cv2
from typing import List, Tuple, Dict
from utils.general import scale_coords, non_max_suppression
from openvino.runtime import Model, Core, serialize, Tensor
from collections import namedtuple
import yaml
from utils.datasets import create_dataloader, letterbox
from utils.general import check_dataset, box_iou, xywh2xyxy, colorstr
import numpy as np
from tqdm.notebook import tqdm
from utils.metrics import ap_per_class
import nncf


def preprocess_image(img0: np.ndarray):
    # resize
    img = letterbox(img0, auto=False)[0]
    img = cv2.resize(img, (img_size_set, img_size_set))
    # Convert
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0


def prepare_input_tensor(image: np.ndarray):
    input_tensor = image.astype(np.float32)  # uint8 to fp16/32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def detect(model: Model, image_path: "", conf_thres: float = 0.25, iou_thres: float = 0.45, classes: List[int] = None,
           agnostic_nms: bool = False):
    output_blob = model.output(0)
    img = np.array(Image.open(image_path))
    preprocessed_img, orig_img = preprocess_image(img)
    input_tensor = prepare_input_tensor(preprocessed_img)
    predictions = torch.from_numpy(model(input_tensor)[output_blob])
    pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    return pred, orig_img, input_tensor.shape


def draw_boxes(predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str],
               colors: Dict[str, int]):
    if not len(predictions):
        return image
    predictions[:, :4] = scale_coords(input_shape[2:], predictions[:, :4], image.shape).round()

    # Write results
    for *xyxy, conf, cls in reversed(predictions):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image, label=label, color=colors[names[int(cls)]], line_thickness=1)
    return image


def transform_fn(data_item):
    img = data_item[0].numpy()
    input_tensor = prepare_input_tensor(img)
    return input_tensor


def test(data, model: Model, dataloader: torch.utils.data.DataLoader, conf_thres: float = 0.001,
         iou_thres: float = 0.65,  # for NMS
         single_cls: bool = False, v5_metric: bool = False, names: List[str] = None, num_samples: int = None
         ):
    model_output = model.output(0)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    seen = 0
    p, r, mp, mr, map50, map = 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    for sample_id, (img, targets, _, shapes) in enumerate(tqdm(dataloader)):
        if num_samples is not None and sample_id == num_samples:
            break
        img = prepare_input_tensor(img.numpy())
        targets = targets
        height, width = img.shape[2:]

        with torch.no_grad():
            # Run model
            out = torch.from_numpy(model(Tensor(img))[model_output])  # inference output
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels

            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=None, multi_label=True)
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device='cpu')
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, v5_metric=v5_metric, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return mp, mr, map50, map, maps, seen, nt.sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/zr/armor_0805.onnx', help='initial weights path')
    parser.add_argument('--data', type=str, default='data/armor/armor_detect.yaml', help='model.yaml path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', type=str, default='GPU', help='CPU,GPU or GNA')
    opt = parser.parse_args()


    model = mo.convert_model(opt.weights)
    filename, extension = os.path.splitext(opt.weights)
    img_size_set = opt.img_size
    serialize_model = filename + '.xml'
    serialize_model_int8 = filename + '_int8.xml'
    serialize(model, serialize_model)
    core = Core()
    model = core.read_model(serialize_model)
    device = opt.device

    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    compiled_model = core.compile_model(model, device)
    # Dataloader
    TASK = 'val'  # path to train/val/test images
    Option = namedtuple('Options',
                        ['single_cls'])  # imitation of commandline provided options for single class evaluation
    opt = Option(False)
    dataloader = create_dataloader(data['val'], img_size_set, 1, 32, opt, pad=0.5, prefix=colorstr(f'{TASK}: '))[0]

    mp, mr, map50, map, maps, num_images, labels = test(data=data, model=compiled_model, dataloader=dataloader,
                                                        names=data['names'])
    # Print results
    print("FP32_result")
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', num_images, labels, mp, mr, map50, map))

    quantization_dataset = nncf.Dataset(dataloader, transform_fn)
    quantized_model = nncf.quantize(model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)
    serialize(quantized_model, serialize_model_int8)
    int8_compiled_model = core.compile_model(quantized_model, device)
    int8_result = test(data=data, model=int8_compiled_model, dataloader=dataloader, names=data['names'])
    mp, mr, map50, map, maps, num_images, labels = int8_result

    # Print results
    print("INT8_result")
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', num_images, labels, mp, mr, map50, map))
