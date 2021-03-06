import cv2
import datetime
import logging
import numpy as np
import time
import torch
from deep_sort import DeepSort
from functools import reduce
from yolo3.models import Darknet
from yolo3.utils.helper import load_classes
from yolo3.utils.model_build import p1p2Toxywh, resize_boxes, soft_non_max_suppression


def detect_from_frame(frame, model, classes, thres=0.5, nms_thres=0.4, half=False):
    model = model
    model.eval()
    if half: model.half()
    device = next(model.parameters()).device

    h, w, _ = frame.shape
    image = cv2.resize(frame, (model.img_size[1], model.img_size[0]), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(image).to(device)
    image = image.permute((2, 0, 1)) / 255.
    image = image.unsqueeze(0)
    if half: image = image.half()

    prev_time = time.time()
    with torch.no_grad():
        detections = model(image)
        detections = soft_non_max_suppression(detections, thres, nms_thres)
        detections = detections[0]
        if detections is not None:
            detections = resize_boxes(detections, model.img_size, (h, w))

    curr_time = time.time()
    infe_time = datetime.timedelta(seconds=curr_time - prev_time)
    logging.info(f'Inference time: {infe_time}')
    return detections

def draw_rect_on_frame(frame, detections, classes, colors, thickness=2, with_label=True):
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        cls = int(det[-1])
        c1 = (int(x1), int(y1))
        c2 = (int(x2), int(y2))
        cv2.rectangle(frame, c1, c2, colors[cls], thickness)

        if with_label:
            font_size = frame.shape[0] / 1000.
            label = str(int(det[4])) + ":" + classes[int(det[-1])]
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, 1)
            font_w, font_h = text_size
            cv2.rectangle(frame, (c1[0], max(0, int(c1[1] - 3 - 18 * font_size))), (c1[0] + font_w, max(c1[1], int(3 + 18 * font_size))), colors[cls], -1)
            cv2.putText(frame, label, (c1[0], max(c1[1] - 3, font_h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 0, 0), 1)
    return frame

def get_colors_for_classes(classes):
    np.random.seed(1)
    colors = (np.random.rand(min(999, len(classes)), 3) * 255).astype(int)
    np.random.seed(None)
    colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
    return colors


def track(frame, model, classes, class_mask, tracker):
    detections = detect_from_frame(frame, model, classes)
    if detections is not None:
        boxs = p1p2Toxywh(detections[:, :4])
        class_ids = detections[:, -1]
        confidences = detections[:, 4]
        mask_set = [class_ids == mask_id for mask_id in class_mask]
        mask = reduce(lambda a, b: a | b, mask_set)
        boxs = boxs[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        detections = tracker.update(boxs.float(), confidences, frame, class_ids)
    return detections


def setup(model_path, model_img_size, weights_path, device, classes_path, deepsort_weight_path):
    model = Darknet(model_path, img_size=(model_img_size[0], model_img_size[1]))
    model.load_darknet_weights(weights_path)
    model.to(device)

    classes = load_classes(classes_path)
    colors = get_colors_for_classes(classes)

    tracker = DeepSort(deepsort_weight_path,
                       min_confidence=1,
                       use_cuda=True,
                       nn_budget=30,
                       n_init=3,
                       max_iou_distance=0.7,
                       max_dist=0.3,
                       max_age=30)
    
    return model, classes, colors, tracker
