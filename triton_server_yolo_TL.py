import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import grpc
import tritonclient.grpc as grpcclient

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

from utils.augmentations import letterbox


def get_traffic_light_color(im0, x1, y1, x2, y2):
    traffic_light = im0[int(y1):int(y2), int(x1):int(x2)]
    hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)

    # Red color
    low_red = np.array([161, 155, 84])
    low_red_2 = np.array([0, 155, 84])
    high_red = np.array([179, 255, 255])
    high_red_2 = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, low_red, high_red)
    red_mask_2 = cv2.inRange(hsv, low_red_2, high_red_2)
    red = cv2.bitwise_and(traffic_light, traffic_light, mask=red_mask)
    red_2 = cv2.bitwise_and(traffic_light, traffic_light, mask=red_mask_2)

    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv, low_green, high_green)
    green = cv2.bitwise_and(traffic_light, traffic_light, mask=green_mask)

    # Yellow color
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(traffic_light, traffic_light, mask=yellow_mask)

    if red.any() or yellow.any():
        traffic_light_color = 'red'
    elif green.any():
        traffic_light_color = 'green'
    else:
        traffic_light_color = 'unknown'

    return traffic_light_color


def run(frame, conf_thres=0.25, iou_thres=0.45, max_det=1000,
        classes=[0, 1, 2, 3, 5, 7, 9, 10], agnostic_nms=False, augment=False):
    im = letterbox(frame, imgsz, stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    dt = (Profile(), Profile(), Profile())
    with dt[0]:
        im = torch.from_numpy(im).to(device)
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    with dt[1]:
        # Connect to the Triton server
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        # Create the data for the input tensor
        data = grpcclient.InferInput('input_0', im.shape, "FP16")
        # Set the data for the input tensor
        data.set_data_from_numpy(im, binary_data=True)
        # Execute the model
        result = triton_client.infer("yolov5", model_version="1", inputs=[data])
        # Get the output tensor
        output = result.as_numpy('output_0')
    with dt[2]:
        pred = non_max_suppression(output, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    for i, det in enumerate(pred):  # per image
        im0 = frame.copy()
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if c == 9:
                    x1, y1, x2, y2 = [coord.item() for coord in xyxy]
                    color = get_traffic_light_color(im0, x1, y1, x2, y2)
                    return x1, y1, x2, y2, color, conf.item()
                    # return f"{c}, Traffic Light detected in color: {color}"
    return f"unknown"


device = select_device('')
weights = ROOT / 'yolov5m.pt'
data = ROOT / 'data/coco128.yaml'
video_name = 'TLD120new.mp4'

video = ROOT / f'data/videos/{video_name}'
imgsz = (1080, 1920)
stride, pt = 32, False
imgsz = check_img_size(imgsz, s=stride)  # check image size
bs = 1  # batch_size
fp16 = False

cap = cv2.VideoCapture(f"{video}")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_video = cv2.VideoWriter(f'data/videos/out_{video_name}', fourcc, 25.0, (1920, 1080))
seen = 0
# 创建一个队列，用于存储检测到的交通灯颜色，长度为5，初始值为unknown
color_queue = deque(maxlen=5)
color_queue.append('unknown')
color_queue.append('unknown')
color_queue.append('unknown')
color_queue.append('unknown')
color_queue.append('unknown')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out = run(frame=frame)
    seen += 1
    if len(out) == 6:
        x1, y1, x2, y2, color, conf = out
        color_queue.popleft()
        color_queue.append(color)
        # 如果队列中有red，且red的数量大于1，则认为是红灯
        if 'red' in color_queue and color_queue.count('red') > 1:
            color_max = 'red'
        # 如果队列中有green，且green的数量大于2，则认为是绿灯
        elif 'green' in color_queue and color_queue.count('green') > 3:
            color_max = 'green'
        else:
            color_max = 'unknown'
        print(f"Frame: {seen}, {color_max}")
        cv2.putText(frame, f"Frame: {seen}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if color_max != 'unknown':
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2)
            cv2.putText(frame, color_max, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        color_queue.popleft()
        color_queue.append('unknown')
        print(f"Frame: {seen}, {out}")
        cv2.putText(frame, f"Frame: {seen}, {out}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    # out_video.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
