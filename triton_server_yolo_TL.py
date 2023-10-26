import sys
import os

import cv2
import torch
import numpy as np
import tritonclient.grpc as grpcclient

from utils.augmentations import letterbox


# triton server connection
try:
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
except:
    print("channel creation failed: " + str(Exception))
    sys.exit()

model_name = "yolov5"
input_width = 640
input_height = 640
conf_thresh = 0.45
batch_size = 1
input_shape = [batch_size, 3, input_height, input_width]
input_name = "images"
output_name = "output0"
output_size = 25200
fp = "FP16"
stride = 32
np_dtype = np.float16
IOU_THRESHOLD = 0.45


def predict(input_images, input_name=input_name, output_name=output_name, fp=fp, triton_client=triton_client):
    inputs = []
    outputs = []

    inputs.append(grpcclient.InferInput(input_name, [*input_images.shape], fp))

    # data initialize
    inputs[-1].set_data_from_numpy(input_images)
    outputs.append(grpcclient.InferRequestedOutput(output_name))
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    return results.as_numpy(output_name)


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
        inter_rect_y2 - inter_rect_y1 + 1, 0, None
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
    boxes = prediction[prediction[:, 4] >= conf_thres]
    boxes[:, :4] = xywh2xyxy(boxes[:, :4])
    boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
    confs = boxes[:, 4]
    boxes = boxes[np.argsort(-confs)]
    keep_boxes = []
    while boxes.shape[0]:
        large_overlap = (bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres)
        label_match = np.round(boxes[0, -1]) == np.round(boxes[:, -1])
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        keep_boxes += [boxes[0]]
        boxes = boxes[~invalid]
    boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    return boxes


def preprocess(img, stride=stride, input_width=input_width, input_height=input_height, np_dtype=np_dtype):
    img = letterbox(img, max(input_width, input_height), stride=stride, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np_dtype)
    img = img / 255.0
    img = img.reshape([1, *img.shape])
    return img


def post_process(output, origin_h, origin_w, input_height=input_height, input_width=input_width):
    boxes = non_max_suppression(output, origin_h, origin_w, conf_thres = conf_thresh, nms_thres = IOU_THRESHOLD)
    result_boxes = boxes[:, :4] if len(boxes) else np.array([])
    result_scores = boxes[:, 4] if len(boxes) else np.array([])
    result_classid = boxes[:, 5] if len(boxes) else np.array([])

    result_boxes = scale_boxes((input_height, input_width), result_boxes, (origin_h, origin_w))
    return result_boxes, result_scores, result_classid


def postprocess(host_outputs, batch_origin_h, batch_origin_w, output_size=output_size, min_accuracy=0.5):
    output = host_outputs[0]
    answer = []
    valid_scores = []
    for i in range(batch_size):
        result_boxes, result_scores, result_classid = post_process(output[i * output_size:(i + 1) * output_size], batch_origin_h, batch_origin_w)
        for box, score in zip(result_boxes, result_scores):
            if score > min_accuracy:
                answer.append(box)
                valid_scores.append(score)
    return answer, valid_scores


def draw_boxes(image, coords, scores):
    box_color = (51, 51, 255)
    font_color = (255, 255, 255)

    line_width = max(round(sum(image.shape) / 2 * 0.0025), 2)
    font_thickness = max(line_width - 1, 1)
    draw_image = image.copy()

    if coords and len(coords):
        for idx, tb in enumerate(coords):
            if tb[0] >= tb[2] or tb[1] >= tb[3]:
                continue
            obj_coords = list(map(int, tb[:4]))

            # bbox
            p1, p2 = (int(obj_coords[0]), int(obj_coords[1])), (
                int(obj_coords[2]),
                int(obj_coords[3]),
            )
            cv2.rectangle(
                draw_image,
                p1,
                p2,
                box_color,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )

            # Conf level
            label = str(int(round(scores[idx], 2) * 100)) + "%"
            w, h = cv2.getTextSize(label, 0, fontScale=2, thickness=3)[
                0
            ]  # text width, height
            outside = obj_coords[1] - h - 3 >= 0  # label fits outside box

            w, h = cv2.getTextSize(
                label, 0, fontScale=line_width / 3, thickness=font_thickness
            )[
                0
            ]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(draw_image, p1, p2, box_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                draw_image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                line_width / 3,
                font_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )
    return draw_image


# image data
src = f"data/videos/TLD120new.mp4"
cap = cv2.VideoCapture(src)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess(frame)
    pred = predict(img)
    boxes, scores = postprocess(pred, frame.shape[0], frame.shape[1])
    debug_image = draw_boxes(frame, boxes, scores)

    cv2.imshow('Video', debug_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
