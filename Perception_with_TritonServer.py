"""
Traffic Light >> YOLOv5m
Stop Line and Lane line >> TwinLite
模型串行
"""
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import torch
from sklearn.cluster import DBSCAN

from models import TwinLite as net
from models.common import DetectMultiBackend
from utils.general import check_img_size, Profile, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox


# utils
def get_traffic_light_color(im0, x1, y1, x2, y2):
    """
    :param im0: image
    :param x1: left top x
    :param y1: left top y
    :param x2: right bottom x
    :param y2: right bottom y
    :return: traffic light color
    """
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


def get_color_queue(max_len, fault_value='unknown'):
    """
    :param max_len: max length
    :param fault_value: fault value
    :return: color queue
    """
    color_queue = deque(maxlen=max_len)
    for i in range(max_len):
        color_queue.append(fault_value)
    return color_queue


def get_queue_color(color_queue):
    """
    :param color_queue: color queue
    :return: color
    """
    if color_queue.count('red') > 1:
        color = 'red'
    elif color_queue.count('green') > 3:
        color = 'green'
    else:
        color = 'unknown'
    return color


def yolo_predict(triton_client, input_images, input_name="images", output_name="output0", fp="FP16"):
    inputs = []
    outputs = []

    inputs.append(grpcclient.InferInput(input_name, [*input_images.shape], fp))

    # data initialize
    inputs[-1].set_data_from_numpy(input_images)
    outputs.append(grpcclient.InferRequestedOutput(output_name))
    results = triton_client.infer(model_name="yolov5", inputs=inputs, outputs=outputs)

    return results.as_numpy(output_name)


def detect_traffic_light(triton_server, image_0, imgsz, stride, conf_thres=0.25, iou_thres=0.45, max_det=1000,
                         classes=[0, 1, 2, 3, 5, 7, 9, 10], agnostic_nms=False, augment=False):
    """
    :param model: yolo model
    :param image_0: image :param imgsz: size
    :param stride: stride
    :param conf_thres: confidence threshold
    :param iou_thres: iou threshold
    :param max_det: max detection
    :param classes: classes
    :param agnostic_nms: agnostic nms
    :param augment: augment
    :return: x1, y1, x2, y2, color, conf
    """
    frame = image_0.copy()
    im = letterbox(frame, imgsz, stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    # dt = (Profile(), Profile(), Profile())
    # with dt[0]:
    #     im = torch.from_numpy(im).to(model.device)
    #     im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    #     im /= 255
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dim
    # with dt[1]:
    #     pred = model(im, augment=augment, visualize=False)
    # with dt[2]:
    #     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    pred = yolo_predict(triton_server, im)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    for i, det in enumerate(pred):  # per image
        im0 = frame.copy()
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if c == 9:
                    x1, y1, x2, y2 = [coord.item() for coord in xyxy]
                    color = get_traffic_light_color(im0, x1, y1, x2, y2)
                    return x1, y1, x2, y2, color
    return f"unknown"


def lane_element_detect(model, img0, Matrix, prev_points_for_fit_0, prev_points_for_fit_1, roi=None):
    img = img0.copy()
    if roi is not None:
        img = cv2.resize(img, (640, 360))
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [roi], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
        img_rs = img.copy()
    else:
        img = cv2.resize(img, (640, 360))
        img_rs = img.copy()

    img = img[:, :, ::-1].transpose([2, 0, 1])
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    # x0 = img_out[0]
    x1 = img_out[1]

    # _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    # DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    # 对DA进行腐蚀和膨胀以及边缘平滑处理，降低锯齿
    # DA = post_process(DA, size_e=3, size_g=5, size_ed=5)
    # cv2.warpPerspective的flags有INTER_NEAREST、INTER_LINEAR、INTER_AREA、INTER_CUBIC、INTER_LANCZOS4
    flag = cv2.INTER_LINEAR
    # DA_ipm = cv2.warpPerspective(DA, Matrix, (640, 360), flags=flag)
    LL_ipm = cv2.warpPerspective(LL, Matrix, (640, 360), flags=flag)
    # 对DA进行腐蚀和膨胀以及边缘平滑处理，降低锯齿
    # DA_ipm = post_process(DA_ipm, size_e=7, size_g=5, size_ed=3)
    # 对LL_ipm的每一行像素值为255的点进行计数，若当前行的像素值为255的点数大于阈值200，则将当前行的所有点的像素信息存入stop中，
    # 否则，将当前点的像素信息存入stop中之后全部置零
    stop = []
    threshold_s = 120
    # threshold_l = 35
    for i in range(360):
        if np.sum(LL_ipm[i] == 255) > threshold_s:
            stop.append(LL_ipm[i])
        else:
            stop.append(np.zeros(640))

    stop = np.array(stop)
    # 构建一个矩形区域，此区域包含stop的所有像素点，且每个方向上都有一定的余量
    # 获取stop中所有点坐标x和y的最大值和最小值，以此构建矩形区域
    indices = np.where(stop == 255)
    boarder = 30
    if np.any(indices):
        stop_x_min = np.min(np.where(stop == 255)[1])
        stop_x_max = np.max(np.where(stop == 255)[1])
        stop_y_min = np.min(np.where(stop == 255)[0])
        stop_y_max = np.max(np.where(stop == 255)[0])
        left_top = (max(stop_x_min - boarder, 0), max(stop_y_min - boarder, 0))
        right_bottom = (min(stop_x_max + boarder, 640), min(stop_y_max + boarder, 360))
        stop_left_end, stop_right_end = get_start_end_point(stop_x_min, stop_x_max, stop_y_min, stop_y_max)
    else:
        left_top = (0, 0)
        right_bottom = (0, 0)
        stop_left_end, stop_right_end = (0, 0), (0, 0)

    trans_ratio = 0.031
    bias = 2.1
    dist_SL = -300
    for i in range(359, -1, -1):
        if np.sum(stop[i] == 255) > 0:
            dist_SL = 359 - i
            break
    dist = dist_SL * trans_ratio + bias
    img_rs_ipm = cv2.warpPerspective(img_rs, Matrix, (640, 360), flags=flag)
    # DA_ipm[LL_ipm == 255] = 0
    # img_rs_ipm[DA_ipm > 100] = [0, 0, 255]

    img_rs_ipm[stop == 255] = [255, 0, 0]
    # lane定义为LL_ipm中stop_area区域置零后的图像
    lane = LL_ipm.copy()
    lane = pixel_set_zero(left_top, right_bottom, lane)
    # Get coordinates of non-zero pixels in the lane
    coords = np.column_stack(np.nonzero(lane))
    down_sample = 5
    coords = coords[::down_sample]

    if len(coords) <= 40 // down_sample:
        return img_rs_ipm, dist_SL

    db = DBSCAN(eps=9, min_samples=3).fit(coords)
    lanes = [coords[db.labels_ == i] for i in range(max(db.labels_) + 1)]
    # Get the number of points in each lane
    num_points = np.array([len(lane) for lane in lanes])
    # Get the mean number of points in a lane
    mean_num_points = num_points.mean()
    # Loop over each lane and remove it if it has less than mean_num_points * factor
    factor = 0.4
    lanes = np.array(lanes, dtype=object)[num_points > mean_num_points * factor].tolist()
    # get mid lanes
    mid_lanes = get_mid_lane(lanes)
    # get the side lanes
    if not mid_lanes:
        side_lanes = lanes
    else:
        side_lanes = [lane for lane in lanes if not any(np.array_equal(lane, mid_lane) for mid_lane in mid_lanes)]
    # Define colors for lanes
    center_color = (0, 0, 255)
    mid_color = (255, 0, 0)
    side_color = (0, 255, 0)
    # Display lanes with different colors
    if mid_lanes:
        points_for_fit_0, points_for_fit_1 = get_points_for_fit(mid_lanes)
        if not points_for_fit_0 or not points_for_fit_1 or len(points_for_fit_0) < 10 or len(points_for_fit_1) < 10:
            points_for_fit_0 = points_for_fit_modify(points_for_fit_0, prev_points_for_fit_0)
            points_for_fit_1 = points_for_fit_modify(points_for_fit_1, prev_points_for_fit_1)
    else:
        points_for_fit_0 = points_for_fit_modify(None, prev_points_for_fit_0)
        points_for_fit_1 = points_for_fit_modify(None, prev_points_for_fit_1)
    # calculate center points of mid-lanes
    center_points = []
    for i in range(len(points_for_fit_0)):
        center_points.append([(points_for_fit_0[i][0] + points_for_fit_1[i][0]) / 2,
                              (points_for_fit_0[i][1] + points_for_fit_1[i][1]) / 2])
    draw_fitted_lane(img_rs_ipm, points_for_fit_0, color=mid_color)
    draw_fitted_lane(img_rs_ipm, points_for_fit_1, color=mid_color)
    draw_fitted_lane(img_rs_ipm, center_points, color=center_color)
    # draw circles for points for fit
    for point in points_for_fit_0:
        cv2.circle(img_rs_ipm, (int(point[1]), int(point[0])), 3, (0, 0, 255), -1)
    for point in points_for_fit_1:
        cv2.circle(img_rs_ipm, (int(point[1]), int(point[0])), 3, (0, 0, 255), -1)
    for point in center_points:
        cv2.circle(img_rs_ipm, (int(point[1]), int(point[0])), 3, (0, 255, 0), -1)
    if mid_lanes:
        for i, lane in enumerate(mid_lanes):
            lane = np.array(lane)
            img_rs_ipm[lane[:, 0], lane[:, 1]] = mid_color
    if side_lanes:
        for i, lane in enumerate(side_lanes):
            lane = np.array(lane)
            img_rs_ipm[lane[:, 0], lane[:, 1]] = side_color

    return img_rs_ipm, dist, stop_left_end, stop_right_end, points_for_fit_0, points_for_fit_1


def get_start_end_point(stop_x_min, stop_x_max, left_top_y, right_bottom_y):
    """
    :param stop_x_min: stop x min
    :param stop_x_max: stop x max
    :param left_top_y: left top y
    :param right_bottom_y: right bottom y
    :return: left end point, right end point
    """
    left_end_point = (stop_x_min, int((left_top_y + right_bottom_y) / 2))
    right_end_point = (stop_x_max, int((left_top_y + right_bottom_y) / 2))
    return left_end_point, right_end_point


def get_mid_lane(lanes, window_mid=320):
    """
    :param lanes: points set of each lane
    :param window_mid: the x-coordinate of the window center
    :return: mid-lane
    """
    if len(lanes) < 2:
        return None
    lanes.sort(key=lambda x: x[0][0])
    left_lanes = [lane for lane in lanes if lane[:, 1].min() <= window_mid]
    right_lanes = [lane for lane in lanes if lane[:, 1].min() > window_mid]
    if not left_lanes or not right_lanes:
        return None
    left_lane = min(left_lanes, key=lambda lane: window_mid - lane[:, 1].min())
    right_lane = min(right_lanes, key=lambda lane: lane[:, 1].min() - window_mid)
    return [left_lane, right_lane]


def get_points_for_fit(mid_lanes, points_used=10):
    """
    :param mid_lanes: mid-lanes
    :param points_used: num of points used for fit
    :return: points for fit
    """
    points_for_fit_0, points_for_fit_1 = [], []
    height_0 = mid_lanes[0][:, 0].max() - mid_lanes[0][:, 0].min()
    height_1 = mid_lanes[1][:, 0].max() - mid_lanes[1][:, 0].min()
    per_height_0 = int(height_0 / points_used)
    per_height_1 = int(height_1 / points_used)
    thresh_height = 60
    if per_height_0 < thresh_height / points_used or per_height_1 < thresh_height / points_used:
        return None, None
    for i in range(points_used):
        height_range_0 = [per_height_0 * i, per_height_0 * (i + 1)]
        height_range_1 = [per_height_1 * i, per_height_1 * (i + 1)]
        point_list_0 = [point for point in mid_lanes[0] if height_range_0[0] <= point[0] <= height_range_0[1]]
        point_list_1 = [point for point in mid_lanes[1] if height_range_1[0] <= point[0] <= height_range_1[1]]
        if len(point_list_0) == 0 or len(point_list_1) == 0:
            continue
        point_0 = np.mean(point_list_0, axis=0)
        point_1 = np.mean(point_list_1, axis=0)
        points_for_fit_0.append(point_0)
        points_for_fit_1.append(point_1)

    return points_for_fit_0, points_for_fit_1


def points_for_fit_modify(points_for_fit, prev_points_for_fit):
    if not points_for_fit or len(points_for_fit) < 8:
        points_for_fit = []
        if prev_points_for_fit[0] and prev_points_for_fit[1] and prev_points_for_fit[2]:
            for i in range(len(prev_points_for_fit[0])):
                point = prev_points_for_fit[0][i]
                points_for_fit.append(point)
            print("points_for_fit_modify")
        elif prev_points_for_fit[0]:
            points_for_fit = prev_points_for_fit[0]
    return points_for_fit


def points_check(lane_points, thresh=10):
    """
    :param lane_points: points of lane
    :param thresh: threshold
    :return: True or False
    """
    return True if lane_points and len(lane_points) >= thresh else False


def get_lane_fit_coefficients(points, deg=2):
    """
    :param points: points
    :param deg: degree
    :return: coefficients
    """
    # if not points_check(points):
    #     return None
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    coefficients = np.polyfit(x, y, deg)
    return coefficients


def get_fitted_lane_point(y, coefficients):
    """
    :param y: y
    :param coefficients: coefficients
    :return: fitted lane point location
    """
    return [int(sum(coefficient * y ** i for i, coefficient in enumerate(coefficients[::-1]))), y]


def is_point_in_img(point):
    """
    :param point: point
    :return: True or False
    """
    return True if 0 <= point[0] < 640 and 0 <= point[1] < 360 else False


def draw_fitted_lane(img, lane_points_for_fit, color):
    """
    :param img: image
    :param lane_points_for_fit: points for fit
    :param color: color
    """
    coefficients = get_lane_fit_coefficients(lane_points_for_fit)
    if coefficients is not None:
        for y in range(10, 350):
            point = get_fitted_lane_point(y, coefficients)
            if is_point_in_img(point):
                img[point[1], point[0]] = color


def post_process(src, size_e, size_g, size_ed):
    """
    :param src: source image
    :param size_e: erode kernel size
    :param size_g: gaussian kernel size
    :param size_ed: dilate kernel size
    :return: processed image
    """
    kernel_e = np.ones((size_e, size_e), np.uint8) if size_e and size_e != 0 else None
    kernel_ed = np.ones((size_ed, size_ed), np.uint8) if size_ed and size_ed != 0 else None
    if kernel_e is not None:
        src = cv2.erode(src, kernel_e, iterations=1)
    if size_g and size_g != 0:
        src = cv2.GaussianBlur(src, (size_g, size_g), 0)
    if kernel_ed is not None:
        src = cv2.erode(src, kernel_ed, iterations=1)
        src = cv2.dilate(src, kernel_ed, iterations=1)
    return src


def pixel_set_zero(left_top, right_bottom, src_bin):
    """
    :param left_top: left top point
    :param right_bottom: right bottom point
    :param src_bin: source binary image
    :return: processed source binary image
    """
    if left_top == (0, 0) and right_bottom == (0, 0):
        return src_bin
    for i in range(left_top[1], right_bottom[1]):
        for j in range(left_top[0], right_bottom[0]):
            src_bin[i][j] = 0
    return src_bin


def perception_preparation():
    """
    :return: yolo model, twinlite model, image size, stride, M, roi
    """
    # dir prepare
    # FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[0]  # YOLOv5 root directory
    # if str(ROOT) not in sys.path:
    #     sys.path.append(str(ROOT))  # add ROOT to PATH
    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    # twinlite model prepares
    model_twinlite = net.TwinLiteNet()
    model_twinlite = torch.nn.DataParallel(model_twinlite)
    model_twinlite = model_twinlite.cuda()
    model_twinlite.load_state_dict(torch.load('twinlite.pth'))
    model_twinlite.eval()
    # print("twinlite model loaded" if model_twinlite else "twinlite model load failed")

    # twinlite data prepare
    input_pts = np.float32([[100, 200], [540, 200], [540, 360], [100, 360]])
    output_pts = np.float32([[0, 0], [640, 0], [360, 360], [280, 360]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    roi = np.array([[[80, 185], [540, 185], [560, 360], [100, 360]]], dtype=np.int32)

    # yolo model prepares
    # device = select_device('')
    # weights = ROOT / 'yolov5m.pt'
    # data = ROOT / 'data/coco128.yaml'
    # model_yolo = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=True)
    # stride, pt = model_yolo.stride, model_yolo.pt
    # imgsz = (1080, 1920)
    # imgsz = check_img_size(imgsz, s=stride)  # check image size
    # bs = 1  # batch_size
    # model_yolo.warmup(imgsz=(1 if pt or model_yolo.triton else bs, 3, *imgsz))
    # print("yolo model loaded" if model_yolo else "yolo model load failed")
    # triton server connection
    try:
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    return triton_client, model_twinlite, M, roi


def perception_out(frame, triton_server, model_twinlite, M, prev_points_for_fit_0, prev_points_for_fit_1, roi=None):
    traffic_light = detect_traffic_light(triton_server, frame)
    lane_elements = lane_element_detect(model_twinlite, frame, M, prev_points_for_fit_0, prev_points_for_fit_1, roi)

    return traffic_light, lane_elements


def put_text(frame, texts):
    """
    :param frame: image
    :param texts: texts
    :return: image with texts
    """
    position = (20, 40)
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (position[0], position[1] + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return
