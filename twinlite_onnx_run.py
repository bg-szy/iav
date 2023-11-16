import time

import numpy as np
import onnxruntime
import torch
import cv2


def preprocess(src):
    src = cv2.resize(src, (640, 360))
    image = src[:, :, ::-1].transpose((2, 0, 1))
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)  # add a batch dimension
    image = image / 255.0
    image = np.array(image, dtype=np.float16)
    return src, image


# load twinlite.onnx model
session = onnxruntime.InferenceSession("twinlite_fp16.onnx", providers=["CUDAExecutionProvider"])
print(session.get_providers())
input_name = [input.name for input in session.get_inputs()][0]
output_name = [output.name for output in session.get_outputs()]
print(input_name, output_name)
# image
# ['da', 'll']

cap = cv2.VideoCapture('data/videos/TLD120new.mp4')
seen = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_0 = cv2.resize(frame, (640, 360))
    image = img_0[:, :, ::-1].transpose((2, 0, 1))
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)  # add a batch dimension
    image = image / 255.0
    image = np.array(image, dtype=np.float16)
    start_time = time.time()
    da, ll = session.run(output_name, {input_name: image})
    end_time = time.time()
    da_tensor = torch.from_numpy(da)
    ll_tensor = torch.from_numpy(ll)
    _, da_predict = torch.max(da_tensor, 1)
    _, ll_predict = torch.max(ll_tensor, 1)
    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_0[DA > 100] = [255, 0, 0]
    img_0[LL > 100] = [0, 255, 0]
    total_time += (end_time - start_time)
    fps = 1.0 / (end_time - start_time)
    cv2.putText(img_0, f"fps: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('twinlite', img_0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"total time: {total_time:.2f}")
cap.release()
cv2.destroyAllWindows()




