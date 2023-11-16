import time

import openpyxl
import torch
import numpy as np
import cv2

from models import TwinLite as net

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('twinlite.pth'))
model.eval()
total_time = 0

cap = cv2.VideoCapture('data/videos/TLD120new.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 360))
    img_rs = img.copy()
    img = img[:, :, ::-1].transpose([2, 0, 1])
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    start_time = time.time()
    with torch.no_grad():
        img_out = model(img)
        end_time = time.time()
    x0 = img_out[0]
    x1 = img_out[1]
    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]
    total_time += (end_time - start_time)
    fps = 1.0 / (end_time - start_time)
    cv2.putText(img_rs, f"fps: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('twinlite', img_rs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"total time: {total_time:.2f}")
cap.release()
cv2.destroyAllWindows()
