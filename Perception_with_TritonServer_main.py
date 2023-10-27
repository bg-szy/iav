import time

import Perception_with_TritonServer
from Perception_with_TritonServer import *

if __name__ == '__main__':
    # color queue
    color_queue = Perception_with_TritonServer.get_color_queue(5, 'unknown')
    # video prepare
    use_camera = False
    save_out_video = False
    video = 'data/videos/TLD120new.mp4' if not use_camera else 0
    cap = cv2.VideoCapture(video)
    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(f'data/videos/perception_fit_deg2_TLD120new.mp4', fourcc, 25.0, (640, 360))
    # perception preparation
    triton_server, model_twinlite, M, roi = Perception_with_TritonServer.perception_preparation()
    # frame loop
    seen = 0
    prev_points_for_fit_0 = [deque(maxlen=10) for _ in range(3)]
    prev_points_for_fit_1 = [deque(maxlen=10) for _ in range(3)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        seen += 1
        time_start = time.time()
        tl, road = Perception_with_TritonServer.perception_out(frame, triton_server, model_twinlite, M,
                                                               prev_points_for_fit_0, prev_points_for_fit_1)
        prev_points_for_fit_0[2].extend(prev_points_for_fit_0[1])
        prev_points_for_fit_0[1].extend(prev_points_for_fit_0[0])
        prev_points_for_fit_0[0].extend(road[-2][:10])
        prev_points_for_fit_1[2].extend(prev_points_for_fit_1[1])
        prev_points_for_fit_1[1].extend(prev_points_for_fit_1[0])
        prev_points_for_fit_1[0].extend(road[-1][:10])
        color = tl[-1]
        color_queue.popleft()
        color_queue.append(color)
        color_out = get_queue_color(color_queue)
        time_cost = time.time() - time_start
        fps = 1 / time_cost
        out_text = [f"fps: {fps:.2f}", f"frame: {seen}",
                    f"stop line: {round(road[1], 2) if road[1] > 0 else None}",
                    f"stop_left_end: {road[2] if road[1] > 0 and len(road)==4 else None}",
                    f"stop_right_end: {road[3] if road[1] > 0 and len(road)==4 else None}",
                    f"traffic light: {color_out}"]
        Perception_with_TritonServer.put_text(road[0], out_text)
        cv2.imshow('Perception', road[0])
        out_video.write(road[0]) if save_out_video else None
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


