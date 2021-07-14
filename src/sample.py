import argparse
import cv2
import time
import logging
from yolo3.utils.common import setup, track, draw_rect_on_frame


def main(src, device):
    model, classes, colors, tracker = setup(model_path='config/yolov3.cfg', 
                                            model_img_size=(640, 480),
                                            weights_path='weights/yolov3.weights',
                                            device=device,
                                            classes_path='config/coco.names',
                                            deepsort_weight_path='weights/ckpt.t7')
    class_mask = [0, 2, 4]
    cap = cv2.VideoCapture(src)
    fps = 0
    try:
        while True:
            _, frame = cap.read()
            if frame is None: break

            t_start = time.time()
            detections = track(frame, model, classes, class_mask, tracker)
            if detections is None:
                image = frame
            else:
                image = draw_rect_on_frame(frame, detections, classes, colors)
            cv2.putText(image, text=f'FPS: {fps}', org=(3,15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(255, 0, 0), thickness=2)
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
            t_end = time.time()

            proc_time = t_end - t_start
            fps = round(1.0 / proc_time, 2)
            print(f'Processing time = {proc_time}, fps = {fps}')

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Commadline option
    parse = argparse.ArgumentParser()
    parse.add_argument('--gpu', action='store_true')
    parse.add_argument('--video_path', default='data/sample.mp4', type=str)
    args = parse.parse_args()

    device='cuda:0' if args.gpu else 'cpu'
    main(args.video_path, device)
