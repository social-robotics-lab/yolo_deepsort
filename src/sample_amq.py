import argparse
import cv2
import time
import logging
from yolo3.utils.common import setup, track, draw_rect_on_frame
from yolo3.utils.amq import AmqClient

def main(src, device, amq):
    model, classes, colors, tracker = setup(model_path='config/yolov3.cfg', 
                                            model_img_size=(640, 480),
                                            weights_path='weights/yolov3.weights',
                                            device=device,
                                            classes_path='config/coco.names',
                                            deepsort_weight_path='weights/ckpt.t7')
    class_mask = [0, 2, 4]
    cap = cv2.VideoCapture(src)
    fps = 0
    skip_frames = 0
    try:
        while True:
            _, frame = cap.read()
            if frame is None: break

            if skip_frames > 0:
                skip_frames = skip_frames - 1
                continue

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
            skip_frames = int(30 * proc_time)
            print(f'Processing time = {proc_time}, fps = {fps}, Skip frames = {skip_frames}')
            if detections is not None and len(detections) > 0:
                ret = [(int(d[0]), int(d[1]), int(d[2]), int(d[3]), str(int(d[4]))) for d in detections]
                print(ret)
                amq.publish(topic='HumanDetected', data=ret)

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Commadline option
    parse = argparse.ArgumentParser()
    parse.add_argument('--gpu', action='store_true')
    parse.add_argument('--gst_host', required=True, type=str)
    parse.add_argument('--gst_port', default='5000', type=str)
    parse.add_argument('--amq_host', required=True, type=str)
    parse.add_argument('--amq_port', default='61613', type=str)
    args = parse.parse_args()
    amq = AmqClient(args.amq_host, args.amq_port)
    device='cuda:0' if args.gpu else 'cpu'
    elements = [
        'tcpclientsrc host={} port={}'.format(args.gst_host, args.gst_port),
        'application/x-rtp-stream,encoding-name=JPEG',
        'rtpstreamdepay',
        'rtpjpegdepay',
        'jpegdec',
        'videoconvert',
        'appsink'
    ]
    src = ' ! '.join(elements)
    main(src, device, amq)
