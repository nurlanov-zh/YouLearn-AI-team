import argparse
import logging
import time
from enum import Enum

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


# logger = logging.getLogger('TfPoseEstimator-Video')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)


def run_and_save(path_to_video=""):
    fps_time = 0
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default=path_to_video)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    # logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    # print(w, h)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=False)
    cap = cv2.VideoCapture(args.video)

    frame_i = 0
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        im_w, im_h = image.shape[:2]
        print(image.shape)
        print(np.max(image), np.min(image))
        if im_w < im_h:
            # Crop it
            target_ratio = 432.0 / 368
            im_h_new = int(im_w / target_ratio)
            delta_h = im_h - im_h_new
            image = image[:, delta_h // 2: im_h - (delta_h // 2), :]

        print(image.shape)
        image = cv2.resize(image, (368, 432))

        print(image.shape)
        print(np.max(image), np.min(image))

        if ret_val is False:
            print("Can not read cap!")
            exit()
        frame_i += 1

        # humans = e.inference(image)
        humans = e.inference(image, resize_to_default=True, upsample_size=1.0)

        ############# My code here #########
        if len(humans) >= 1:
            main_human = humans[0]
            print(main_human.body_parts.keys())
            print(main_human)
            print(main_human.body_parts)
            out_tensor = []
            for i in range(18):
                if i in main_human.body_parts.keys():
                    body_part = main_human.body_parts[i]
                    x, y, score = body_part.x, body_part.y, body_part.score
                    print("Body part : ", CocoPart(i))
                    print(x, y, score)
                    out_tensor.append([x, y, score])
                else:
                    print("Cannot find Body part : ", CocoPart(i))
                    out_tensor.append([0, 0, 0])
        else:
            print("frame_i: ", frame_i, "couldn't find any human")
        ####################################

        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
# logger.debug('finished+')


if __name__ == "__main__":
    run_and_save("../data/watching_attentively/watching_attentively.mp4")
