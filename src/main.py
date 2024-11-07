import pandas as pd
import cv2
from PIL import Image
import numpy as np
import time
import copy
import argparse
import yaml

from input_process.process_EuRoC_bag import BagImageStreamer
from frontend.visual_odometry import VisualOdometry

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-bag", type=str)
    parser.add_argument("-topic", type=str, default="/cam0/image_raw")
    parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    bag_path = args.bag
    bag_topic = args.topic

    # debug constant from config or arg
    DEBUG = False

    with open("config/config.yaml", "r") as f:
        debug_config = yaml.safe_load(f)["DEBUG"]
        if args.debug or debug_config:
            DEBUG = True
        f.close()

    # init streamer
    streamer = BagImageStreamer(bag_path=bag_path, topic=bag_topic)
    # init visual odometry
    vo = VisualOdometry()

    streamer.start() # start image stream

    frame_start = time.time()
    frame_count = 0

    try:
        while True:
            # Get the next pair of frames
            frame_pair = streamer.get_next_frame_pair()
            
            if frame_pair:
                frame_count += 1

                frame1, frame2 = frame_pair
                # Display the two frames side-by-side or process them
                combined = np.hstack((frame1, frame2))  # Combine frames horizontally

                # show frame rate
                frame_new = time.time()
                fps = 1 / (frame_new - frame_start)

                winName = "Output"
                if frame_count % 10 == 1:
                    winTitle = winName + " FPS: {}".format(int(fps))
                    cv2.setWindowTitle(winName, winTitle)
                cv2.imshow(winName, combined)

                # run front-end
                vo.run_frontend_pipeline(frame_pair=frame_pair)

                frame_start = copy.deepcopy(frame_new)

                # Press 'q' to quit
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.07) # wait if queue empty
    finally:
        # stop the streamer
        streamer.stop()
        cv2.destroyAllWindows()

    #     img1_cv = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
    #     img2_cv = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2BGR)

    #     # feature extractraction
    #     orb = cv2.ORB_create(nfeatures=1000)
    #     keypoints1, descriptors1 = orb.detectAndCompute(img1_cv, None)
    #     keypoints2, descriptors2 = orb.detectAndCompute(img2_cv, None)


    #     keyppoint_image1 = cv2.drawKeypoints(img1_cv, keypoints1, 0, (0, 255, 0))
    #     keyppoint_image2 = cv2.drawKeypoints(img2_cv, keypoints2, 0, (0, 255, 0))

    #     cv2.imshow("Keypoint video Stream 1", keyppoint_image1)
    #     cv2.imshow("Keypoint video Stream 2", keyppoint_image2)

    #     # feature matching for 2 consecutive frames

    #     # initialize the BFMatcher with default params
    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #     # match descriptors between frame I and frame I+1
    #     matches = bf.match(descriptors1, descriptors2)

    #     # sort matches by distance (best matches first)
    #     matches = sorted(matches, key=lambda x: x.distance)

    #     # parameters for lucas kanade optical flow
    #     lk_params = dict(winSize=(15, 15),
    #                     maxLevel=2,
    #                     criteria=(cv2.TERM_CRITERIA_EPS |   cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #     feature_params = dict( maxCorners = 100, 
    #                     qualityLevel = 0.3, 
    #                     minDistance = 7, 
    #                     blockSize = 7 )
    #     img_gray = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
    #     points1 = cv2.goodFeaturesToTrack(img_gray, mask = None, 
    #                             **feature_params) 
        
    #     # Calculate optical flow
    #     points2, status, err = cv2.calcOpticalFlowPyrLK(img1_cv, img2_cv, points1, None, **lk_params)
        
    #     # Select good points
    #     good_new = points2[status.ravel() == 1]
    #     good_old = points1[status.ravel() == 1]

    #     point1_flatten = points1.reshape(-1, 2)
    #     point2_flatten = points2.reshape(-1, 2)
    #     print(point1_flatten[0])


    #     # keyppoint_new_image1 = copy.deepcopy(img1_cv)
    #     # for point_idx in range(len(point1_flatten)):
    #     #     x1, y1 = point1_flatten[point_idx]
    #     #     x2, y2 = point2_flatten[point_idx]
    #     #     keyppoint_new_image1 = cv2.circle(keyppoint_new_image1, (int(x1), int(y1)), 3, (0,0,255), 2)
    #     #     keyppoint_new_image1 = cv2.circle(keyppoint_new_image1, (int(x2), int(y2)), 3, (0,255,0), 2)

    #     # keyppoint_new_image1 = cv2.drawKeypoints(img1_cv, points1, 0, (0, 255, 0))
    #     # keyppoint_new_image2 = cv2.drawKeypoints(img2_cv, points2, 0, (0, 255, 0))

    #     img_matches = cv2.drawMatches(img1_cv, points1, img2_cv, points2, good_new, None, matchColor=(0, 100, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #     frame_new = time.time()
    #     fps = 1 / (frame_new - frame_start)

    #     winName = "Output"
    #     if frame_count % 10 == 1:
    #         winTitle = winName + " FPS: {}".format(int(fps))
    #         cv2.setWindowTitle(winName, winTitle)
    #     cv2.imshow(winName, img_matches)
    #     # cv2.imshow("Keypoint video Stream 2", keyppoint_new_image2)

    #     frame_start = copy.deepcopy(frame_new)

    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         break

    # cv2.destroyAllWindows()
