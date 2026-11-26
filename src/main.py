import cv2
import numpy as np
import time
import argparse
import yaml
from collections import deque
from dataclasses import replace

from input_process.process_EuRoC_asl import EurocAslSequence
from input_process.tum_rgbd import TumRgbdSequence
from frontend.visual_odometry import VisualOdometry
from mapping.renderer import SlamRenderer
from mapping.gaussian_mapper import GaussianMapper
from mapping.depth import AlignedMonoDepth
from utils.geometry import pose_moved

MIN_BASELINE = 0.05  # metres between two keyframes used to triangulate depth anchors


def run_bag(args, debug):
    """Monocular VO on a ROS bag, with the sparse OpenGL viewer."""
    # imported here: rosbag replaces the logging class with one that hangs when torch is imported
    from input_process.process_EuRoC_bag import BagImageStreamer

    # init renderer
    renderer = None
    render_queue = None
    if not args.no_render:
        renderer = SlamRenderer()
        render_queue = renderer.get_queue()

    # init streamer
    streamer = BagImageStreamer(bag_path=args.bag, topic=args.topic)
    # init visual odometry
    vo = VisualOdometry(render_queue=render_queue)

    if renderer is not None:
        renderer.start()

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

                # run front-end
                matches, keypoints1, keypoints2 = vo.run_frontend_pipeline(frame_pair=frame_pair)

                if debug:
                    display = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                else:
                    display = frame1

                # show frame rate
                frame_new = time.time()
                fps = 1 / (frame_new - frame_start)

                winName = "Output"
                if frame_count % 10 == 1:
                    winTitle = winName + " FPS: {}".format(int(fps))
                    cv2.setWindowTitle(winName, winTitle)
                cv2.imshow(winName, display)

                frame_start = frame_new

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.001) # yield CPU while queue is empty
    finally:
        # stop the streamer
        streamer.stop()
        if renderer is not None:
            renderer.stop()
        cv2.destroyAllWindows()


def run_sequence(args):
    """Build a Gaussian map live from an image sequence.

    Poses come from the dataset's ground truth or from the tracker (--poses vo).
    Depth comes from the dataset's sensor, or from a monocular depth network made
    metric against the previous keyframes (the only option for EuRoC).
    """
    if args.dataset == "tum":
        sequence = TumRgbdSequence(args.path)
    else:
        sequence = EurocAslSequence(args.path)

    mono_depth = AlignedMonoDepth() if args.depth == "model" else None

    mapper = GaussianMapper(stride=args.stride, output=args.output)
    mapper.start()

    vo = VisualOdometry(K=sequence.K) if args.poses == "vo" else None
    recent = deque(maxlen=5)  # the first frame, then keyframes; newest last
    keyframe = None           # latest keyframe (has depth); the tracker's reference
    try:
        for frame in sequence:
            anchors = None  # tracked points with known depth in this frame
            if vo is not None:
                if not recent:  # the first frame defines the world
                    T_wc = frame.T_wc if frame.T_wc is not None else np.eye(4)
                elif keyframe is None:  # monocular start: no depth to track against yet
                    T_wc = vo.initialize(recent[0], frame.image_bgr)
                else:
                    tracked = vo.track(keyframe, frame.image_bgr)
                    T_wc, anchors = (tracked[0], tracked[1:]) if tracked else (None, None)
                if T_wc is None:
                    continue
                frame = replace(frame, T_wc=T_wc)

            if not recent or pose_moved(recent[-1].T_wc, frame.T_wc):
                # second view for triangulation: latest keyframe far enough away to give parallax
                reference = next((k for k in reversed(recent)
                                  if np.linalg.norm(k.T_wc[:3, 3] - frame.T_wc[:3, 3]) > MIN_BASELINE), None)
                depth = mono_depth.estimate(frame, reference, anchors) if mono_depth else frame.depth
                if depth is not None:
                    keyframe = replace(frame, depth=depth)
                    mapper.submit(keyframe)
                if depth is not None or not recent:
                    recent.append(frame)

            if not args.headless:
                cv2.imshow("Output", frame.image_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if not args.headless:
            print("Sequence finished. The map stays in the viewer; press 'q' in the image window or Ctrl+C to save and exit.")
            while cv2.waitKey(50) & 0xFF != ord('q'):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        mapper.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bag", "tum", "euroc"], default="bag",
                        help="bag: monocular VO on a ROS bag; tum / euroc: Gaussian mapping on a "
                             "TUM RGB-D or EuRoC ASL sequence")
    parser.add_argument("-bag", type=str, help="path to a .bag file (--dataset bag)")
    parser.add_argument("-topic", type=str, default="/cam0/image_raw")
    parser.add_argument("--path", type=str, help="sequence folder (--dataset tum|euroc)")
    parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-render', default=False, action='store_true',
                        help='disable the 3D OpenGL viewer')
    parser.add_argument("--depth", choices=["sensor", "model"], default=None,
                        help="Gaussian mapping depth source: dataset depth sensor, or a monocular depth "
                             "network (default: sensor for tum, model for euroc)")
    parser.add_argument("--poses", choices=["gt", "vo"], default="gt",
                        help="Gaussian mapping pose source: dataset ground truth, or visual odometry")
    parser.add_argument("--stride", type=int, default=4,
                        help="Gaussian mapping: one Gaussian per this many pixels in each direction")
    parser.add_argument("--output", type=str, default="gaussian_map.ply",
                        help="Gaussian mapping: where to save the map (3DGS .ply)")
    parser.add_argument("--headless", action="store_true",
                        help="Gaussian mapping: no image window; save and exit when the sequence ends")
    args = parser.parse_args()

    if args.dataset in ("tum", "euroc"):
        if not args.path:
            parser.error(f"--dataset {args.dataset} requires --path")
        if args.dataset == "euroc" or args.depth is None:  # EuRoC has no depth sensor
            args.depth = "model" if args.dataset == "euroc" else "sensor"
        run_sequence(args)
    else:
        if not args.bag:
            parser.error("--dataset bag requires -bag")

        # debug constant from config or arg
        with open("config/config.yaml", "r") as f:
            debug_config = yaml.safe_load(f)["DEBUG"]
        run_bag(args, debug=args.debug or debug_config)

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
