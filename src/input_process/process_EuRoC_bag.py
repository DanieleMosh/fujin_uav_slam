import rosbag
import numpy as np
import cv2
import multiprocessing as mp
import time

class BagImageStreamer:
    def __init__(self, bag_path, topic="/cam0/image_raw"):
        self.bag_path = bag_path
        self.topic = topic
        self.queue = mp.Queue()  # multiprocess queue to store frames
        self.stop_event = mp.Event()  # event to signal stopping
        self.thread = mp.Process(target=self._stream_images) # process used for streaming


    def _decode_ros_image_msg(self, msg):
        encoding = msg.encoding
        width = msg.width
        height = msg.height
        data = msg.data

        # adapt to different formats
        try:
            if encoding == 'rgb8':
                cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif encoding == 'bgr8':
                cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            elif encoding == 'mono8':
                cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            elif encoding == 'mono16':
                cv_image = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
            else:
                print(f"Unsupported encoding type: {encoding}")
                return None
        except Exception as e:
            print(f"Error decoding image data: {e}")
            return None

        return cv_image

    def _stream_images(self):

        with rosbag.Bag(self.bag_path, 'r') as bag:
            frame_pair = []

            for topic, msg, t in bag.read_messages(topics=[self.topic]):
                # stop streaming if the stop event is set
                if self.stop_event.is_set():
                    break

                # decode the image message into cv2 frame
                cv_image = self._decode_ros_image_msg(msg)
                
                # skip if decoding failed
                if cv_image is None:
                    continue

                # add frame to the temporary pair list
                frame_pair.append(cv_image)

                # if two frames are ready -> put them in the queue as a pair
                if len(frame_pair) == 2:
                    self.queue.put(tuple(frame_pair))
                    frame_pair.pop(0)  # keep only the last frame

                # add stream latency
                time.sleep(0.05)

    # start image streaming in a separate process
    def start(self):
        self.process.start()

    def stop(self):
        self.stop_event.set()
        self.process.join()

    def get_next_frame_pair(self):
        if not self.queue.empty():
            return self.queue.get()
        else:
            return None