import rosbag
import numpy as np
import cv2

def decode_ros_image_msg(msg):
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

def stream_images_from_bag(bag_path, topic="/cam0/image_raw"):
    # open bag file
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            # decode the image message
            cv_image = decode_ros_image_msg(msg)
            
            # skip message if failed
            if cv_image is None:
                continue
            
            # Display the image using OpenCV
            cv2.imshow("Image Stream", cv_image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break 

    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    bag_file_path = "path-to-bag.bag"
    stream_images_from_bag(bag_file_path, topic="/cam0/image_raw")