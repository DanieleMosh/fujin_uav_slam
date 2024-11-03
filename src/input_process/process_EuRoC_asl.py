import pandas as pd

# Data dir format for ETH's 'The EuRoC MAV Dataset' in ASL data format
# https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

# defapt paths for testing
DATA_ROOT = 'data/'
RECORDING_PATH = 'mav03/'
CAM0_PATH = DATA_ROOT + RECORDING_PATH + 'cam0/'
CAM0_IMAGE_PATH = CAM0_PATH + 'data/'
CAM0_DATA_CSV = CAM0_PATH + 'data.csv'
CAM1_PATH = DATA_ROOT + RECORDING_PATH + 'cam1/'
CAM1_IMAGE_PATH = CAM1_PATH + 'data/'
CAM1_DATA_CSV = CAM1_PATH + 'data.csv'

def read_stream(path="") -> pd.DataFrame:
    # read image data file names from data.csv
    cam0_data = None

    if not path:
        path = CAM0_DATA_CSV

    try :
        cam0_data = pd.read_csv(path)
        print(f'Data loaded from {path}')
    except:
        print(f'Could not load data from {path}!')

    # print(cam0_data)
    return path, cam0_data

    # for i in range(len(cam_data) - 1):
    #     frame_count += 1

    #     timestamp1 = cam_data.iloc[i]['#timestamp [ns]']
    #     filename1 = cam_data.iloc[i]['filename']
        
    #     timestamp2 = cam_data.iloc[i + 1]['#timestamp [ns]']
    #     filename2 = cam_data.iloc[i + 1]['filename']

    #     image1_path = f"{video_path}{filename1}"
    #     frame1 = Image.open(image1_path)

    #     image2_path = f"{video_path}{filename2}"
    #     frame2 = Image.open(image2_path)
