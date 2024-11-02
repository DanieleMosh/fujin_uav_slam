import pandas as pd

# Data dir format for ETH's 'The EuRoC MAV Dataset' in .bag data format
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
