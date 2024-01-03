import pandas as pd
import argparse
import os
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R


parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="--dir <path of directory in which you have trajectories output>")
parser.add_argument("--par", help="Param to explore in testing.", default='')
parser.add_argument("--dataset", required=True, help="Dataset name", type=str)
args = parser.parse_args()
absolute_path = args.dir
par = args.par
result_name_fix = "CameraTrajectory" + par + ".txt"
result_name_times_fix = "CameraTrajectory_fix" + par + ".txt"
dataset_name = args.dataset

names = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']

df_result = pd.read_csv(os.path.join(absolute_path, result_name_fix), ' ', header=None, index_col=False, names=names)
df_result.drop(df_result.head(1).index,inplace=True)
df_result['timestamp'] *= 1e-9

x_rot, y_rot, z_rot = 0, 0, 0
if "tum" in dataset_name.lower():
    x_rot = 90
    y_rot = 180
elif dataset_name.lower() == "euroc":
    z_rot = 270
elif dataset_name.lower() == "indoor_factory" or dataset_name.lower() == "office_building" or dataset_name.lower() == "office_space" or dataset_name.lower() == "abandoned_factory" or dataset_name.lower() == "city_park" or dataset_name.lower() == "urbancity":
    y_rot = 270
    z_rot = 270

rot_matrix = R.from_euler("xyz", [x_rot, y_rot, z_rot], degrees=True).as_matrix()

qx_ind = df_result.columns.get_loc("qx")
qy_ind = df_result.columns.get_loc("qy")
qz_ind = df_result.columns.get_loc("qz")
qw_ind = df_result.columns.get_loc("qw")

for i in range(len(df_result)):
    curr_matrix = R.from_quat([df_result.iloc[i, qx_ind], df_result.iloc[i, qy_ind], df_result.iloc[i, qz_ind], df_result.iloc[i, qw_ind]]).as_matrix()
    new_matrix = np.dot(curr_matrix, rot_matrix)
    new_quat = R.from_matrix(new_matrix).as_quat()

    df_result.iloc[i,qx_ind] = new_quat[1]
    df_result.iloc[i,qy_ind] = new_quat[0]
    df_result.iloc[i,qz_ind] = new_quat[2]
    df_result.iloc[i,qw_ind] = new_quat[3]

df_result.to_csv(os.path.join(absolute_path, result_name_times_fix), sep=' ', header=False, index=False)
