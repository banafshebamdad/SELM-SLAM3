# Read Text Files with Pandas using read_csv()

# importing pandas
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from pyquaternion import Quaternion
from evo.tools import file_interface
import warnings

warnings.filterwarnings("ignore")

#traj = file_interface.read_tum_trajectory_file("corridor4poses.tum") # EuRoC.tum or tum.tum
#traj.timestamps = traj.timestamps * 1e9
#file_interface.write_tum_trajectory_file("Camera_gt_adjusted.tum", traj)

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="--dir <path of directory in which you have trajectories output>")
parser.add_argument("--dataset", required=True, help="Dataset name", type=str)
#parser.add_argument("--par", help="Param to explore in testing.", default='')

args = parser.parse_args()
absolute_path = args.dir
dataset_name = args.dataset

#par = args.par
#result_name_fix = "traj" + par + ".txt"
#result_name_times_fix = "traj_times_fix" + par + ".txt"

# read text file into pandas DataFrame
df = pd.read_csv("KeyFrameTrajectory.txt", sep=" ", header=None)
data = []
#print(df[0])
df[0] /= 1e9

#df_pose = df.drop(df.columns[0], axis=1, inplace=False) # Rimozione del timestamp
#q = df.drop(df.columns[0:3], axis=1, inplace=False) # Mantiene qx qy qz qw
#pose = df.drop(df.columns[3:7],axis=1,inplace=False) # Mantiene x y z

# Euler rotation angles
x_rot = 0
y_rot = 0
z_rot = 0

if dataset_name.lower() == "kitti":
    x_rot = 0
    y_rot = 0
    z_rot = 0
if dataset_name.lower() == "tum":
    x_rot = 0
    y_rot = 0
    #df[0]/=1e18
elif dataset_name.lower() == "euroc":
    z_rot = 270
    #df[0]/=1e18
rot_mat = R.from_euler("xyz", [x_rot, y_rot, z_rot], degrees=True).as_matrix()
# Quaternions conversion
for i in range(len(df)):
    r = R.from_quat([df.iloc[i][4], df.iloc[i][5], df.iloc[i][6], df.iloc[i][7]]).as_matrix() # Get rotation matrix from q = qx + i*qy + j*qz + k*qw

    # Rotation matrix (EuRoC: 270 deg around z;
    #                  Tum: 90 deg around x;
    #                         180 deg around y)
    new_r = np.dot(r, rot_mat) # Get new rotation matrix new_r
    new_q = R.from_matrix(new_r).as_quat() # Get quaternion from new_r
    # print(new_quat) # verify new quaternion values
    # print(R.from_matrix(r).as_quat()) # Verify quaternion values

    df.iloc[i, 4] = new_q[0]
    df.iloc[i, 5] = new_q[1]
    df.iloc[i, 6] = new_q[2]
    df.iloc[i, 7] = new_q[3]
    #print(df.loc[i,])
#new_df = pd.DataFrame(data)
print(df.head)
path = '/home/ubuntu/SuperORB_SLAM3'
traj_fixed = "KeyFrame_trajectory_fixed.txt"
df.to_csv(os.path.join(path, traj_fixed), sep=' ', header=False,index=False)# Saving new trajectory file
