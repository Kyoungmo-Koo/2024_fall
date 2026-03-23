
"""simulation program for the raus system integration testing 
"""

# from vortex_get_vol_data import OCTEngine 

import pyrealsense2 as rs
import numpy as np
import cv2 
import igmr_robotics_toolkit
import open3d
import os
from random import seed, uniform
from threading import Thread
import threading
from random import seed, uniform
import scipy 
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import torch
from datetime import datetime
from matplotlib.patches import Circle
from skimage.transform import resize
import mediapipe as mp
from skimage.transform import resize
from scipy.signal import savgol_filter
import yaml

# igmr-robotics-toolkits related work
from typing import List, Optional
import igmr_robotics_toolkit.util.default_logging
from igmr_robotics_toolkit.viewer.core import create_simple_viewer
from igmr_robotics_toolkit.collision import Collider
# from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.viewer.widget import PathWidget, PointCloudWidget, TransformWidget, RobotWidget, TransformListWidget, LineWidget
from igmr_robotics_toolkit.control.simulator import Simulator
# from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.motion.trajectory import TrajectoryGenerator
from igmr_robotics_toolkit.viewer.motion import show_trajectory
from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.motion.path import ik_path, check_joint_path
from igmr_robotics_toolkit.math import hinv
from scipy.spatial.transform import Rotation, Slerp
from igmr_robotics_toolkit.viewer.core import _T
from panda3d.core import NodePath
from igmr_robotics_toolkit.viewer.core import create_simple_viewer
from igmr_robotics_toolkit.viewer.widget import ControlledRobotWidget, PathWidget, TransformWidget
from igmr_robotics_toolkit.control.simple import PointToPoint
from igmr_robotics_toolkit.math import average_pose
from igmr_robotics_toolkit.control.action import ActionProgram
from igmr_robotics_toolkit.control.action.position_trajectory import BufferedJointTrajectoryAction, BufferedCartesianTrajectoryAction

# third-party
import socket

# klampt math module 
from klampt.math import so3 

# ultrasound module
# us system 
from ultrasound_ge_class import Ultrasound, FORMAT_CFM, FORMAT_NAMES

import shutil
import time 

LINEAR = 1
ARC = 2
INTEGRATED = 3

# ========== Settings ==========
SERIAL_NUMBER = "234422060685"
RESOLUTION = (1280, 720)
SAVE_DIR = "20260323_robot_deform2"
SAVE_DIR2 = "20260323_image_deform2"

# RealSense / point cloud controls
SUBSAMPLE = 4
Z_CLIP = 1.5
VIS = False

# 2D projection image controls
GRID_SIZE = 640
XY_RANGE = 0.30
FLIP_Y = True
GAUSSIAN_BLUR = 3
SAVE_OVERLAY = True

# MediaPipe controls
MAX_NUM_HANDS = 2
STATIC_IMAGE_MODE = True
MODEL_COMPLEXITY = 1
MIN_DET_CONF = 0.3
MIN_TRACK_CONF = 0.3

# record_i = 0

seed(1337)
np.random.seed(0)

# UR3 2017332424 RUSS
delta_theta = [ 4.61804064568337139e-05, -0.0596917792801850075, 0.187982530479293558, -0.128421417771249463, 1.86920644791972534e-05, -3.70876954857309694e-05]
delta_a = [ 3.74645469127644245e-05, 0.000432751255883601083, 0.0015154637341757704, 3.54739391322167398e-05, 5.12409078229780638e-05, 0]
delta_d = [ 0.000223224169787122895, -6.29131988413453058, 8.56941731705468257, -2.27804644471935935, -7.31385696645381334e-05, 0.000410720409206782877]
delta_alpha = [ 0.000118200678426161332, 0.0023051124802664618, 0.0120016019438704737, 0.000466335407550255709, -0.000233982646400399119, 0]

from PyUniversalRobot import kinematics
table = kinematics.UR3.table

for (dh, dt, dr, dd, da) in zip(table, delta_theta, delta_a, delta_d, delta_alpha):
    dh.t += dt
    dh.r += dr
    dh.d += dd
    dh.a += da

ur3 = kinematics.UR3
generic_ik = kinematics.Kinematics('UR3 generic', table, ur3)
generic_ik.set_max_deviation(10/180*np.pi)
ptol = 1e-5
atol = 1e-4
etol = 1e-6
generic_ik.solver.set_error_tolerances(etol, np.sqrt(etol / ptol**2), 180 / np.pi * np.sqrt(etol / atol**2))
generic_ik.solver.set_time_limit(1)
generic_ik.solver.set_random_seed(0)

def log_timestamp(txt_path="timestamps.txt"):
    """
    Appends the current timestamp to a text file.
    Creates the file if it does not exist.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms precision
    with open(txt_path, "a") as f:
        f.write(now + "\n")

def robust_ik_path(q_start: np.ndarray, ee_path: List[np.ndarray], kin) -> List[np.ndarray]:
    # check that initial joint config matches starting pose
    if not np.allclose(ee_path[0], kin.forward(q_start), atol=1e-5):
        raise RuntimeError(f'initial joint configuration does not match starting pose: {q_start}')

    q_path = [q_start]
    perturbations = [] 

    for (i, pose) in enumerate(ee_path[1:]):
        a = 0
        e = None
        while True:
            if a > 0.1:
                raise RuntimeError(f"Failed to find nearby IK solution at waypoint {i+1} after high perturbation (a = {a:.3f})")
            q_ref = q_path[-1] + [uniform(-a, a) for _ in range(len(q_path[-1]))]
            try:
                q_new = kin.inverse_nearest(pose, q_ref)
                diff = np.mean(np.abs(q_new - q_path[-1]) ** 2)
                if diff < 0.5:
                    q_path.append(q_new)
                    break  # Accept and exit loop
            except kinematics.InverseException as exc:
                e = exc  # Store error in case we eventually fail
            a += 0.001  # Increment perturbation regardless of success/failure

        perturbations.append(a)
        if e and len(q_path) == i + 1:  # If no valid solution was appended
            raise e
    print(f"Max perturbation used: {max(perturbations)}")
    return q_path

def rs_to_xyzrgb(color_frame, depth_frame):
    """Convert aligned RealSense frames to (x,y,z,r,g,b)."""
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    color_image = np.asanyarray(color_frame.get_data())
    H, W, _ = color_image.shape
    u = (tex[:, 0] * W).astype(np.int32)
    v = (tex[:, 1] * H).astype(np.int32)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(vtx[:, 2]) & (vtx[:, 2] > 0)
    if Z_CLIP is not None:
        valid &= (vtx[:, 2] <= Z_CLIP)
    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)
    vtx = vtx[valid]
    u, v = u[valid], v[valid]
    rgb = color_image[v, u, :][:, ::-1].astype(np.float32) / 255.0
    xyzrgb = np.column_stack((vtx, rgb)).astype(np.float32)
    return xyzrgb


def pcs_to_xy_image(pcs, grid_size=640, xy_range=0.3, flip_y=True, subsample=None, gaussian_blur=0):
    """Project list of xyzrgb point clouds to a single 2D XY image."""
    if len(pcs) == 0:
        return None, None, None

    all_xyz = np.concatenate([pc[:, :3] for pc in pcs], axis=0)
    all_rgb = np.concatenate([pc[:, 3:] for pc in pcs], axis=0)
    if subsample and subsample > 1:
        all_xyz = all_xyz[::subsample]
        all_rgb = all_rgb[::subsample]

    x, y = all_xyz[:, 0], all_xyz[:, 1]
    m = (x >= -xy_range) & (x <= xy_range) & (y >= -xy_range) & (y <= xy_range)
    if not np.any(m):
        blank = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        return blank, cv2.cvtColor(blank, cv2.COLOR_BGR2RGB), all_xyz

    x, y, rgb = x[m], y[m], all_rgb[m]
    xi = np.clip(((x + xy_range) / (2 * xy_range) * grid_size).astype(np.int32), 0, grid_size - 1)
    yi = np.clip(((y + xy_range) / (2 * xy_range) * grid_size).astype(np.int32), 0, grid_size - 1)
    if flip_y:
        yi = (grid_size - 1) - yi

    lin = yi * grid_size + xi
    count = np.bincount(lin, minlength=grid_size * grid_size).astype(np.float32)
    sum_r = np.bincount(lin, weights=rgb[:, 0], minlength=grid_size * grid_size).astype(np.float32)
    sum_g = np.bincount(lin, weights=rgb[:, 1], minlength=grid_size * grid_size).astype(np.float32)
    sum_b = np.bincount(lin, weights=rgb[:, 2], minlength=grid_size * grid_size).astype(np.float32)
    count[count == 0] = 1.0
    r = (sum_r / count).reshape(grid_size, grid_size)
    g = (sum_g / count).reshape(grid_size, grid_size)
    b = (sum_b / count).reshape(grid_size, grid_size)

    img_rgb = np.stack([r, g, b], axis=-1)
    img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if gaussian_blur and gaussian_blur % 2 == 1:
        img_bgr = cv2.GaussianBlur(img_bgr, (gaussian_blur, gaussian_blur), 0)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb, all_xyz


def find_joint_xyz(all_xyz, xy_norm, grid_size, xy_range, flip_y=True, radius_mm=4.0):
    """Estimate (x,y,z) by averaging nearby 3D points within radius_mm."""
    x_norm, y_norm = xy_norm
    x = (x_norm - 0.5) * 2 * xy_range
    y = (0.5 - y_norm if flip_y else y_norm - 0.5) * 2 * xy_range
    radius = radius_mm / 1000.0
    dist2 = (all_xyz[:, 0] - x) ** 2 + (all_xyz[:, 1] - y) ** 2
    near = dist2 <= radius ** 2
    if np.any(near):
        pts = all_xyz[near]
        return np.median(pts[:, :3], axis=0)
    else:
        return None


def visualize_and_detect(pcs, save_dir, grid_size=GRID_SIZE, xy_range=XY_RANGE,
                         flip_y=FLIP_Y, subsample=SUBSAMPLE, gaussian_blur=GAUSSIAN_BLUR):
    """Combine PCs → XY image, run Mediapipe Hands, draw MCP/PIP joints, print 3D positions."""
    img_bgr, img_rgb, all_xyz = pcs_to_xy_image(
        pcs, grid_size=grid_size, xy_range=xy_range, flip_y=flip_y,
        subsample=subsample, gaussian_blur=gaussian_blur
    )
    if img_bgr is None:
        print("No points to project; skipping visualization/detection.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    )
    results = hands.process(img_rgb)
    hands.close()

    joint_xyz_dict = {}
    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            MCP_IDXS = [2, 5, 9, 13, 17]
            PIP_IDXS = [3, 6, 10, 14, 18]
            print(f"\n🖐 Hand {hand_id+1}: Estimated Joint Positions (mm)")
            print("Joint\t\tX(mm)\tY(mm)\tZ(mm)")
            print("-" * 40)
            for i, (mcp_idx, pip_idx) in enumerate(zip(MCP_IDXS, PIP_IDXS), start=1):
                for j_type, idx in zip(["MCP", "PIP"], [mcp_idx, pip_idx]):
                    lm = hand_landmarks.landmark[idx]
                    avg_xyz = find_joint_xyz(all_xyz, (lm.x, lm.y), grid_size, xy_range, flip_y)
                    if avg_xyz is not None:
                        x_mm, y_mm, z_mm = avg_xyz * 1000
                        print(f"{j_type}{i}\t\t{x_mm:7.2f}\t{y_mm:7.2f}\t{z_mm:7.2f}")
                        joint_xyz_dict[f"{j_type}{i}"] = avg_xyz
                        cx, cy = int(lm.x * img_bgr.shape[1]), int(lm.y * img_bgr.shape[0])
                        color = (255, 0, 0) if j_type == "MCP" else (0, 255, 0)
                        cv2.circle(img_bgr, (cx, cy), 6, color, -1)
                        cv2.putText(img_bgr, f"{j_type}{i}", (cx + 8, cy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        print(f"{j_type}{i}\t\t---\t---\t(No nearby points)")
    else:
        print("MediaPipe: no hands detected.")

    np.save(os.path.join(SAVE_DIR, "joint_xyz_dict.npy"), joint_xyz_dict)

    # cv2.imshow("XY Projection + MCP/PIP Joints", img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if SAVE_OVERLAY:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "xy_projection_with_MCP_PIP.png")
        cv2.imwrite(out_path, img_bgr)
        print(f"Saved overlay: {out_path}")

    return joint_xyz_dict

import torch.nn as nn

class CircleRegressorCNN_K10(nn.Module):
    def __init__(self):
        super().__init__()
        window_size = 12
        padding_size = int(window_size // 2)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=window_size, padding=padding_size), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=window_size, padding=padding_size), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=window_size, padding=padding_size), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=window_size, padding=padding_size), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=window_size, padding=padding_size), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.head(self.features(x))

def save_us_img_tmp(self): 

    is_break_inner_loop = False
    is_exit             = False
    is_exit             = False
    is_exit_cfm         = False
    is_exit_bmode       = False

    # save the image
    for frame in self._obj_ultrasound:
        for channel in range(frame.images.shape[-1]):

            if frame.format == FORMAT_CFM:
                cmap = matplotlib.cm.get_cmap('doppler')
                (vmin, vmax) = (-50, 50)
            else:
                cmap= matplotlib.cm.get_cmap('Greys_r')
                (vmin, vmax) = (0, 255)

            # image scaling
            img = frame.images[..., channel] 
            img_raw_for_npy = img.copy() 
            img = cmap((img.astype(float) - vmin) / (vmax - vmin), bytes=True)

            # save the b-mode image 
            if frame.format != FORMAT_CFM:
                img_save_bmode = img
                # if np.max(img_save_bmode.ravel()) > -256:
                is_exit_bmode = True
                # cv2.imwrite( self._path_data_unique_folder + str( self.idx_in_loop ) + "_bmode.png", img_save_bmode)
                # Added 20241016
                # np.save(f"{self._path_data_unique_folder}{self.idx_in_loop}_bmode.npy", img_save_bmode)
                img_save_bmode_uint16 = img_save_bmode.astype(np.uint16)
    
                # Save the image as a .npy file in uint8 format
                # np.save(f"{self._path_data_unique_folder}{self.idx_in_loop}_bmode.npy", img_save_bmode_uint16)
                if self.initial_check == 1 :
                    np.save(f"{self._path_data_unique_folder}initial_check_bmode.npy", img_save_bmode_uint16[:, :, 0])
                    self.initial_check  = 0
                else: 
                    np.save(f"{self._path_data_unique_folder}{self.idx_in_loop}_bmode.npy", img_save_bmode_uint16[:, :, 0])
    
                print(self._tform_ee_to_us_opt_res)
                print((img_save_bmode_uint16[:, :, 0]).shape)


            """cfm config"""
            if self._is_collect_cfm:
                # save the cfm image 
                if frame.format == FORMAT_CFM:
                    if self._mode_ge_img == 'BMIAF':
                        img_save_cfm = img_raw_for_npy.astype(np.uint16)
                    elif self._mode_ge_img == 'CFM':
                        img_save_cfm = img_raw_for_npy.astype(np.int8)
                    print(img_save_cfm.shape)
                    # if np.max(img_save_cfm.ravel()) > -256:
                    is_exit_cfm = True
                    #Edited 20241016
                    #cv2.imwrite( self._path_data_unique_folder + str( self.idx_in_loop ) + "_cfm.png", img_save_cfm)
                    # cv2.imwrite( self._path_data_unique_folder + "channel_" + str(channel) + "_format_" + str(frame.format) + "_idx_" + str( self.idx_in_loop ) + "_cfm.png", img_save_cfm)
                    #Edited 20241016

                    # np.save( self._path_data_unique_folder + "channel_" + str(channel) + "_format_" + str(frame.format) + "_idx_" + str( self.idx_in_loop ) + "_cfm.npy", img_save_cfm)
                    #np.save( self._path_data_unique_folder + str( self.idx_in_loop ) + "_cfm.npy", img_raw_for_npy ) 
                    if ((channel == 0) and (frame.format == 6)):
                        np.save( self._path_data_unique_folder + "idx_" + str( self.idx_in_loop ) + "_cfm.npy", img_save_cfm)
            else: 
                if is_exit_bmode == True: 
                    is_exit = True
                    break

            # both image should be saved 
            if self._is_collect_cfm:
                if (is_exit_bmode == True) and (is_exit_cfm == True): 
                    is_exit = True
                    break

        if is_exit == True: 
            break 

    # display the 3d sensor fusion point cloud 
    # transfer mapping function 


class raus_systsem_test():

    def __init__(self): 

        # UR3 robot
        self.path_ur3_config     = "./database/system/UR3/robot.yaml" 
        self.path_ur5e_config    = "./database/system/UR5e/robot.yaml" 

        # ip address
        self._host_ip_ur3        = '169.254.88.100'
        # self._host_ip_ur5e       = '10.162.76.24'

    def print_current_ee_cen_pts(self):
        """print the current ee center point"""

        tcp_in_world_tform    = self._ee_tform @ self._ee_to_tcp_tform
        # val_test = self._ee_tform @ self._ee_to_tcp_tform
        pts_tcp_in_world      = tcp_in_world_tform[0:3,3]
        rot_tcp_in_world      = tcp_in_world_tform[0:3,0:3]

        return pts_tcp_in_world, rot_tcp_in_world

    def detect_boundary_rowavg_robust(
        self,
        row_avg,
        smooth_win=61,
        poly=3,
        search_range=(120, 900),  # we will interpret search_range[0] as ROW_START (y0)
        guard=0,                  # not needed for first-crossing logic; kept for API compatibility
        sustain_len=15,           # matches your old default behavior better
        thr_mult=1.5              # THR_MULT in your old script
    ):
        
        row_avg = np.asarray(row_avg, dtype=np.float32)
        H = int(len(row_avg))
        if H == 0:
            return None

        # ---- sanitize window/poly ----
        smooth_win = int(smooth_win)
        if smooth_win <= 1:
            smooth = row_avg.copy()
        else:
            if smooth_win % 2 == 0:
                smooth_win += 1
            # cap window to valid odd length <= H
            if smooth_win > H:
                smooth_win = H if (H % 2 == 1) else (H - 1)
            smooth_win = max(smooth_win, 3)
            poly = int(poly)
            poly = min(max(poly, 1), smooth_win - 1)
            smooth = savgol_filter(row_avg, smooth_win, poly)

        # ---- search bounds ----
        y0, y1 = search_range
        y0 = int(max(y0, 0))
        y1 = int(min(y1, H))   # exclusive upper bound
        if y1 <= y0:
            return None

        # ---- baseline + threshold (THIS is the "2×mean" logic) ----
        baseline = float(np.mean(smooth[y0:]))
        thr = float(thr_mult) * baseline

        # ---- first crossing with sustain ----
        sustain_len = int(sustain_len) if sustain_len is not None else 0

        if sustain_len <= 1:
            idx = np.where(smooth[y0:y1] >= thr)[0]
            return (y0 + int(idx[0])) if idx.size else None

        last_start = y1 - sustain_len
        for y in range(y0, max(last_start, y0)):
            if smooth[y] >= thr and np.all(smooth[y:y + sustain_len] >= thr):
                return int(y)

        return None

    
    def adjust_coordinates(self, img):

        # saved as uint16[:, :, 0] in your code, so likely 2D already
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.float32)

        H = img.shape[0]
        row_avg = img.mean(axis=1)

        # ---- detect boundary row ----
        y = self.detect_boundary_rowavg_robust(
            row_avg,
            smooth_win=61,
            poly=3,
            search_range=(150, min(900, H - 1)),
            guard=5,
            sustain_len=50
            # sustain_frac=0.30
        )

        # ---- initialize reference if first time ----
        return int(y)


    def do_discrete_adjust_at_current_pose(self, z_offset_accum, TARGET_ROW, AXIAL_M_PER_PIXEL):

        # ensure a fresh image is saved
        # time.sleep(0.05)

        y = self.adjust_coordinates(self._obj_ultrasound.save_once3())   # returns boundary row (int) or None
        if y is None:
            return z_offset_accum

        dy_px = int(y) - TARGET_ROW
        dz_m  = -(dy_px) * AXIAL_M_PER_PIXEL  # <-- simplified sign logic

        return z_offset_accum + dz_m


    def print_current_q(self):

        global idx_global
        path_data_local = self._path_data_unique_folder

        # define the robot state 
        state = self._ptp_exp.state

        if not state:
            print('no robot state')
            return
        else: 
            print("the current ID = ", idx_global)
            print("the q_pose = ", state.actual_q)
            
            q_use = state.actual_q
            print("q_use = ", q_use)
            # np.save( path_data_local + str(idx_global) + ".npy", q_use )
            # idx_global += 1

            return state.actual_q 
    def preprocess_like_dataset(self, img_np, H, W):
        if img_np.ndim == 3:
            if img_np.shape[-1] in (3,4):
                img_np = img_np.mean(axis=-1)  # RGB → grayscale
        img_resized = resize(img_np, (H, W), preserve_range=True).astype(np.float32)
        mx = float(img_resized.max())
        if mx > 0:
            img_resized = img_resized / (255.0 if mx > 1.0 else mx)
        print(img_resized.shape)
        img_resized = np.expand_dims(img_resized, axis=0)  # (1,H,W)
        print(img_resized.shape)
        return torch.from_numpy(img_resized.astype(np.float32))

    def analyze_now(self, img):

        cmap= matplotlib.cm.get_cmap('Greys_r')
        (vmin, vmax) = (0, 255)
        print("raw image mean:", np.mean(img))
        print("raw image shape:", img.shape)
        img = cmap((img.astype(float) - vmin) / (vmax - vmin), bytes=True).astype(np.uint16)[:, :, 0]
        print("processed image mean:", np.mean(img))
        print("processed image shape:", img.shape)

        H = self.H
        W = self.W
        MAX_R = self.MAX_R
        
        print("W: ", W)
        print("H: ", H)

        model = self.model

        # if img.ndim == 2:
        #     h, w = img.shape
        #     img_4ch = np.zeros((h, w, 4), dtype=np.float32)
        #     img_4ch[..., 0] = img       # R
        #     img_4ch[..., 1] = img       # G
        #     img_4ch[..., 2] = img       # B
        #     img_4ch[..., 3] = 255.0     # A (alpha)
        # else:
        #     img_4ch = img.astype(np.float32)

        # x = self.preprocess_like_dataset(img_4ch, H, W).unsqueeze(0)  # (1,1,H,W)
        mx = float(img.max())
        if mx > 0:
            img = img / (255.0 if mx > 1.0 else mx)
        np.save("20251030_test_0.npy", img)

        DEVICE = torch.device("cpu")

        img = resize(img, (H, W), preserve_range=True).astype(np.float32)
        mx = float(img.max())
        if mx > 0:
            img = img / (255.0 if mx > 1.0 else mx)
        # Add batch and channel dimensions → (1, 1, H, W)
        x = torch.from_numpy(img[None, None, :, :]).to(DEVICE)

        # Shape (1, 1, H, W)
        # x = torch.from_numpy(img[None, None, :, :]).float()  # stay on CPU
        

        with torch.no_grad():
            pred = model(x).squeeze(0)

        with torch.no_grad():
            pred = model(x).squeeze(0).numpy()

        pred_cx = float(pred[0]) * W
        pred_cy = float(pred[1]) * H
        pred_r  = float(pred[2]) * MAX_R
        print(f"Predicted: cx={pred_cx:.2f}, cy={pred_cy:.2f}, r={pred_r:.2f}")

        # self.initial_check = 1
        # img = self.img
        # img = np.load(f"{self._path_data_unique_folder}initial_check_bmode.npy")
        # print("image collected, analysis start")
        # print(img.shape)

        # Pixel offset from image center
        dx_px = pred_cx - (W / 2)
        dy_px = pred_cy - (H * 3/4)

        # Pixel → real-world scaling (example: 0.025 m for half width/height)
        scale_x = 0.025 / (W)
        scale_y = 0.030 / (H)

        current_state = self._ptp_exp.state
        current_q = current_state.actual_q

        T_ee = self._robot.kinematics.forward(current_q)
        T_translate = np.eye(4)
        T_translate[:3, 3] = (
            T_ee[:3, 1] * (dx_px * scale_x) -
            T_ee[:3, 2] * (dy_px * scale_y)
        )

        T_ee_moved = T_ee @ T_translate

        T_q = self._robot.kinematics.inverse_nearest(T_ee_moved, current_q)

        return T_q, pred_r
    
    def monitor_robot_position(self, stop_event, interval=0.1):
        """Background thread: print TCP position periodically."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        i = 0

        while not stop_event.is_set():
            try:
                # === Get timestamp (HH:MM:SS.microsecond) ===
                current_time = datetime.now()
                ts_str = current_time.strftime("%H:%M:%S.%f")[:-3]  # millisecond precision

                # === Get robot joint and end-effector state ===
                record_state = self._ptp_exp.state
                record_q = np.round(record_state.actual_q, 6)
                record_ee = self._robot.kinematics.forward(record_q)  # 4×4 EE matrix

                # === Save EE matrix to .npy ===
                fname = os.path.join(SAVE_DIR, f"record_{i:03d}.npy")
                np.save(fname, record_ee)

                # === Append timestamp + EE matrix on the same line ===
                with open(os.path.join(SAVE_DIR, "timestamps.txt"), "a") as f:
                    # Flatten matrix for compact single-line storage
                    ee_flat = " ".join(f"{v:.6f}" for v in record_ee.flatten())
                    f.write(f"{i:03d} | {ts_str} | {ee_flat}\n")

                # === Console log ===
                # print(f"[{ts_str}] [Monitor] Saved record_{i:03d}.npy: q = {record_q}")

                i += 1
                time.sleep(interval)

            except Exception as e:
                print("Monitor error:", e)
                time.sleep(0.5)
    def rotate_half_pi_to_start(self, radius_in_m):
        current_state = self._ptp_exp.state
        current_q = current_state.actual_q

        print("Checkpoint : Current_q", current_q)

        check_q = current_q + [0, 0, 0, 0, 0, + np.pi / 2]
        check_ee = self._robot.kinematics.forward(check_q)

        # === Define translation along end-effector's Y axis ===
        dy = radius_in_m  # 2 cm forward in EE frame
        T_shift = np.eye(4)
        T_shift[:3, 3] = [0, dy, 0]

        # === New target pose (in base frame) ===
        T_target = check_ee @ T_shift   # right-multiply = move in EE's local coordinates

        # === Inverse kinematics to get joint target ===
        target_q = self._robot.kinematics.inverse_nearest(T_target, check_q)

        self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input * 25)
        self.current_q = check_q
    
    def to_the_end(self, finger_idx, radius_in_m):
        check_q = self.current_q
        check_ee = self._robot.kinematics.forward(check_q)
        dy = radius_in_m  # 2 cm forward in EE frame

        T_shift = np.eye(4)
        T_shift[:3, 3] = [0, -dy, 0]
        # === New target pose (in base frame) ===
        T_target = check_ee @ T_shift   # right-multiply = move in EE's local coordinates

        # === Inverse kinematics to get joint target ===
        target_q = self._robot.kinematics.inverse_nearest(T_target, check_q)

        if(finger_idx < 3):
            self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input / 8)
        else:
             self._ptp_exp.move_joint(target_q, qd_limits= 2.5 * self._qd_limit_input / 8)
        self.current_q = check_q

    def rotate_half_pi_back(self):
        check_q = self.current_q + [0, 0, 0, 0, 0, np.pi / 2]
        self._ptp_exp.move_joint(check_q, qd_limits=self._qd_limit_input * 3)
        self.current_q = check_q
        
    def control_exp(self):

        # set the mode 
        mode_use                        = "exp"
        # mode_use                        = "sim"
        self._is_ultrasound_mode        = True

        # robot configuration 
        # self._robot                 = load_robot( self.path_ur5e_config ) 
        self._robot                     = load_robot( self.path_ur3_config ) 

        # define the home configuration 
        qs_home                         = [-0.03559095, -1.30845672,  1.61374092, -1.85577661, -1.54902679, -1.61667949]
        self._qs_home                   = qs_home

        self._mode_ge_img               = '2D'

        # setup the ultrasound streaming 
        # unit-1: test the ge-ultrasound
        if self._is_ultrasound_mode: 
            self._display_size              = (500, 600)
            self._host_ge                   = '169.254.107.11'
            # self._mode_ge_img               = 'BMIAF'
            self._mode_ge_img               = '2D'
            # self._mode_ge_img               = 'CFM'
            self._obj_ultrasound            = Ultrasound()
            self._obj_ultrasound.connect( host = self._host_ge )
            self._obj_ultrasound.start( self._mode_ge_img )

        # folder definitions 
        idx_folder_unique_name             = "unit_test_1"
        self._path_data_unique_folder      = "./data_raus/" + idx_folder_unique_name +  "/"
        idx_exist_folder                   = os.path.isdir(self._path_data_unique_folder)
        if idx_exist_folder == False:
            print("The folder does not exist")
            os.mkdir(self._path_data_unique_folder)
        
        # global index
        global idx_global
        idx_global                  = 0 
        
        # exp connection 
        if mode_use == "exp":
            self._ctrl_exp          = self._robot.Controller( host  = self._host_ip_ur3, 
                                                              model = self._robot,
                                                              robot_info=('2017332424', '3.4.1.59')  )
            print("_ctrl_exp = ", self._ctrl_exp._robot_info)
        elif mode_use == "sim": 
            self._ctrl_exp          = Simulator( self._robot )
            self._ctrl_exp._q       = qs_home

        # create the sample viewers 
        (self._window, self._root)  = create_simple_viewer()

        # connect the ptp agent
        self._ptp_exp               = PointToPoint(self._robot, self._ctrl_exp )
        self._ptp_exp.connect()

        # step-2: setup the viewer and window
        self._crwa                  = ControlledRobotWidget(model = self._robot, 
                                                            controller = self._ptp_exp, 
                                                            parent = self._root, 
                                                            frames=[0, self._robot.dof ], 
                                                            show_mode='actual_q')
        
        path                        = PathWidget(parent=self._root)

        # basic settings
        self._robot.visual.parent   = self._root
        self._tform_ee_to_us_opt_res =  np.array([[ 1, 0,  0, 0], [0, 1, 0, 0], [0, 0, 1, 0.146], [ 0, 0, 0, 1]])
        self._tcp_xform             = np.array( self._tform_ee_to_us_opt_res )
        self._tcp_xform2 = np.array([[ 1, 0,  0, 0], [0, 1, 0, 0], [0, 0, 1, 0.146], [ 0, 0, 0, 1]])

        # adjust the speed 
        if mode_use == "sim": 
            self._qd_limit_input     = np.pi / 2
        elif mode_use == "exp": 
            self._qd_limit_input     = np.pi / 60
        self.idx_global_oct_data = 0 

        # define the tform widget globally
        self._scan_tform_widget     = TransformListWidget( parent = self._root, scale = 0.5 )

        mode_scan_use = "data_collection_test"

        if mode_scan_use == "data_collection_test":
            self._is_collect_cfm = False
            Thread(target = self.main_data_platform, daemon=True).start()

        # run the windows
        self._window.run()

    def main_data_platform(self): 
        self.quant_count = 0

        # ckpt_path = "circle_regressor_k10_ninesets_valholdout_mps.pt"
        # ckpt_path = "circle_regressor_k10_10sets_perimage_split_mps.pt"
        # ckpt_path = "circle_regressor_20_sets_permilage_with_koo_mps.pt"
        ckpt_path = "circle_regressor_volunteer_split_w12.pt"

        # Load checkpoint on CPU
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Recreate model and load weights
        model = CircleRegressorCNN_K10()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()   # set to inference mode

        # Retrieve training parameters
        H, W, MAX_R = checkpoint["H"], checkpoint["W"], checkpoint["MAX_R"]

        print(f"Model loaded on CPU: H={H}, W={W}, MAX_R={MAX_R}")
        self.H, self.W, self.MAX_R = H, W, MAX_R
        self.model = model

        print("patient data collection platform")
        time.sleep(1.0)

        self._is_collect_cfm = False

        # --- Get current robot state and pose ---
        initial_state = self._ptp_exp.state
        initial_q = initial_state.actual_q
        initial_ee = self._robot.kinematics.forward(initial_q)
        print("Initial end effector pose:\n", initial_ee)

        q_seed = initial_q

        MCP_target_buffer = []
        MCP_target_buffer_ee = []
        MCP_indices = [0, 2, 4, 6, 8]
        x_offset = 0.00
        y_offset = 0.00

        for i in MCP_indices:
            if i >= len(joint_xyz_robot_np):
                print(f"⚠️ Skipping index {i}: out of range")
                continue

            target_ee = initial_ee.copy()
            target_ee[:2, 3] = joint_xyz_robot_np[i, :2]
            target_ee[0, 3]  = joint_xyz_robot_np[i, 0] + x_offset
            target_ee[1, 3]  = joint_xyz_robot_np[i, 1] + y_offset
            target_ee[2, 3]  = 0.140 + joint_xyz_robot_np[i, 2]
            target_q = self._robot.kinematics.inverse_nearest(target_ee, q_seed)
            target_q[5] = target_q[5] - tilt_rad_buffer[i // 2]

            MCP_target_buffer.append(target_q)
            MCP_target_buffer_ee.append(self._robot.kinematics.forward(target_q))

            # try:
            #     self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input)

            # except Exception as e:
            #     print(f"❌ IK failed at MCP index {i}: {e}")
            #     continue
        
        MCP_target_buffer = np.array(MCP_target_buffer)
        # print(MCP_target_buffer)

        PIP_target_buffer= []
        PIP_target_buffer_ee = []
        PIP_indices = [1, 3, 5, 7, 9]

        for i in PIP_indices:
            if i >= len(joint_xyz_robot_np):
                print(f"⚠️ Skipping index {i}: out of range")
                continue

            target_ee = initial_ee.copy()
            target_ee[:2, 3] = joint_xyz_robot_np[i, :2]
            target_ee[0, 3]  = joint_xyz_robot_np[i, 0] + x_offset
            target_ee[1, 3]  = joint_xyz_robot_np[i, 1] + y_offset
            target_ee[2, 3]  = 0.140 + joint_xyz_robot_np[i, 2]
            target_q = self._robot.kinematics.inverse_nearest(target_ee, q_seed)
            target_q[5] = target_q[5] - tilt_rad_buffer[i // 2]

            PIP_target_buffer.append(target_q)
            PIP_target_buffer_ee.append(self._robot.kinematics.forward(target_q))
            # try:
            #     self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input)

            # except Exception as e:
            #     print(f"❌ IK failed at PIP index {i}: {e}")
            #     continue

        PIP_target_buffer = np.array(PIP_target_buffer)

        # --- Find and Move to the PIP joint with the HIGHEST Z target position ---
        if len(PIP_target_buffer_ee) > 0:
            # 1. Extract Z values from the [2, 3] index (Z-translation) of the 4x4 matrices
            pip_z_values = [target[2, 3] for target in PIP_target_buffer_ee]
            
            # 2. Identify the maximum Z value and its index in the list
            max_z_val = max(pip_z_values)
            max_z_list_idx = pip_z_values.index(max_z_val)
            
            # 3. Map back to the original PIP joint ID and the corresponding joint angles (q)
            target_joint_id = PIP_indices[max_z_list_idx]
            target_q_to_move = PIP_target_buffer[max_z_list_idx]
            
            print(f"🔝 Moving to Highest PIP Target: Joint Index {target_joint_id} at Z = {max_z_val:.4f}m")
            
            # 4. Execute the movement command
            try:
                self._ptp_exp.move_joint(target_q_to_move, qd_limits= 12 * self._qd_limit_input)
                print("✅ Movement to max joint complete.")
            except Exception as e:
                print(f"❌ Movement failed: {e}")
        else:
            print("⚠️ PIP target buffer is empty. No movement performed.")
        
        new_PIP, radius = self.analyze_now(self._obj_ultrasound.save_once())

        self._ptp_exp.move_joint(new_PIP, qd_limits= 12 * self._qd_limit_input)
        time.sleep(1.0)
        new_PIP, radius = self.analyze_now(self._obj_ultrasound.save_once())

        self._ptp_exp.move_joint(new_PIP, qd_limits= 12 * self._qd_limit_input)
        time.sleep(1.0)
        self._obj_ultrasound.save_once()
        T_orig = self._robot.kinematics.forward(PIP_target_buffer[max_z_list_idx])
        T_adj  = self._robot.kinematics.forward(new_PIP)

        adj_ee = T_adj[:3, 3] - T_orig[:3, 3]

        # exit()

        elongated_PIP_target_buffer_ee = []
        elongated_MCP_target_buffer_ee = []
        elongated_PIP_target_buffer = []
        elongated_MCP_target_buffer = []
        elongated_PIP_target_buffer_ee2 = []
        elongated_MCP_target_buffer_ee2 = []
        elongated_PIP_target_buffer2 = []
        elongated_MCP_target_buffer2 = []

        elongation_factor = 1.30   # 20% beyond MCP
        reverse_factor = -0.30     # -0.2 for extrapolation
        addition_PIP = 0.10
        addition_MCP = 0.05

        for i in range(5):
            # Create a copy of the initial ee transformation
            ee_new_MCP = MCP_target_buffer_ee[i].copy()
            ee_new_PIP = PIP_target_buffer_ee[i].copy()

            ee_new_MCP2 = MCP_target_buffer_ee[i].copy()
            ee_new_PIP2 = PIP_target_buffer_ee[i].copy()
            
            # Extract translation vectors
            p_MCP = MCP_target_buffer_ee[i][:3, 3]
            p_PIP = PIP_target_buffer_ee[i][:3, 3]
            
            # Extrapolate beyond MCP along the line (PIP → MCP)
            ee_new_MCP[:3, 3] = p_MCP * elongation_factor + p_PIP * reverse_factor + addition_MCP * (p_PIP - p_MCP)
            ee_new_PIP[:3, 3] = p_PIP * elongation_factor + p_MCP * reverse_factor + addition_PIP * (p_PIP - p_MCP)

            ee_new_MCP2[:3, 3] = p_MCP * (2-elongation_factor) - p_PIP * reverse_factor + addition_MCP * (p_PIP - p_MCP)
            ee_new_PIP2[:3, 3] = p_PIP * (2-elongation_factor) - p_MCP * reverse_factor + addition_PIP * (p_PIP - p_MCP)

            PIP_target_buffer_ee[i][:3, 3] = p_PIP + addition_PIP * (p_PIP - p_MCP)
            PIP_target_buffer[i]           = self._robot.kinematics.inverse_nearest(PIP_target_buffer_ee[i], q_seed)
            MCP_target_buffer_ee[i][:3, 3] = p_MCP + addition_MCP * (p_PIP - p_MCP)
            MCP_target_buffer[i]           = self._robot.kinematics.inverse_nearest(MCP_target_buffer_ee[i], q_seed)

            ee_new_MCP[2, 3] = p_MCP[2].copy() + 0.01
            ee_new_PIP[2, 3] = p_PIP[2].copy()
            ee_new_MCP2[2, 3] = p_MCP[2].copy()
            ee_new_PIP2[2, 3] = p_PIP[2].copy()
            
            # Append to list
            elongated_MCP_target_buffer_ee.append(ee_new_MCP)
            elongated_PIP_target_buffer_ee.append(ee_new_PIP)
            elongated_MCP_target_buffer.append(self._robot.kinematics.inverse_nearest(ee_new_MCP, q_seed))
            elongated_PIP_target_buffer.append(self._robot.kinematics.inverse_nearest(ee_new_PIP, q_seed))

            elongated_MCP_target_buffer_ee2.append(ee_new_MCP2)
            elongated_PIP_target_buffer_ee2.append(ee_new_PIP2)
            elongated_MCP_target_buffer2.append(self._robot.kinematics.inverse_nearest(ee_new_MCP2, q_seed))
            elongated_PIP_target_buffer2.append(self._robot.kinematics.inverse_nearest(ee_new_PIP2, q_seed))

            
        
        elongated_MCP_target_buffer_ee = np.array(elongated_MCP_target_buffer_ee)
        elongated_PIP_target_buffer_ee = np.array(elongated_PIP_target_buffer_ee)
        elongated_MCP_target_buffer = np.array(elongated_MCP_target_buffer)
        elongated_PIP_target_buffer = np.array(elongated_PIP_target_buffer)

        elongated_MCP_target_buffer_ee2 = np.array(elongated_MCP_target_buffer_ee2)
        elongated_PIP_target_buffer_ee2 = np.array(elongated_PIP_target_buffer_ee2)
        elongated_MCP_target_buffer2 = np.array(elongated_MCP_target_buffer2)
        elongated_PIP_target_buffer2 = np.array(elongated_PIP_target_buffer2)

        np.save("MCP_target_buiffer.npy", MCP_target_buffer_ee)
        np.save("PIP_target_buffer.npy", PIP_target_buffer_ee)
        np.save("elongated_MCP_target_buffer_ee.npy", elongated_MCP_target_buffer_ee)
        np.save("elongated_PIP_target_buffer_ee.npy", elongated_PIP_target_buffer_ee)
        np.save("elongated_MCP_target_buffer_ee2.npy", elongated_MCP_target_buffer_ee2)
        np.save("elongated_PIP_target_buffer_ee2.npy", elongated_PIP_target_buffer_ee2)

        log_timestamp()

        new_PIP2_buffer = []
        new_MCP_buffer = []
        new_MCP2_buffer = []
        center_MCP_buffer = []
        robot_speed = 1.0

        for finger_idx in range(5):
            os.makedirs(os.path.join(SAVE_DIR2, f"finger_{finger_idx + 1}"), exist_ok=True)
            self._obj_ultrasound.update_finger_idx(finger_idx + 1)
            log_timestamp()
            # time.sleep(0.5)
            PIP_q =  PIP_target_buffer[finger_idx]

            def rodrigues(axis, theta):
                """Rotation matrix: rotate by theta (rad) about 'axis' (3,)."""
                axis = np.asarray(axis, dtype=float)
                axis = axis / (np.linalg.norm(axis) + 1e-12)

                ax, ay, az = axis
                K = np.array([[0, -az, ay],
                            [az, 0, -ax],
                            [-ay, ax, 0]], dtype=float)
                I = np.eye(3)
                return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

            PIP_rotated_q = PIP_q + [0, 0, 0, 0, 0, + np.pi / 2]
            PIP_rotated_ee =  self._robot._kinematics.forward(PIP_rotated_q)
            PIP_rotated_ee[:3, 3] = PIP_rotated_ee[:3, 3] + adj_ee

            PIP_intermediate = PIP_rotated_ee.copy()
            PIP_intermediate[2, 3] = PIP_rotated_ee[2, 3] + 0.015
            PIP_intermediate_q = self._robot.kinematics.inverse_nearest(PIP_intermediate, PIP_target_buffer[finger_idx])

            self._ptp_exp.move_joint(PIP_intermediate_q, qd_limits=self._qd_limit_input * 12)

            dy = 0.007
            PIP_shift = np.eye(4)
            PIP_shift2 = np.eye(4)
            PIP_shift[:3, 3] = [0, dy, 0]
            PIP_shift2[:3, 3] = [0, - dy, 0]
            # === New target pose (in base frame) ===
            PIP_start = PIP_rotated_ee @ PIP_shift   # right-multiply = move in EE's local coordinates
            PIP_end = PIP_rotated_ee @ PIP_shift2
            PIP_start_q = self._robot.kinematics.inverse_nearest(PIP_start, PIP_q)
            PIP_end_q = self._robot.kinematics.inverse_nearest(PIP_end, PIP_q)

            self._ptp_exp.move_joint(PIP_start_q, qd_limits=self._qd_limit_input * 12)
            # time.sleep(0.5)
            np.save("PIP_start.npy", PIP_start)

            log_timestamp()

            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=self.monitor_robot_position, args=(stop_event, 0.1)
            )
            monitor_thread.start()

            self._obj_ultrasound.save()
            if(finger_idx < 3):
                self._ptp_exp.move_joint(PIP_end_q, qd_limits= 12 * robot_speed * self._qd_limit_input )
            else:
                self._ptp_exp.move_joint(PIP_end_q, qd_limits= 12 * robot_speed * self._qd_limit_input )
            np.save("PIP_end.npy", PIP_end)

            self._obj_ultrasound.no_save()

            stop_event.set()
            monitor_thread.join()

            log_timestamp()

            current_state2 = self._ptp_exp.state
            self.current_q = current_state2.actual_q
            print(self._robot._kinematics.forward(self.current_q))
            print(self._obj_ultrasound._sagittal_analysis2(SAVE_DIR2))

            peak_idx, avg_angle, avg_mid_y_phys = self._obj_ultrasound._sagittal_analysis2(SAVE_DIR2)

            R = PIP_rotated_ee[:3, :3]
            p = PIP_rotated_ee[:3, 3].copy()

            # center of rotation (your definition)
            sagittal_center = p + (0.128 + 0.00 * avg_mid_y_phys) * R[:, 2]
            c = sagittal_center
            axis = R[:, 1]
            theta = avg_angle / 180.0 * np.pi * 1.0

            R_rot = rodrigues(axis, theta)

            # rotate pose about point c
            p_new = c + R_rot @ (p - c)
            R_new = R_rot @ R

            new_sagittal_pivot = PIP_rotated_ee.copy()
            new_sagittal_pivot[:3, :3] = R_new
            new_sagittal_pivot[:3, 3]  = p_new + (avg_mid_y_phys * np.cos(theta) - 1) * 0.01 * R_new[:, 2] - avg_mid_y_phys * np.sin(theta) * 0.01 * R_new[:, 0]

            # new_pip_q = self._robot.kinematics.inverse_nearest(new_sagittal_pivot, PIP_rotated_q)

            # self._ptp_exp.move_joint(new_pip_q, qd_limits= 0.5 * robot_speed * self._qd_limit_input )

            # === New target pose (in base frame) ===
            PIP_start = new_sagittal_pivot @ PIP_shift   # right-multiply = move in EE's local coordinates
            PIP_end = new_sagittal_pivot @ PIP_shift2
            PIP_start_q = self._robot.kinematics.inverse_nearest(PIP_start, PIP_q)
            PIP_end_q = self._robot.kinematics.inverse_nearest(PIP_end, PIP_q)

            # self._ptp_exp.move_joint(PIP_start_q, qd_limits= 12 * self._qd_limit_input)
            # time.sleep(0.5)

            # stop_event = threading.Event()
            # monitor_thread = threading.Thread(
            #     target=self.monitor_robot_position, args=(stop_event, 0.1)
            # )
            # monitor_thread.start()

            # self._obj_ultrasound.save()
            # if(finger_idx < 3):
            #     self._ptp_exp.move_joint(PIP_end_q, qd_limits= 2 * robot_speed * self._qd_limit_input )
            # else:
            #     self._ptp_exp.move_joint(PIP_end_q, qd_limits= 3 * robot_speed * self._qd_limit_input )
            # np.save("PIP_end.npy", PIP_end)

            # self._obj_ultrasound.no_save()

            # stop_event.set()
            # monitor_thread.join()

            # log_timestamp()
            # self._obj_ultrasound.save_once()

            num_steps = 10
            interpolated_points = np.linspace(PIP_start_q, PIP_end_q, num_steps)

            for target_q in interpolated_points:
                self._ptp_exp.move_joint(target_q, qd_limits=12 * self._qd_limit_input)
                time.sleep(0.1)
                self._obj_ultrasound.save_once()
            
            PIP_original_ee =  self._robot._kinematics.forward(PIP_q)
            PIP_original_ee[:3, 3] = PIP_original_ee[:3, 3] + adj_ee
            print("PIP_original_ee: ", PIP_original_ee)
            PIP_original_q = self._robot.kinematics.inverse_nearest(PIP_original_ee, PIP_q)
            print("PIP_original_q : ", PIP_original_q)


            current_state2 = self._ptp_exp.state
            current_q = current_state2.actual_q
            rotated_q = current_q + [0, 0, 0, 0, 0, - np.pi / 2]
            print("rotated_current_ee:", self._robot._kinematics.forward(rotated_q))

            
            cross_sec_ees = self._obj_ultrasound._sagittal_analysis3(SAVE_DIR2, finger_idx * 20 + 3, finger_idx * 20 + 13, new_sagittal_pivot, -10, 35)
            print("target_ee: ", cross_sec_ees[0])
            target_q = self._robot.kinematics.inverse_nearest(cross_sec_ees[0], PIP_q)
            print("target_q: ", target_q )

            # cross_sec_qs_array = np.array(cross_sec_qs)
            # np.save('cross_sec_qs.npy', cross_sec_qs_array)

            # exit()

            cross_sec_qs = []
            old_q = PIP_q

            z_offset_accum = 0.0
            p_accum = np.zeros(3, dtype=float)   # shape: (3,)
            TARGET_ROW = 370
            AXIAL_M_PER_PIXEL = 0.030 / 1083.0

            for i in range(cross_sec_ees.shape[0]):

                # apply accumulated translation
                cross_sec_ees[i, :3, 3:4] = cross_sec_ees[i, :3, 3:4] + p_accum[:, None]

                cross_sec_q = self._robot.kinematics.inverse_nearest(cross_sec_ees[i], old_q)
                self._ptp_exp.move_joint(cross_sec_q, qd_limits=12 * self._qd_limit_input)

                z_offset_accum = 0.0
                z_offset_accum = self.do_discrete_adjust_at_current_pose(
                    z_offset_accum,
                    TARGET_ROW,
                    AXIAL_M_PER_PIXEL
                )

                R_fixed = cross_sec_ees[i, :3, :3].copy()

                # R_fixed[:, 2] has shape (3,), so keep p_accum as (3,)
                p_accum = p_accum - z_offset_accum * R_fixed[:, 2]

                # update pose with new accumulated translation
                cross_sec_ees[i, :3, 3:4] = cross_sec_ees[i, :3, 3:4] + p_accum[:, None]

                cross_sec_q = self._robot.kinematics.inverse_nearest(cross_sec_ees[i], old_q)
                self._ptp_exp.move_joint(cross_sec_q, qd_limits=12 * self._qd_limit_input)
                self._obj_ultrasound.save_once3()

                old_q = cross_sec_q
                time.sleep(0.1)
            
            # exit()
            
            # 20260320 update
            current_state2 = self._ptp_exp.state
            current_q = current_state2.actual_q
            PIP_safe = self._robot._kinematics.forward(current_q)
            PIP_safe[2, 3] = PIP_safe[2, 3] + 0.015
            PIP_safe_q = self._robot.kinematics.inverse_nearest(PIP_safe, PIP_target_buffer[finger_idx])
            self._ptp_exp.move_joint(PIP_safe_q, qd_limits= 12 * self._qd_limit_input)

            os.makedirs(os.path.join(SAVE_DIR2, f"finger_{finger_idx + 6}"), exist_ok=True)
            self._obj_ultrasound.update_finger_idx(finger_idx + 6)
            log_timestamp()

            cross_sec_qs_array = np.array(cross_sec_qs)
            np.save('cross_sec_qs.npy', cross_sec_qs_array)

            print("Done")

            # exit()
        
            MCP_q =  MCP_target_buffer[finger_idx]
            MCP_rotated_q = MCP_q + [0, 0, 0, 0, 0, + np.pi / 2]
            MCP_rotated_ee =  self._robot._kinematics.forward(MCP_rotated_q)
            MCP_rotated_ee[:3, 3] = MCP_rotated_ee[:3, 3] + adj_ee

            MCP_intermediate = MCP_rotated_ee.copy()
            MCP_intermediate[2, 3] = MCP_rotated_ee[2, 3] + 0.015
            MCP_intermediate_q = self._robot.kinematics.inverse_nearest(MCP_intermediate, MCP_target_buffer[finger_idx])

            self._ptp_exp.move_joint(MCP_intermediate_q, qd_limits=self._qd_limit_input * 12)

            dy = 0.007
            MCP_shift = np.eye(4)
            MCP_shift2 = np.eye(4)
            MCP_shift[:3, 3] = [0, dy, 0]
            MCP_shift2[:3, 3] = [0, - dy, 0]
            # === New target pose (in base frame) ===
            MCP_start = MCP_rotated_ee @ MCP_shift   # right-multiply = move in EE's local coordinates
            MCP_end = MCP_rotated_ee @ MCP_shift2
            MCP_start_q = self._robot.kinematics.inverse_nearest(MCP_start, MCP_q)
            MCP_end_q = self._robot.kinematics.inverse_nearest(MCP_end, MCP_q)

            self._ptp_exp.move_joint(MCP_start_q, qd_limits=self._qd_limit_input * 12)
            # time.sleep(0.5)

            log_timestamp()

            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=self.monitor_robot_position, args=(stop_event, 0.1)
            )
            monitor_thread.start()

            self._obj_ultrasound.save()
            if(finger_idx < 8):
                self._ptp_exp.move_joint(MCP_end_q, qd_limits= 12 * robot_speed * self._qd_limit_input )
            else:
                self._ptp_exp.move_joint(MCP_end_q, qd_limits= 12 * robot_speed * self._qd_limit_input )
            np.save("PIP_end.npy", PIP_end)

            self._obj_ultrasound.no_save()

            stop_event.set()
            monitor_thread.join()

            log_timestamp()

            peak_idx, avg_angle, avg_mid_y_phys = self._obj_ultrasound._sagittal_analysis2(SAVE_DIR2)

            R = MCP_rotated_ee[:3, :3]
            p = MCP_rotated_ee[:3, 3].copy()

            # center of rotation (your definition)
            sagittal_center = p + (0.128 + 0.00 * avg_mid_y_phys) * R[:, 2]
            c = sagittal_center
            axis = R[:, 1]
            theta = avg_angle / 180.0 * np.pi * 1.0

            R_rot = rodrigues(axis, theta)

            # rotate pose about point c
            p_new = c + R_rot @ (p - c)
            R_new = R_rot @ R

            new_sagittal_pivot = MCP_rotated_ee.copy()
            new_sagittal_pivot[:3, :3] = R_new
            if finger_idx == 0:
                new_sagittal_pivot[:3, 3]  = p_new + (avg_mid_y_phys * np.cos(theta) - 1.3) * 0.01 * R_new[:, 2] - avg_mid_y_phys * np.sin(theta) * 0.01 * R_new[:, 0]
            else:
                new_sagittal_pivot[:3, 3]  = p_new + (avg_mid_y_phys * np.cos(theta) - 0.9) * 0.01 * R_new[:, 2] - avg_mid_y_phys * np.sin(theta) * 0.01 * R_new[:, 0]

            MCP_start = new_sagittal_pivot @ MCP_shift   # right-multiply = move in EE's local coordinates
            MCP_end = new_sagittal_pivot @ MCP_shift2
            MCP_start_q = self._robot.kinematics.inverse_nearest(MCP_start, MCP_q)
            MCP_end_q = self._robot.kinematics.inverse_nearest(MCP_end, MCP_q)

            # self._ptp_exp.move_joint(MCP_start_q, qd_limits= 12 * self._qd_limit_input)
            # # time.sleep(0.5)

            # stop_event = threading.Event()
            # monitor_thread = threading.Thread(
            #     target=self.monitor_robot_position, args=(stop_event, 0.1)
            # )
            # monitor_thread.start()

            # self._obj_ultrasound.save()
            # if(finger_idx < 8):
            #     self._ptp_exp.move_joint(MCP_end_q, qd_limits= 2 * robot_speed * self._qd_limit_input )
            # else:
            #     self._ptp_exp.move_joint(MCP_end_q, qd_limits= 3 * robot_speed * self._qd_limit_input )

            # self._obj_ultrasound.no_save()

            # stop_event.set()
            # monitor_thread.join()

            # log_timestamp()
            # self._obj_ultrasound.save_once()

            num_steps = 10
            interpolated_points = np.linspace(MCP_start_q, MCP_end_q, num_steps)

            for target_q in interpolated_points:
                self._ptp_exp.move_joint(target_q, qd_limits=12 * self._qd_limit_input)
                time.sleep(0.1)
                self._obj_ultrasound.save_once()
            
            MCP_original_ee =  self._robot._kinematics.forward(MCP_q)
            MCP_original_ee[:3, 3] = MCP_original_ee[:3, 3] + adj_ee
            print("MCP_original_ee: ", MCP_original_ee)
            MCP_original_q = self._robot.kinematics.inverse_nearest(MCP_original_ee, MCP_q)
            print("PIP_original_q : ", MCP_original_q)

            current_state2 = self._ptp_exp.state
            current_q = current_state2.actual_q
            rotated_q = current_q + [0, 0, 0, 0, 0, - np.pi / 2]
            print("rotated_current_ee:", self._robot._kinematics.forward(rotated_q))
            
            cross_sec_ees = self._obj_ultrasound._sagittal_analysis3(SAVE_DIR2, finger_idx * 20 + 13, finger_idx * 20 + 23, new_sagittal_pivot, 0, 35)
            print("target_ee: ", cross_sec_ees[0])
            # target_q = self._robot.kinematics.inverse_nearest(cross_sec_ees[0], MCP_q)
            # print("target_q: ", target_q )

            # cross_sec_qs_array = np.array(cross_sec_qs)
            # np.save('cross_sec_qs.npy', cross_sec_qs_array)

            # exit()

            cross_sec_qs = []
            old_q = MCP_q

            # for i in range(cross_sec_ees.shape[0]):
            #     print(i)
            #     cross_sec_q = self._robot.kinematics.inverse_nearest(cross_sec_ees[i], old_q)
            #     print(cross_sec_q)
            #     if i < 5:
            #         self._ptp_exp.move_joint(cross_sec_q, qd_limits= 12 * self._qd_limit_input)
            #         self._obj_ultrasound.save_once3()
            #     else:
            #         self._ptp_exp.move_joint(cross_sec_q, qd_limits= 12 * self._qd_limit_input)
            #         self._obj_ultrasound.save_once3()
                
            #     old_q = cross_sec_q
            #     time.sleep(0.1)

            z_offset_accum = 0.0
            p_accum = np.zeros(3, dtype=float)   # shape: (3,)
            TARGET_ROW = 370
            AXIAL_M_PER_PIXEL = 0.030 / 1083.0

            for i in range(cross_sec_ees.shape[0]):

                # apply accumulated translation
                cross_sec_ees[i, :3, 3:4] = cross_sec_ees[i, :3, 3:4] + p_accum[:, None]

                cross_sec_q = self._robot.kinematics.inverse_nearest(cross_sec_ees[i], old_q)
                self._ptp_exp.move_joint(cross_sec_q, qd_limits=12 * self._qd_limit_input)

                z_offset_accum = 0.0
                z_offset_accum = self.do_discrete_adjust_at_current_pose(
                    z_offset_accum,
                    TARGET_ROW,
                    AXIAL_M_PER_PIXEL
                )

                R_fixed = cross_sec_ees[i, :3, :3].copy()

                # R_fixed[:, 2] has shape (3,), so keep p_accum as (3,)
                p_accum = p_accum - z_offset_accum * R_fixed[:, 2]

                # update pose with new accumulated translation
                cross_sec_ees[i, :3, 3:4] = cross_sec_ees[i, :3, 3:4] + p_accum[:, None]

                cross_sec_q = self._robot.kinematics.inverse_nearest(cross_sec_ees[i], old_q)
                self._ptp_exp.move_joint(cross_sec_q, qd_limits=12 * self._qd_limit_input)
                self._obj_ultrasound.save_once3()

                old_q = cross_sec_q
                time.sleep(0.1)
            
            # 20260320 update
            current_state2 = self._ptp_exp.state
            current_q = current_state2.actual_q
            MCP_safe = self._robot._kinematics.forward(current_q)
            MCP_safe[2, 3] = MCP_safe[2, 3] + 0.015
            MCP_safe_q = self._robot.kinematics.inverse_nearest(MCP_safe, MCP_target_buffer[finger_idx])
            self._ptp_exp.move_joint(MCP_safe_q, qd_limits= 12 * self._qd_limit_input)

            # exit()

if __name__ == "__main__":
    log_timestamp()

    # ========= 1️⃣ RealSense initialization =========
    os.makedirs(SAVE_DIR, exist_ok=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL_NUMBER)
    config.enable_stream(rs.stream.color, *RESOLUTION, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, *RESOLUTION, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale:.6f} m/unit")

    # ========= 2️⃣ Capture 5 point clouds =========
    print("\n📸 Capturing 5 pointclouds (1s apart)...")
    pcs = []
    for i in range(5):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        c = aligned.get_color_frame()
        d = aligned.get_depth_frame()
        if not c or not d:
            print(f"⚠️ Missing frame at index {i}")
            time.sleep(1)
            continue
        pc = rs_to_xyzrgb(c, d)
        if pc.size > 0:
            pcs.append(pc)
            fname = os.path.join(SAVE_DIR, f"pc_{i}.npy")
            np.save(fname, pc)
            print(f"✅ Saved {fname} ({pc.shape[0]} pts)")
        # time.sleep(1.0)

    pipeline.stop()
    print(f"✅ Finished capturing {len(pcs)} valid pointclouds.\n")

    # exit()

    # ========= 3️⃣ MediaPipe Hand Detection =========
    if len(pcs) == 0:
        print("❌ No valid pointclouds — aborting detection and robot move.")
    else:
        print("🖐 Running MediaPipe hand joint detection...")
        joint_xyz = visualize_and_detect(pcs, SAVE_DIR)
        print("✅ Detection complete. Overlay saved to:", SAVE_DIR)

        # ========= 4️⃣ Transform to robot base frame =========
        # print("\n🔄 Transforming hand joint coordinates to robot base frame...")

        # T_base_from_cam = np.array([
        #     [-0.0022, -0.9990,  0.0450, -0.3994],
        #     [-0.9991,  0.0040,  0.0413, -0.1605],
        #     [-0.0415, -0.0448, -0.9981,  0.6085],
        #     [0,        0,        0,       1.0000]
        # ])

        # est_points = np.load("est_points.npy")
        # true_points = np.load("true_points.npy")
        # joint_xyz_robot = {}
        # for name, p_cam in joint_xyz.items():
        #     p_h = np.append(p_cam, 1.0)  # homogeneous
        #     print(p_h)
        #     p_est = [100 * p_h[0], 100 * p_h[1], 50 - 100 * p_h[2]]
        #     dists = np.linalg.norm(est_points - p_est, axis = 1)
        #     min_idx = np.nanargmin(dists)
        #     expected_real_poisition = true_points[min_idx]
        #     p_h2 = [expected_real_poisition[0] * 0.01, expected_real_poisition[1] * 0.01, ( 50 - expected_real_poisition[2]) * 0.01, 1]
        #     print(p_h2)

        #     p_robot = T_base_from_cam @ p_h
        #     p_robot2 = T_base_from_cam @ p_h2
        #     print("p_robot before optics : ", p_robot)
        #     print("p_robot after optics: ", p_robot2)
        #     joint_xyz_robot[name] = p_robot[:3]
        #     print(f"{name:6s} → Robot frame [x y z] = [{p_robot[0]*1000:8.2f}, {p_robot[1]*1000:8.2f}, {p_robot[2]*1000:8.2f}] mm")

        # joint_xyz_robot_np = np.vstack([v for v in joint_xyz_robot.values()])

        # tilt_rad_buffer = []   # empty list to store tilt angles in radians

        # for i in range(0, 10, 2):   # pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
        #     mcp = joint_xyz_robot_np[i]
        #     pip = joint_xyz_robot_np[i + 1]
        #     dy = pip[1] - mcp[1]
        #     dx = pip[0] - mcp[0]
        #     tilt_rad = np.arctan2(dy, dx)   # use arctan2 for correct sign/quadrant
        #     tilt_rad_buffer.append(tilt_rad)
        #     print(f"Finger {i//2 + 1} tilt: {np.degrees(tilt_rad):7.3f}°")

        # tilt_rad_buffer = np.array(tilt_rad_buffer)   # convert to numpy array if needed
    
        # ========= 4️⃣ Direct 2D RGB capture and MediaPipe detection =========
    print("📷 Capturing a 2D RGB frame for MediaPipe detection...")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL_NUMBER)
    config.enable_stream(rs.stream.color, *RESOLUTION, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # === Save camera intrinsics ===
    np.save(os.path.join(SAVE_DIR, "color_intrinsics.npy"), {
        "width": color_intrin.width,
        "height": color_intrin.height,
        "fx": color_intrin.fx,
        "fy": color_intrin.fy,
        "ppx": color_intrin.ppx,
        "ppy": color_intrin.ppy,
        "coeffs": color_intrin.coeffs
    })

    # Wait for a valid color frame
    for _ in range(30):  # warm-up
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            break

    color_image = np.asanyarray(color_frame.get_data())
    color_path = os.path.join(SAVE_DIR, "rgb_image.png")
    cv2.imwrite(color_path, color_image)
    print(f"✅ Saved 2D RGB image → {color_path}")

    # === Save raw RGB array ===
    np.save(os.path.join(SAVE_DIR, "rgb_image.npy"), color_image)

    # ========= 5️⃣ Run MediaPipe Hands on the RGB frame =========
    print("🖐 Running MediaPipe detection on captured RGB image...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    hands.close()

    # === Save raw MediaPipe results ===
    np.save(os.path.join(SAVE_DIR, "mediapipe_results_present.npy"), results.multi_hand_landmarks is not None)

    if results.multi_hand_landmarks:
        annotated = color_image.copy()
        all_landmarks_px = []
        all_landmarks_phys = []

        for hand_landmarks in results.multi_hand_landmarks:
            # === Draw all landmarks ===
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # === Highlight MCP and PIP joints ===
            MCP_indices = [2, 5, 9, 13, 17]
            PIP_indices = [3, 6, 10, 14, 18]

            fx, fy = color_intrin.fx, color_intrin.fy
            ppx, ppy = color_intrin.ppx, color_intrin.ppy

            for idx in MCP_indices + PIP_indices:
                lm = hand_landmarks.landmark[idx]
                px, py = int(lm.x * RESOLUTION[0]), int(lm.y * RESOLUTION[1])
                joint_type = "MCP" if idx in MCP_indices else "PIP"

                # === Convert to physical scale (meters) ===
                x_phys = (px - ppx) / fx
                y_phys = (py - ppy) / fy

                print(f"{joint_type} ({idx}): X={px}px ({x_phys*1000:.2f} mm), "
                      f"Y={py}px ({y_phys*1000:.2f} mm)")

                # === Save landmark data ===
                all_landmarks_px.append([joint_type, idx, px, py])
                all_landmarks_phys.append([joint_type, idx, x_phys, y_phys])

                # Draw joint markers
                cv2.circle(annotated, (px, py), 6,
                           (0, 255, 255) if joint_type == "MCP" else (255, 0, 255),
                           -1)
                cv2.putText(annotated, joint_type, (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255) if joint_type == "MCP" else (255, 0, 255), 1)

        # === Save joint data arrays ===
        np.save(os.path.join(SAVE_DIR, "joint_pixels.npy"), np.array(all_landmarks_px, dtype=object))
        np.save(os.path.join(SAVE_DIR, "joint_physical.npy"), np.array(all_landmarks_phys, dtype=object))

    else:
        print("❌ No hands detected in 2D image.")

    pipeline.stop()

    # --- Extract XYZ from point cloud ---
    xyz = pc[:, :3]

    # --- Find closest point for each joint (ray intersection) ---
    closest_points = []
    for idx in range(10):
        x_phys = float(all_landmarks_phys[idx][2])
        y_phys = float(all_landmarks_phys[idx][3])
        ray_dir = np.array([x_phys, y_phys, 1.0])
        ray_dir /= np.linalg.norm(ray_dir)

        cross_prod = np.cross(xyz, ray_dir)
        distances = np.linalg.norm(cross_prod, axis=1)
        min_idx = np.argmin(distances)
        closest_point = xyz[min_idx]
        closest_points.append(closest_point)

    closest_points = np.array(closest_points)  # shape (10, 3)

    np.save((os.path.join(SAVE_DIR, "closest_points.npy")), closest_points)

    # --- Build interleaved MCP/PIP order (both key + value) ---
    joint_order = [f"{j}{i}" for i in range(1, 6) for j in ("MCP", "PIP")]
    value_order = [i for pair in zip(range(5), range(5, 10)) for i in pair]  # [0,5,1,6,2,7,3,8,4,9]

    joint_xyz_dict2 = {
        joint_order[i]: closest_points[value_order[i]]
        for i in range(10)
    }

    # --- Print summary (same as visualize_and_detect) ---
    print("\n🦴 Estimated Joint Positions (Ray Intersection, mm)")
    print("Joint\t\tX(mm)\tY(mm)\tZ(mm)")
    print("-" * 40)
    for name in joint_order:
        x_mm, y_mm, z_mm = joint_xyz_dict2[name] * 1000  # convert meters → mm
        print(f"{name}\t\t{x_mm:7.2f}\t{y_mm:7.2f}\t{z_mm:7.2f}")
    
    print("\n🔄 Transforming hand joint coordinates to robot base frame...")

    # T_base_from_cam = np.array([
    #     [-0.0022, -0.9990,  0.0450, -0.3994],
    #     [-0.9991,  0.0040,  0.0413, -0.1605],
    #     [-0.0415, -0.0448, -0.9981,  0.6085],
    #     [0,        0,        0,       1.0000]
    # ])
    T_base_from_cam = np.array([
        [ 0.01035144, -0.9993305,   0.03509121, -0.39767365],
        [-0.99896007, -0.00877654,  0.04474084, -0.16193797],
        [-0.0444029,  -0.03551785, -0.99838212,  0.60935745],
        [ 0.       ,   0.        ,  0.        ,  1.        ]
        ])
    # T_base_from_cam = np.array([
    # [ 0.01035144, -0.9993305,   0.03509121, -0.39767365],
    # [-0.99896007, -0.00877654,  0.04474084, -0.14593797],
    # [-0.0444029,  -0.03551785, -0.99838212,  0.60935745],
    # [ 0.       ,   0.        ,  0.        ,  1.        ]
    # ])
        


    est_points = np.load("est_points.npy")
    true_points = np.load("true_points.npy")
    joint_xyz_robot = {}
    for name, p_cam in joint_xyz_dict2.items():
        p_h = np.append(p_cam, 1.0)  # homogeneous
        print(p_h)
        p_est = [100 * p_h[0], 100 * p_h[1], 50 - 100 * p_h[2]]
        dists = np.linalg.norm(est_points - p_est, axis = 1)
        min_idx = np.nanargmin(dists)
        expected_real_poisition = true_points[min_idx]
        p_h2 = [expected_real_poisition[0] * 0.01, expected_real_poisition[1] * 0.01, ( 50 - expected_real_poisition[2]) * 0.01, 1]
        print(p_h2)

        p_robot = T_base_from_cam @ p_h
        p_robot2 = T_base_from_cam @ p_h2
        print("p_robot before optics : ", p_robot)
        print("p_robot after optics: ", p_robot2)
        joint_xyz_robot[name] = p_robot[:3]
        print(f"{name:6s} → Robot frame [x y z] = [{p_robot[0]*1000:8.2f}, {p_robot[1]*1000:8.2f}, {p_robot[2]*1000:8.2f}] mm")

    joint_xyz_robot_np = np.vstack([v for v in joint_xyz_robot.values()])
    print(joint_xyz_robot_np)

    # # Convert dict → ordered numpy array following sorted joint names
    # names_sorted = sorted(joint_xyz_robot.keys())   # e.g. ['MCP1','PIP1','MCP2',...
    # joint_xyz_robot_np = np.array([joint_xyz_robot[name] for name in names_sorted])

    # toward finger direction (robot frame)
    joint_xyz_robot_np[:, 0] += 0.004
    #toward left side (frame)
    joint_xyz_robot_np[:, 1] += 0.020

    #intentional perturbation
    joint_xyz_robot_np[:, 1] += 0.007
    joint_xyz_robot_np[:, 2] += 0.007

    print(joint_xyz_robot_np)


    tilt_rad_buffer = []   # empty list to store tilt angles in radians

    for i in range(0, 10, 2):   # pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
        mcp = joint_xyz_robot_np[i]
        pip = joint_xyz_robot_np[i + 1]
        dy = pip[1] - mcp[1]
        dx = pip[0] - mcp[0]
        tilt_rad = np.arctan2(dy, dx)   # use arctan2 for correct sign/quadrant
        tilt_rad_buffer.append(tilt_rad)
        print(f"Finger {i//2 + 1} tilt: {np.degrees(tilt_rad):7.3f}°")

    tilt_rad_buffer = np.array(tilt_rad_buffer)   # convert to numpy array if needed


    # exit()

    # ========= 5️⃣ Initialize robot and rotate 90° =========
    print("\n🤖 Initializing UR3 connection...")
    test_class = raus_systsem_test()
    test_class.control_exp()  # start robot viewer/control
