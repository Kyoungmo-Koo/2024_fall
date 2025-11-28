
"""simulation program for the raus system integration testing 
"""

# from vortex_get_vol_data import OCTEngine 

import numpy as np
import cv2 
import igmr_robotics_toolkit
import open3d
import os
from random import seed, uniform
from threading import Thread
from random import seed, uniform
import scipy 
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

from matplotlib.patches import Circle
from skimage.transform import resize
import torch

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

seed(1337)
np.random.seed(0)

# UR3 2017332424 RUSS
delta_theta = [ 4.61804064568337139e-05, -0.0596917792801850075, 0.187982530479293558, -0.128421417771249463, 1.86920644791972534e-05, -3.70876954857309694e-05]
delta_a = [ 3.74645469127644245e-05, 0.000432751255883601083, 0.0015154637341757704, 3.54739391322167398e-05, 5.12409078229780638e-05, 0]
delta_d = [ 0.000223224169787122895, -6.29131988413453058, 8.56941731705468257, -2.27804644471935935, -7.31385696645381334e-05, 0.000410720409206782877]
delta_alpha = [ 0.000118200678426161332, 0.0023051124802664618, 0.0120016019438704737, 0.000466335407550255709, -0.000233982646400399119, 0]

from PyUniversalRobot import kinematics
table = kinematics.UR3.table

# from igmr_robotics_toolkit.robot.loader import load_robot
# help(load_robot)
# kinematics = load_robot('UR3').kinematics
# print(dir(kinematics))
# # table = kinematics.UR3.table
# table = kinematics.table

# update table for calibration offsets
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

def robust_ik_path(q_start: np.ndarray, ee_path: List[np.ndarray], kin) -> List[np.ndarray]:
    # check that initial joint config matches starting pose
    if not np.allclose(ee_path[0], kin.forward(q_start), atol=1e-5):
        raise RuntimeError(f'initial joint configuration does not match starting pose: {q_start}')

    q_path = [q_start]
    perturbations = [] 
    # for (i, pose) in enumerate(ee_path[1:]):
    #     a = 0
    #     while True:
    #     # for a in [0] + [0.01] * 100 + [0.1] * 100 + [1] * 100:
    #         q_ref = q_path[-1] + [uniform(-a, a) for _ in range(len(q_path[-1]))]
    #         try:
    #             q_path.append(kin.inverse_nearest(pose, q_ref))
    #         except kinematics.InverseException as exc:
    #             e = exc
    #             a += 0.001
    #         else:
    #             e = None
    #             break
    #     perturbations.append(a)
    #     if e:
    #         raise e
        
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

import torch
import torch.nn as nn

class CircleRegressorCNN_K10(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=10, padding=5), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=10, padding=5), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=10, padding=5), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=10, padding=5), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=10, padding=5), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
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

        """selection of the mode"""
        mode_scan_use = "data_collection_test"

        # generate the scanning patterns
        if mode_scan_use == "linear": 
            self._scan_mode                 = "linear"
            self._global_robot_config_init  = []
            self._linear_scan_num_of_line   = 20 
            self._linear_scan_len_line_half = 0.015
            self.linear_scan_mode_load_traj()

        # arc-scanning pattern 
        if mode_scan_use == "arc":
            self._scan_mode                 = "arc"
            self._num_of_arc_scan_pts       = 20 
            self._ang_scan_half_range       = 15    # [ -15, +15 ] degrees
            self.arc_scan_mode_load_traj()

        if mode_scan_use == "data_collection_test":
            self._is_collect_cfm = False
            Thread(target = self.main_data_platform, daemon=True).start()

        # run the windows
        self._window.run()
    
    def arc_scan_mode_load_traj(self, radius = 0.0): 
        """arc-scanning pattern"""

        var_data                        = self._scan_mode  
        self._global_robot_config_init  = []
        num_of_arc_scan_pts             = self._num_of_arc_scan_pts # 15 
        ang_scan_half_range             = self._ang_scan_half_range # 15

        # define the state parameter (this is im)
        state_tmp = self._ptp_exp.state
        if not state_tmp:
            print('no robot state')
            exit() 
        else: 
            print("the q_pose = ", state_tmp.actual_q)
        self._qs_pivot_global = state_tmp.actual_q
        # self._qs_pivot_global = [-0.0370865,  -1.30914073,  1.6146681, -1.85598725, -1.54905718, 0]
        # print(self._qs_pivot_global)

        q_local_use = self._qs_pivot_global
            
        # compute arc rotational offsets
        # TODO: check the slerp function 
        slerp    = Slerp([0, 1], Rotation.from_euler('xyz', [[-ang_scan_half_range, 0, 0], [+ang_scan_half_range, 0, 0]], degrees = True))
        r        = np.linspace(0, 1, num_of_arc_scan_pts)
        steps    = slerp(r)
        
        # frames 
        ee_xform = generic_ik.forward( q_local_use )
        self._tcp_xform3 = np.eye(4)
        self._tcp_xform3[2, 3] = radius
        print("tcp_xform is: ", self._tcp_xform)
        print("tcp_xform3 is: ", self._tcp_xform3)
        # ee_path  = [ee_xform @ self._tcp_xform @ _T(s) @ hinv( self._tcp_xform2 ) for s in steps]
        ee_path = [ee_xform @ self._tcp_xform3 @ _T(s) @ hinv(self._tcp_xform3) for s in steps]

        # self._tcp_xform4 = np.eye(4)
        # self._tcp_xform4[2, 3] = 0.128
        # print(ee_path[0])
        # ee_path_new2 = [ee_path[0] @ self._tcp_xform4 @ _T(steps[0]) @ hinv(self._tcp_xform4)]
        # print(ee_path_new2)
        # # print("_T(0):", _T(steps[0]))

        # inverse kinematics
        ref                     = q_local_use
        val_check               = generic_ik.forward( ref )  
        ee_path_new             = [val_check] + ee_path
        q_path                  = robust_ik_path(ref, ee_path_new, generic_ik )
        q_path                  = q_path[1:]
        self._scan_q_path       = None
        self._scan_q_path_index = None
        self._scan_tform_widget.load(ee_path)

        # summarize the scanning path 
        self._scan_q_path       = q_path

        # initialize the scanning index
        self._scan_q_path_index = 0

        # define the initialize robot configuration
        self._global_robot_config_init = self._scan_q_path[0]

        # save the robot information
        np.save(self._path_data_unique_folder + var_data + '_ee-path.npy', ee_path)
        np.save(self._path_data_unique_folder + var_data + '_q-path.npy', q_path)
        print("finished arc scan ee information")

        return q_path, ee_path



    def linear_scan_mode_load_traj(self):

        self._global_robot_config_init  = []
        var_data                        = self._scan_mode  
        num_of_line                     = self._linear_scan_num_of_line
        len_line_half                   = self._linear_scan_len_line_half

        # tcp-tform 
        tcp_xform                       = self._tcp_xform 

        # state-tmp
        # TODO: we use the the current robot configuration to initialize the linear scanning patterns.
        state_tmp = self._ptp_exp.state
        if not state_tmp:
            print('no robot state')
            exit() 
        else: 
            print("the q_pose = ", state_tmp.actual_q)
        # q_tmp_use = self._qs_pivot_global
        self._qs_pivot_global = state_tmp.actual_q
        # self._qs_pivot_global = [-0.0370865,  -1.30914073,  1.6146681, -1.85598725, -1.54905718, 0]
        # print(self._qs_pivot_global)

        # ee-tform
        ee_xform = generic_ik.forward( self._qs_pivot_global )

        # compute line translational offsets
        offsets         = np.linspace([0, -len_line_half, 0], [0, +len_line_half, 0], num_of_line)
        steps           = []
        for offset in offsets:
            xform = np.eye(4)
            xform[:3, 3] = offset
            steps.append(xform)

        # transform in to EE frame
        # ee_path = [ee_xform @ tcp_xform @ s @ hinv(tcp_xform) for s in steps]
        ee_path = [ee_xform @ tcp_xform @ s @ hinv(tcp_xform) for s in steps][::-1]
        self._scan_tform_widget.load(ee_path)

        # inverse kinematics
        # ref     = self._robot.kinematics.inverse_nearest( ee_path[0], self._qs_home )
        ref     = generic_ik.inverse_nearest( ee_path[0], self._qs_home )
        q_path  = robust_ik_path(ref, ee_path, generic_ik)
        try:
            check_joint_path(q_path, np.pi / 180 * 20)
        except Exception as e:
            print('failed to create arc joint path', e)
            self._scan_q_path = None
            self._scan_q_path_index = None
            return

        # reset the scanning index 
        self._scan_q_path_index = 0

        # initialize the scanning path
        self._scan_q_path = q_path
        print('loaded new linear path')

        # define the initialize robot configuration
        self._global_robot_config_init = self._scan_q_path[0]

        # save the robot information
        np.save(self._path_data_unique_folder + var_data + '-ee-path.npy', ee_path)
        np.save(self._path_data_unique_folder + var_data + '-q-path.npy', q_path)
        print("finish saving the robot and ee information")

        return q_path, ee_path

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
                        np.save(f"{self._path_data_unique_folder}initial_check_bmode3.npy", img_save_bmode_uint16[:, :, 0])
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

    def rotate_half_pi(self):
        check_q = self.current_q + [0, 0, 0, 0, 0, - np.pi / 2]
        self._ptp_exp.move_joint(check_q, qd_limits=self._qd_limit_input * 3)
        self.current_q = check_q

    def rotate_half_pi_back(self):
        check_q = self.current_q + [0, 0, 0, 0, 0, np.pi / 2]
        self._ptp_exp.move_joint(check_q, qd_limits=self._qd_limit_input * 3)
        self.current_q = check_q
    
    def preprocess_image(self, img_np, H, W):
        if img_np.ndim == 2:
            print("hello")
            h, w = img_np.shape
            img_4ch = np.zeros((h, w, 4), dtype=np.float32)
            img_4ch[..., 0] = img_np       # R
            img_4ch[..., 1] = img_np       # G
            img_4ch[..., 2] = img_np       # B
            img_4ch[..., 3] = 255.0     # A (alpha)
        else:
            img_4ch = img_np.astype(np.float32)
            # if img_np.ndim == 3:
            #     img_np = img_np.mean(axis=-1)  # grayscale
        print(img_4ch.shape)

        img_resized = resize(img_4ch, (H, W), preserve_range=True).astype(np.float32)
        print(np.mean(img_resized))

        print(img_resized.shape)

        img_resized = img_resized[..., :3].mean(axis=-1).astype(np.float32)

        print(img_resized.shape)

        mx = float(img_resized.max())
        if mx > 0:
            img_resized = img_resized / (255.0 if mx > 1.0 else mx)
        print(img_resized.shape)
        img_resized = np.expand_dims(img_resized, axis=0)  # (1,H,W)
        print(img_resized.shape)
        img_resized = np.expand_dims(img_resized, axis=0)  # (1,H,W)
        return torch.from_numpy(img_resized.astype(np.float32))
    
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

    def analyze_now(self):

        self.initial_check = 1
        self.quant_count = self.quant_count + 1
        self.save_us_img_tmp()

        img = np.load(f"{self._path_data_unique_folder}initial_check_bmode.npy")
        print("image collected, analysis start")
        print(img.shape)

        H = self.H
        W = self.W
        MAX_R = self.MAX_R

        model = self.model

        if img.ndim == 2:
            h, w = img.shape
            img_4ch = np.zeros((h, w, 4), dtype=np.float32)
            img_4ch[..., 0] = img       # R
            img_4ch[..., 1] = img       # G
            img_4ch[..., 2] = img       # B
            img_4ch[..., 3] = 255.0     # A (alpha)
        else:
            img_4ch = img.astype(np.float32)

        x = self.preprocess_like_dataset(img_4ch, H, W).unsqueeze(0)  # (1,1,H,W)
        with torch.no_grad():
            pred = model(x).squeeze(0)

        # with torch.no_grad():
        #     pred = model(x).squeeze(0).numpy()

        pred_cx = float(pred[0]) * W
        pred_cy = float(pred[1]) * H
        pred_r  = float(pred[2]) * MAX_R
        print(f"Predicted: cx={pred_cx:.2f}, cy={pred_cy:.2f}, r={pred_r:.2f}")

        self.initial_check = 1
        self.save_us_img_tmp()

        img = np.load(f"{self._path_data_unique_folder}initial_check_bmode3.npy")
        print("image collected, analysis start")
        print(img.shape)

        H = self.H
        W = self.W
        MAX_R = self.MAX_R

        model = self.model

        if img.ndim == 2:
            h, w = img.shape
            img_4ch = np.zeros((h, w, 4), dtype=np.float32)
            img_4ch[..., 0] = img       # R
            img_4ch[..., 1] = img       # G
            img_4ch[..., 2] = img       # B
            img_4ch[..., 3] = 255.0     # A (alpha)
        else:
            img_4ch = img.astype(np.float32)

        x = self.preprocess_like_dataset(img_4ch, H, W).unsqueeze(0)  # (1,1,H,W)
        with torch.no_grad():
            pred = model(x).squeeze(0)

        # with torch.no_grad():
        #     pred = model(x).squeeze(0).numpy()

        pred_cx = float(pred[0]) * W
        pred_cy = float(pred[1]) * H
        pred_r  = float(pred[2]) * MAX_R
        print(f"Predicted: cx={pred_cx:.2f}, cy={pred_cy:.2f}, r={pred_r:.2f}")

        # Pixel offset from image center
        dx_px = pred_cx - (W / 2)
        dy_px = pred_cy - (H * 3/5)

        # Pixel → real-world scaling (example: 0.025 m for half width/height)
        scale_x = 0.025 / (W)
        scale_y = 0.030 / (H)

        # Build translation along robot local Y (image x) and local Z (image y)
        T_ee = self._robot.kinematics.forward(self.current_q)
        T_translate = np.eye(4)
        T_translate[:3, 3] = (
            T_ee[:3, 1] * (dx_px * scale_x) -
            T_ee[:3, 2] * (dy_px * scale_y)
        )

        T_ee_moved = T_ee @ T_translate

        np.save(f"ee_{self.quant_count}.npy", T_ee_moved)

        return T_ee_moved
    
    def move_in_cartesian(self, x_mm, y_mm):

        self.quant_count = self.quant_count + 1
        
        T_ee = self._robot.kinematics.forward(self.initial_state2.actual_q)
        T_translate = np.eye(4)
        T_translate[:3, 3] = (
            T_ee[:3, 1] * (0.001 * x_mm) -
            T_ee[:3, 2] * (0.001 * y_mm)
        )

        T_ee_moved = T_ee @ T_translate
        np.save(f"ee_{self.quant_count}.npy", T_ee_moved)

        return T_ee_moved

    def main_data_platform(self): 
        self.quant_count = 0

        ckpt_path = "circle_regressor_k10_ninesets_valholdout_mps.pt"

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
        time.sleep(2.0)

        self._is_collect_cfm = False

        # Folder setup
        name_of_folder = "20250823_finger_bscan"
        path_local_main = "./data_raus/"
        path_current_folder = path_local_main + name_of_folder + r"\\"
        path_local_linear_scan = path_current_folder + "combine_linear_scan\\"
        path_local_arc_scan = path_current_folder + "combine_arc_scan\\"
        path_local_actual_pos = path_current_folder + "actual_pos\\"

        for p in [path_current_folder, path_local_linear_scan, path_local_arc_scan, path_local_actual_pos]:
            if not os.path.isdir(p):
                os.mkdir(p)

        self._path_data_unique_folder = path_current_folder

        self.initial_state = self._ptp_exp.state
        self.current_q = self.initial_state.actual_q
        # self.rotate_half_pi()

        self.initial_state2 = self._ptp_exp.state

        ## 1 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(-2, 2), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        exit()
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 2 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(0, 6), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 3 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(-6, 6), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 4 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(6, 0), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 5 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(0, 0), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 6 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(-6, 0), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 7 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(6, -6), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 8 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(0, -6), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)

        ## 9 ##
        qs_home2 = self._robot.kinematics.inverse_nearest(self.move_in_cartesian(-6, -6), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(1)

        qs_home2 = self._robot.kinematics.inverse_nearest(self.analyze_now(), self.current_q)
        self._ptp_exp.move_joint(qs_home2, qd_limits=self._qd_limit_input)
        self.current_q = qs_home2

        time.sleep(5)


        self.rotate_half_pi_back()
        exit()

        
        # Linear scan trajectory
        self._scan_mode = "linear"
        self._global_robot_config_init = []
        self._linear_scan_num_of_line = 200
        self._linear_scan_len_line_half = 0.0100
        q_path_linear, _ = self.linear_scan_mode_load_traj()

        # Arc scan 1 (default radius)
        self._scan_mode = "arc"
        self._ang_scan_half_range = 50
        # self._num_of_arc_scan_pts = int(2 * self._ang_scan_half_range * np.pi / 180 * 0.011 * (self._linear_scan_num_of_line / (2 * self._linear_scan_len_line_half)))
        self._num_of_arc_scan_pts = 100
        q_path_arc_default, _ = self.arc_scan_mode_load_traj(radius=0.148)

        self._tcp_xform_start = np.eye(4)
        self._tcp_xform_start[2, 3] = 0.128
        q_start = q_path_arc_default[0]
        ee_start = generic_ik.forward(q_start)

        rot_local = Rotation.from_euler('x', 50, degrees=True).as_matrix()
        delta_rot = np.eye(4)
        delta_rot[:3, :3] = rot_local
        ee_rotated = ee_start @ self._tcp_xform_start @ delta_rot @ hinv(self._tcp_xform_start)
        # q_rotated = self._robot.kinematics.inverse_nearest(ee_rotated, q_start)
        # q_rotated = generic_ik.inverse_nearest(ee_rotated, q_start)
        #self._ptp_exp.move_joint(q_rotated, qd_limits=self._qd_limit_input)

        # Define rotation from +50° to 0° around local x-axis
        rotations = Rotation.from_euler(
            'xyz', 
            [[50, 0, 0], [0, 0, 0]],  # From 50° to 0°
            degrees=True
        )
        slerp = Slerp([0, 1], rotations)
        r_vals = np.linspace(0, 1, int(self._num_of_arc_scan_pts / 2))
        interpolated_rots = slerp(r_vals)

        count = 0
        # Build interpolated ee poses
        interpolated_poses = []
        for rot in interpolated_rots:
            delta_rot = np.eye(4)
            delta_rot[:3, :3] = rot.as_matrix()
            pose = ee_start @ self._tcp_xform_start @ delta_rot @ np.linalg.inv(self._tcp_xform_start)
            interpolated_poses.append(pose)
        np.save(self._path_data_unique_folder + self._scan_mode + '_ee-path_start.npy', interpolated_poses)

        # q_seed = q_start

        # for pose in reversed(interpolated_poses):
        #     q_interp = generic_ik.inverse_nearest(pose, q_seed)
        #     self._ptp_exp.move_joint(q_interp, qd_limits=self._qd_limit_input)
        #     time.sleep(0.1)

        #     # update seed with the last computed solution
        #     q_seed = q_interp

        #     self.idx_in_loop = count
        #     count += 1
        #     print("count: ", count)

        #     if self._is_ultrasound_mode:
        #         self.save_us_img_tmp()

        # print("done")

        q_seed = q_start
        q_solutions = []

        for pose in reversed(interpolated_poses):
            q_interp = generic_ik.inverse_nearest(pose, q_seed)
            # update seed with the last computed solution
            q_seed = q_interp
            q_solutions.append(q_interp)
        
        for q_solution in reversed(q_solutions):
            self._ptp_exp.move_joint(q_solution, qd_limits = self._qd_limit_input)
            time.sleep(0.1)
            self.idx_in_loop = count
            count = count + 1
            print("count: ", count)
            if self._is_ultrasound_mode:
                self.save_us_img_tmp()

        for q in q_path_arc_default[:int(self._num_of_arc_scan_pts)]:
            self._ptp_exp.move_joint(q, qd_limits=self._qd_limit_input)
            time.sleep(0.1)
            self.idx_in_loop = count
            count = count + 1
            print("count: ", count)
            if self._is_ultrasound_mode:
                self.save_us_img_tmp()
        
        self._tcp_xform_end = np.eye(4)
        self._tcp_xform_end[2, 3] = 0.128
        q_end = q_path_arc_default[-1]
        ee_end = generic_ik.forward(q_end)

        rot_local = Rotation.from_euler('x', -50, degrees=True).as_matrix()
        delta_rot = np.eye(4)
        delta_rot[:3, :3] = rot_local
        ee_rotated = ee_end @ self._tcp_xform_end @ delta_rot @ hinv(self._tcp_xform_end)
        # q_rotated = self._robot.kinematics.inverse_nearest(ee_rotated, q_end)
        # q_rotated = generic_ik.inverse_nearest(ee_rotated, q_end)

                # Define rotation from +50° to 0° around local x-axis
        rotations = Rotation.from_euler(
            'xyz', 
            [[0, 0, 0], [-50, 0, 0]],  # From 50° to 0°
            degrees=True
        )
        slerp = Slerp([0, 1], rotations)
        r_vals = np.linspace(0, 1, int(self._num_of_arc_scan_pts / 2))
        interpolated_rots = slerp(r_vals)

        # Build interpolated ee poses
        interpolated_poses = []
        for rot in interpolated_rots:
            delta_rot = np.eye(4)
            delta_rot[:3, :3] = rot.as_matrix()
            pose = ee_end @ self._tcp_xform_end @ delta_rot @ np.linalg.inv(self._tcp_xform_end)
            interpolated_poses.append(pose)
        np.save(self._path_data_unique_folder + self._scan_mode + '_ee-path_end.npy', interpolated_poses)

        # # Move through interpolated poses
        # for pose in interpolated_poses:
        #     # q_interp = self._robot.kinematics.inverse_nearest(pose, q_end)
        #     q_interp = generic_ik.inverse_nearest(pose, q_end)
        #     self._ptp_exp.move_joint(q_interp, qd_limits=self._qd_limit_input)
        #     time.sleep(0.1)
        #     self.idx_in_loop = count
        #     count = count + 1
        #     print("count: ", count)
        #     if self._is_ultrasound_mode:
        #         self.save_us_img_tmp()

        # Move through interpolated poses (CHASE: seed-chaining IK)
        q_seed = q_end  # start chasing from q_end

        for pose in interpolated_poses:
            # solve IK near the last known solution
            q_interp = generic_ik.inverse_nearest(pose, q_seed)

            # execute joint move
            self._ptp_exp.move_joint(q_interp, qd_limits=self._qd_limit_input)
            time.sleep(0.1)
            q_seed = q_interp
            self.idx_in_loop = count
            count += 1
            print("count:", count)

            if self._is_ultrasound_mode:
                self.save_us_img_tmp()

        print("Finished all scans including dual arc trajectories")



if __name__ == "__main__": 

    test_class = raus_systsem_test()

    # purely experiment
    # test_class.mode_move_to_single_robot_config()
    test_class.control_exp()  

    # purely simulation 
    # test_class.control_sim() 
    