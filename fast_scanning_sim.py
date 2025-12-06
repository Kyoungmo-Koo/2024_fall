
"""fast scanning modules 
    1. get a single hand image. 
    2. python-to-matlab: switch the option between the two ( label the hand object with the joint positions).
    3. perform the robot simulation and realistic experiments (this is im)
"""

import numpy as np
from pathlib import Path
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
import yaml
from scipy.spatial.transform import Rotation, Slerp
from panda3d.core import NodePath
from igmr_robotics_toolkit.viewer.widget import MeshWidget

# igmr-robotics-toolkits
import igmr_robotics_toolkit.util.default_logging
from igmr_robotics_toolkit.viewer.core import create_simple_viewer
from igmr_robotics_toolkit.collision import Collider
from igmr_robotics_toolkit.viewer.widget import PathWidget, PointCloudWidget, TransformWidget, RobotWidget, TransformListWidget, LineWidget
from igmr_robotics_toolkit.control.simulator import Simulator
from igmr_robotics_toolkit.motion.trajectory import TrajectoryGenerator
from igmr_robotics_toolkit.viewer.motion import show_trajectory
from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.motion.path import ik_path, check_joint_path
from igmr_robotics_toolkit.math import hinv
from igmr_robotics_toolkit.viewer.core import _T
from igmr_robotics_toolkit.viewer.core import create_simple_viewer
from igmr_robotics_toolkit.viewer.widget import ControlledRobotWidget, PathWidget, TransformWidget
from igmr_robotics_toolkit.control.simple import PointToPoint
from igmr_robotics_toolkit.math import average_pose
from igmr_robotics_toolkit.control.action import ActionProgram
from igmr_robotics_toolkit.control.action.position_trajectory import BufferedJointTrajectoryAction, BufferedCartesianTrajectoryAction
from igmr_robotics_toolkit.util import parse_xform

# third-party
import socket

# klampt math module 
from klampt.math import so3 

# ultrasound module
# us system 
import sys
sys.path.append("./utility/")
# from ultrasound_ge_class import Ultrasound, FORMAT_CFM, FORMAT_NAMES

import shutil
import time 

seed(1337)
np.random.seed(0)

class raus_fast_scanning(): 

    def __init__(self):
        
        # ur3 related configurations
        self.path_ur3_config     = "./database/system/UR3/robot.yaml" 
        self.path_ur5_config     = "./database/system/UR5e/robot.yaml" 
        self._host_ip_ur3        = '169.254.88.100'
        
        # hand-finger object 
        self._path_finger_img   = [] 
        self._path_finger_tex   = []
        self._path_finger_vex   = [] 
        
        self._path_xyz_pc       = "./data_fast_scanning/pts_xyz_in_matlab.mat"
        self._path_rgb_pc       = "./data_fast_scanning/pts_rgb_in_matlab.mat"
        self._path_xyz_finger   = "./data_fast_scanning/pts_finger_joints_in_matlab.mat"

    def integration_testing_exp(self): 
        """integration testing: show all the information within the same scene"""

        # TODO: zc
        # perform the generic simulation 
        mode_use                    = "sim"
        # mode_use                    = "exp"

        # ultrasound mode
        self._is_ultrasound_mode    = False
        self._is_collect_cfm        = False
        if self._is_ultrasound_mode: 
            self._display_size              = (500, 600)
            self._host_ge                   = '169.254.107.11'
            # self._mode_ge_img               = 'CFM'
            self._mode_ge_img               = '2D'
            self._obj_ultrasound            = Ultrasound()
            self._obj_ultrasound.connect( host = self._host_ge )
            self._obj_ultrasound.start( self._mode_ge_img )
        else: 
            print("the ultrasound is not started")

        # initialized robot settings 
        self._robot                 = load_robot( self.path_ur3_config ) 
        self._qs_home               = [-0.02121813, -1.2175811, 1.36699295, -1.73946172, -1.55588037, -1.59545452]

        # exp connection 
        if mode_use == "exp":
            self._ctrl_exp          = self._robot.Controller( host  = self._host_ip_ur3,
                                                              model = self._robot, 
                                                              robot_info=('2017332424', '3.4.1.59')  )
        elif mode_use == "sim": 
            self._ctrl_exp          = Simulator( self._robot )
            self._ctrl_exp._q       = self._qs_home

        # create the sample viewers 
        (self._window, self._root)  = create_simple_viewer()

        # connect the ptp agent
        self._ptp_exp               = PointToPoint(self._robot, self._ctrl_exp )
        self._ptp_exp.connect()

        self._ptp_exp._controller.joint_delta_max = 2.0

        # step-2: setup the viewer and window
        self._crwa                  = ControlledRobotWidget(model       = self._robot, 
                                                            controller  = self._ptp_exp, 
                                                            parent      = self._root, 
                                                            frames      = [0, self._robot.dof], 
                                                            show_mode   = 'actual_q'    ) 

        # adjust the mode and the speed
        if mode_use == "sim": 
            self._qd_limit_input     = np.pi / 1
        elif mode_use == "exp": 
            self._qd_limit_input     = np.pi / 60
        self.idx_global_oct_data = 0 

        """ee-to-us"""
        is_cartesian                 = False 
        self._robot.visual.parent    = self._root
        path_calib_data_from_matlab  = "./database/tform_ee_to_sensor_opt_result.mat"
        self._tform_ee_to_us_opt_res = scipy.io.loadmat( path_calib_data_from_matlab )["tform_ee_to_sensor_opt_result"]
        self._tcp_xform              = np.asarray( self._tform_ee_to_us_opt_res )    
        self._ee_to_tcp_tform        = self._tcp_xform
        self._sensor_widget          = TransformListWidget( parent = self._root, scale = 0.25 )
        self._sensor_widget.load( [self._robot.kinematics.forward( self._qs_home ) @ self._tform_ee_to_us_opt_res] )
        
        """sensor_to_base"""
        self._tform_sensor_to_base  = np.eye(4)
        path_sensor_to_base_matlab  = "./data_rgbd/tform_sensor_to_base_post.mat"
        self._tform_sensor_to_base_in_matlab  = scipy.io.loadmat( path_sensor_to_base_matlab )["tform_sensor_to_base_post"]
        vec_x_from_matlab           = self._tform_sensor_to_base_in_matlab[0:3,0]
        vec_y_from_matlab           = self._tform_sensor_to_base_in_matlab[0:3,1]
        vec_z_from_matlab           = self._tform_sensor_to_base_in_matlab[0:3,2]
        pts_org_from_matlab         = self._tform_sensor_to_base_in_matlab[0:3,3]
        self._tform_sensor_to_base[0:3,0] =  vec_x_from_matlab
        self._tform_sensor_to_base[0:3,1] =  vec_y_from_matlab
        self._tform_sensor_to_base[0:3,2] =  vec_z_from_matlab
        self._tform_sensor_to_base[0:3,3] =  np.asanyarray([ pts_org_from_matlab[0], pts_org_from_matlab[1], pts_org_from_matlab[2]])
        self._sensor_to_base_widget       = TransformListWidget( parent = self._root, scale = 1.0 )
        self._sensor_to_base_widget.load( [self._tform_sensor_to_base] )

        """test with the rgbd-phantom model == 9-sphere targets and the phantoms + detection of the four fiducial markers"""
        pts_phantom_raw                     = scipy.io.loadmat( "./database/pts_phantom.mat" )["pts_phantom"]
        pts_fid_group_1                     = scipy.io.loadmat( "./database/traj_fid_group_1.mat" )["traj_fid_group_1"]
        pts_fid_group_2                     = scipy.io.loadmat( "./database/traj_fid_group_2.mat" )["traj_fid_group_2"]
        pts_fid_group_3                     = scipy.io.loadmat( "./database/traj_fid_group_3.mat" )["traj_fid_group_3"]
        pts_fid_group_4                     = scipy.io.loadmat( "./database/traj_fid_group_4.mat" )["traj_fid_group_4"]
        pts_center_from_fid                 = scipy.io.loadmat( "./database/pts_center_from_fid.mat" )["pts_center_from_fid"]
        traj_fid_sphere                     = scipy.io.loadmat( "./database/traj_fid_sphere.mat" )["traj_fid_sphere"]

        """show the stl model with the phantom mode + model"""
        w_tip_to_ee = TransformWidget( parent = self._crwa.model.ee_link.visual.root )
        tool_mesh_widget = MeshWidget(
                                    path    = Path( r"./MATLAB/robot_ultrasound_system/python/database" ) / "phantom_system_calibration_for_test.obj", 
                                    parent  = self._crwa.model.base_link.visual.root,
                                    name    = 'frame',
                                    scale   = 0.001
                                )
        tool_mesh_widget.pose = parse_xform('rot(90d, 0, 0)trans(-0.4, 0, 0.1)')

        """transformation of the phantom to the world model"""
        tform_form_phantom_to_world         = np.eye(4)
        tform_form_phantom_to_world[0:3,3]  = np.asarray([-0.4, -0.10, 0.0])

        """vis: phantom models"""
        pts_phantom_xyz_hc                  = np.hstack([pts_phantom_raw, np.ones([pts_phantom_raw.shape[0], 1])])
        pts_phantom_xyz_hc_in_world         = np.transpose( tform_form_phantom_to_world @ np.transpose( pts_phantom_xyz_hc ) )
        pts_phantom_in_world                = pts_phantom_xyz_hc_in_world[:,0:3]
        pts_rgbt_raster                     = np.zeros( pts_phantom_in_world.shape) 
        pts_rgbt_raster[:,0]                = 255
        pts_rgbt_raster                     = np.hstack([pts_rgbt_raster, np.ones([pts_rgbt_raster.shape[0], 1]) * 64 ])
        pc_phantom_widget                   = PointCloudWidget(parent = self._root, point_size = 25 )
        pc_phantom_widget.load( np.ascontiguousarray( pts_phantom_in_world ), np.ascontiguousarray( pts_rgbt_raster ) )

        """show the fid point cloud"""
        pts_fid_xyz_hc                      = np.hstack([traj_fid_sphere, np.ones([traj_fid_sphere.shape[0], 1])])
        pts_fid_xyz_hc_in_world             = np.transpose( tform_form_phantom_to_world @ np.transpose( pts_fid_xyz_hc ) )
        pts_fid_in_world                    = pts_fid_xyz_hc_in_world[:,0:3]
        pts_rgbt_raster                     = np.zeros( pts_fid_in_world.shape) 
        pts_rgbt_raster[:,2]                = 255
        pts_rgbt_raster                     = np.hstack([pts_rgbt_raster, np.ones([pts_rgbt_raster.shape[0], 1]) * 255 ])
        pc_fid_widget                       = PointCloudWidget(parent = self._root, point_size = 35 )
        pc_fid_widget.load( np.ascontiguousarray( pts_fid_in_world ), np.ascontiguousarray( pts_rgbt_raster ) )

        """plan for the trajecotry -- for the 9-different joints"""
        self._pts_finger_joint              = pts_fid_in_world
        self.traj_linear_and_arc_from_pivot_point()

        """run the program"""
        # generate the robot simulation widget 
        self._pc_us_widget                  = PointCloudWidget(parent = self._root)
        Thread(target = self.main_loop_for_calib_auto, daemon = True).start()
    
        # run the windows
        view_pose = np.asarray( [ [ 0.66752115,  0.51424116,  0.53849007, -3.39468878  ],
                                  [-0.71273616,  0.23202099,  0.66194669, -1.494105045 ],
                                  [ 0.21545923, -0.82566476,  0.52139719,  5.01182138  ],
                                  [ 0.        ,  0.        ,  0.        ,  1.          ] ] )
        
        self._window.camera_pose = view_pose  
        self._window.run()
    
    def group_by_xy_jump(positions, num_groups=20):
        N = len(positions)
        dist = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)

        # find largest gaps
        split_indices = np.argsort(dist)[-(num_groups - 1):] + 1
        split_indices = np.sort(split_indices)

        groups = np.zeros(N, dtype=int)
        start = 0
        for g, end in enumerate(np.append(split_indices, N)):
            groups[start:end] = g
            start = end

        return groups, split_indices


    def integration_finger_exp(self): 
        """integration testing: show all the information within the same scene"""

        # TODO: zc
        # perform the generic simulation 
        mode_use                    = "sim"
        # mode_use                    = "exp"

        # ultrasound mode
        self._is_ultrasound_mode    = False
        self._is_collect_cfm        = False
        if self._is_ultrasound_mode: 
            self._display_size              = (500, 600)
            self._host_ge                   = '169.254.107.11'
            # self._mode_ge_img               = 'CFM'
            self._mode_ge_img               = '2D'
            self._obj_ultrasound            = Ultrasound()
            self._obj_ultrasound.connect( host = self._host_ge )
            self._obj_ultrasound.start( self._mode_ge_img )
        else: 
            print("the ultrasound is not started")

        # initialized robot settings 
        self._robot                 = load_robot( self.path_ur3_config ) 
        self._qs_home               = [-0.02121813, -1.2175811, 1.36699295, -1.73946172, -1.55588037, -1.59545452]

        pts_finger_xyz_rgb = np.load(r"C:\Users\labadmin\Desktop\Robot_Calibration\robot_ultrasound_system\python\20251129_robot_6\pc_0.npy", allow_pickle=True)
        pts_finger_xyz = pts_finger_xyz_rgb[:, :3]
        pts_finger_rgb = pts_finger_xyz_rgb[:, 3:] * 256
        pts_finger_fid = np.load(r"C:\Users\labadmin\Desktop\Robot_Calibration\robot_ultrasound_system\python\20251129_robot_6\closest_points.npy", allow_pickle=True)
        pts_finger_ee_traj = np.load(r"C:\Users\labadmin\Desktop\Robot_Calibration\robot_ultrasound_system\python\20251129_images_6_ee_interp.npy", allow_pickle = True)

        # # Number of trajectory points
        # N = pts_finger_ee_traj.shape[0]

        # # Storage for q trajectory (UR robots = 6 DOF)
        # pts_finger_q_traj = np.zeros((N, 6))

        # # Home configuration for inverse kinematics
        # qs_home = self._qs_home   # or define explicitly: np.array([...])

        # # Convert all EE poses → joint angles
        # OFFSET_Y = - 0.007

        # for i in range(N):
        #     ee_pose = pts_finger_ee_traj[i]   # shape (4,4)
        #     ee_pose[1, 3] += OFFSET_Y
        #     q = self._robot.kinematics.inverse_nearest(ee_pose, qs_home)
        #     pts_finger_q_traj[i] = q
        #     qs_home = q   # optional: update seed for continuity
        # print(pts_finger_fid)
        # print(pts_finger_xyz.shape)
        # print(pts_finger_rgb.shape)
        # print(pts_finger_fid.shape)
        # print(pts_finger_ee_traj.shape)
        # print(pts_finger_q_traj.shape)

        # # pts_finger_q_traj is already computed by IK
        # pts_finger_q_traj = np.asarray(pts_finger_q_traj, dtype=float)

        # max_step = 0.1  # maximum joint difference allowed
        # safe_q_list = []

        # # start with the first point
        # safe_q_list.append(pts_finger_q_traj[0].copy())

        # for i in range(1, len(pts_finger_q_traj)):
        #     q_prev = pts_finger_q_traj[i-1]
        #     q_next = pts_finger_q_traj[i]

        #     diff = q_next - q_prev
        #     max_diff = np.max(np.abs(diff))

        #     if max_diff <= max_step:
        #         # safe, no interpolation needed
        #         safe_q_list.append(q_next.copy())
        #     else:
        #         # number of interpolation segments needed
        #         num_steps = int(np.ceil(max_diff / max_step))

        #         # interpolate between q_prev → q_next
        #         for s in range(1, num_steps + 1):
        #             alpha = s / num_steps
        #             q_interp = q_prev + alpha * diff
        #             safe_q_list.append(q_interp.copy())

        # # convert to numpy array
        # safe_q_traj = np.vstack(safe_q_list)

        # ============================================================
        # 1) Convert EE → Joint Angles (IK + Y-offset)
        # ============================================================

        N = pts_finger_ee_traj.shape[0]
        pts_finger_q_traj = np.zeros((N, 6))

        qs_home = self._qs_home
        OFFSET_Y = -0.007

        for i in range(N):
            ee_pose = pts_finger_ee_traj[i].copy()
            ee_pose[1, 3] += OFFSET_Y

            q = self._robot.kinematics.inverse_nearest(ee_pose, qs_home)
            pts_finger_q_traj[i] = q
            qs_home = q   # continuity


        print("pts_finger_xyz:", pts_finger_xyz.shape)
        print("pts_finger_rgb:", pts_finger_rgb.shape)
        print("pts_finger_fid:", pts_finger_fid.shape)
        print("pts_finger_ee_traj:", pts_finger_ee_traj.shape)
        print("pts_finger_q_traj:", pts_finger_q_traj.shape)


        # ============================================================
        # 2) Group trajectory using XY jumps of end-effector positions
        # ============================================================

        # Extract XY positions from pts_finger_ee_traj
        positions = pts_finger_ee_traj[:, :3, 3]   # (N,3) → EE translation
        positions_xy = positions[:, :2]            # (N,2)

        # Compute jump distances between samples
        dist_xy = np.linalg.norm(np.diff(positions_xy, axis=0), axis=1)

        # Select 20 largest XY gaps → 20 trajectory groups
        num_groups = 20
        split_indices = np.argsort(dist_xy)[-(num_groups - 1):] + 1
        split_indices = np.sort(split_indices)

        # Build group ranges
        group_ranges = []
        start = 0
        for si in list(split_indices) + [N]:
            end = si - 1
            group_ranges.append((start, end))
            start = si

        self.group_ranges = group_ranges
        print("\n=== ORIGINAL GROUP RANGES (XY-based) ===")
        for g, (s, e) in enumerate(group_ranges):
            print(f"Group {g:02d}: start={s}, end={e}")


        # ============================================================
        # 3) Interpolate only between consecutive groups
        # ============================================================

        max_step = 0.1
        safe_q_list = []
        safe_group_ranges = []

        current_idx = 0

        for g, (s, e) in enumerate(group_ranges):

            # 3-1 Append the group's raw IK results
            group_q = pts_finger_q_traj[s:e+1]
            L = len(group_q)

            safe_q_list.extend(group_q)

            start_after = current_idx
            end_after = current_idx + L - 1
            safe_group_ranges.append((start_after, end_after))
            current_idx = end_after + 1

            # 3-2 Interpolate between this group's end and next group's start
            if g < len(group_ranges) - 1:
                s_next, _ = group_ranges[g+1]
                q_end = pts_finger_q_traj[e]
                q_next = pts_finger_q_traj[s_next]

                diff = q_next - q_end
                max_diff = np.max(np.abs(diff))

                if max_diff > max_step:
                    num_steps = int(np.ceil(max_diff / max_step))
                    for k in range(1, num_steps + 1):
                        alpha = k / num_steps
                        q_interp = q_end + alpha * diff
                        safe_q_list.append(q_interp)
                        current_idx += 1


        # Convert result to array
        safe_q_traj = np.array(safe_q_list, dtype=float)


        # ============================================================
        # 4) Print AFTER-interpolation group ranges
        # ============================================================
        self.safe_group_ranges = safe_group_ranges
        print("\n=== GROUP RANGES AFTER INTERPOLATION ===")
        for g, (s, e) in enumerate(safe_group_ranges):
            print(f"Group {g:02d}: start={s}, end={e}")

        print("\nFinal safe_q_traj shape:", safe_q_traj.shape)

        self.pts_finger_q_traj = safe_q_traj

        self._qs_home = pts_finger_q_traj[0]

        print("ee[0]: ", pts_finger_ee_traj[0])
        print("ee[52]:", pts_finger_ee_traj[52])
        print("finger_fid[0]", pts_finger_fid[0])

        # exp connection 
        if mode_use == "exp":
            self._ctrl_exp          = self._robot.Controller( host  = self._host_ip_ur3,
                                                              model = self._robot, 
                                                              robot_info=('2017332424', '3.4.1.59')  )
        elif mode_use == "sim": 
            self._ctrl_exp          = Simulator( self._robot )
            self._ctrl_exp._q       = self._qs_home

        # connect the ptp agent
        self._ptp_exp               = PointToPoint(self._robot, self._ctrl_exp )
        self._ptp_exp.connect()

        # step-2: setup the viewer and window
        # create the sample viewers 
        (self._window, self._root)  = create_simple_viewer()
        self._crwa                  = ControlledRobotWidget(model       = self._robot, 
                                                            controller  = self._ctrl_exp, 
                                                            parent      = self._root, 
                                                            frames      = [0, self._robot.dof], 
                                                            show_mode   = 'actual_q'    ) 

        # adjust the mode and the speed
        if mode_use == "sim": 
            self._qd_limit_input     = np.pi / ( 0.040 + 0.0 )
        elif mode_use == "exp": 
            self._qd_limit_input     = np.pi / 60
        self.idx_global_oct_data = 0 

        # print("self._ptp_exp.state = ", self._ptp_exp._controller.state.actual_q)
        # self._window.run()
        # # return 0 
        # return 0 

        """ee-to-us"""
        is_cartesian                 = False 
        self._robot.visual.parent    = self._root
        path_calib_data_from_matlab  = "./database/tform_ee_to_sensor_opt_result.mat"
        self._tform_ee_to_us_opt_res = scipy.io.loadmat( path_calib_data_from_matlab )["tform_ee_to_sensor_opt_result"]
        self._tcp_xform              = np.asarray( self._tform_ee_to_us_opt_res )
        # self._tcp_xform[2, 3]        = self._tcp_xform[2, 3] - 0.1
        self._ee_to_tcp_tform        = self._tcp_xform
        self._sensor_widget          = TransformListWidget( parent = self._root, scale = 0.25 )
        self._sensor_widget.load( [self._robot.kinematics.forward( self._qs_home ) @ self._tform_ee_to_us_opt_res] )
        
        """sensor_to_base"""
        self._tform_sensor_to_base  = np.eye(4)
        path_sensor_to_base_matlab  = "./data_rgbd/tform_sensor_to_base_post.mat"
        self._tform_sensor_to_base_in_matlab  = scipy.io.loadmat( path_sensor_to_base_matlab )["tform_sensor_to_base_post"]
        vec_x_from_matlab           = self._tform_sensor_to_base_in_matlab[0:3,0]
        vec_y_from_matlab           = self._tform_sensor_to_base_in_matlab[0:3,1]
        vec_z_from_matlab           = self._tform_sensor_to_base_in_matlab[0:3,2]
        pts_org_from_matlab         = self._tform_sensor_to_base_in_matlab[0:3,3]
        self._tform_sensor_to_base[0:3,0] =  vec_x_from_matlab
        self._tform_sensor_to_base[0:3,1] =  vec_y_from_matlab
        self._tform_sensor_to_base[0:3,2] =  vec_z_from_matlab
        self._tform_sensor_to_base[0:3,3] =  np.asanyarray([ pts_org_from_matlab[0], pts_org_from_matlab[1], pts_org_from_matlab[2]])
        self._tform_sensor_to_base = np.array([
            [ 0.01035144, -0.9993305 ,  0.03509121, -0.39767365],
            [-0.99896007, -0.00877654,  0.04474084, -0.16193797],
            [-0.0444029 , -0.03551785, -0.99838212,  0.60935745],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ], dtype=float)



        tform_form_phantom_to_world         = np.eye(4)
        tform_form_phantom_to_world[0:3,3]  = np.asarray([-0.4, -0.10, 0.0])

        """vis: phantom models"""
        pts_phantom_xyz_hc                  = np.hstack([ pts_finger_xyz, np.ones([ pts_finger_xyz.shape[0], 1])])
        pts_phantom_xyz_hc_in_world         = np.transpose( self._tform_sensor_to_base @ np.transpose( pts_phantom_xyz_hc ) )
        pts_phantom_in_world                = pts_phantom_xyz_hc_in_world[:,0:3]
        pts_rgbt_raster                     = pts_finger_rgb
        pts_rgbt_raster                     = np.hstack([pts_rgbt_raster, np.ones([pts_rgbt_raster.shape[0], 1]) * 8 ])
        pc_phantom_widget                   = PointCloudWidget(parent = self._root, point_size = 15 )
        pc_phantom_widget.load( np.ascontiguousarray( pts_phantom_in_world ), np.ascontiguousarray( pts_rgbt_raster ) )
        self._xyz_finger_in_base            = pts_phantom_in_world
        self._rgb_finger_in_base            = pts_finger_rgb

        """show the fid point cloud"""
        pts_fid_xyz_hc                      = np.hstack([ pts_finger_fid, np.ones([ pts_finger_fid.shape[0], 1])])
        pts_fid_xyz_hc_in_world             = np.transpose( self._tform_sensor_to_base @ np.transpose( pts_fid_xyz_hc ) )
        pts_fid_in_world                    = pts_fid_xyz_hc_in_world[:,0:3]
        pts_rgbt_raster                     = np.zeros( pts_fid_in_world.shape) 
        pts_rgbt_raster[:,2]                = 255
        pts_rgbt_raster                     = np.hstack([pts_rgbt_raster, np.ones([pts_rgbt_raster.shape[0], 1]) * 255 ])
        pc_fid_widget                       = PointCloudWidget(parent = self._root, point_size = 25 )
        pc_fid_widget.load( np.ascontiguousarray( pts_fid_in_world ), np.ascontiguousarray( pts_rgbt_raster ) )
        self._finger_fid_in_base            = pts_fid_in_world

        """plan for the trajecotry -- for the 9-different joints"""
        self._pts_finger_joint              = pts_fid_in_world
        self._qs_home                       = [-0.02121813, -1.2175811, 1.36699295, -1.73946172, -1.55588037, -1.59545452]

        print(pts_phantom_in_world.shape)
        print(pts_phantom_in_world[0])
        print(pts_fid_in_world.shape)
        print(pts_fid_in_world[0])
        # exit()

        """define the global image folders"""
        ###########################
        # input("update the folder first")
        # self._img_folder_in_sim_us = "./finger_exp_2_test/"
        #########################################

        """run the program"""
        # generate the robot simulation widget 
        self._pc_us_widget                  = PointCloudWidget(parent = self._root)
        # Thread(target = self.main_loop_for_calib_auto, daemon = True).start()
        Thread(target = self.main_loop_test2, daemon = True).start()

        # run the windows
        view_pose = np.asarray( [ [ 0.66752115,  0.51424116,  0.53849007, -3.39468878  ],
                                  [-0.71273616,  0.23202099,  0.66194669, -1.494105045 ],
                                  [ 0.21545923, -0.82566476,  0.52139719,  5.01182138  ],
                                  [ 0.        ,  0.        ,  0.        ,  1.          ] ] )

        self._window.camera_pose = view_pose  
        self._window.run()

    # def main_loop_test2(self):
    #     """
    #     Execute a joint-space scanning trajectory using precomputed joint path pts_finger_q_traj.
    #     This replaces main_loop_test + move_to_waypoint_from_traj.
    #     """

    #     print("===================================================")
    #     print("        main loop test 2 (IK-based trajectory)      ")
    #     print("===================================================")

    #     pts_finger_q_traj = self.pts_finger_q_traj

    #     # pts_finger_q_traj is expected to be shape (N, 6)
    #     if pts_finger_q_traj is None or len(pts_finger_q_traj) == 0:
    #         print("❌ No joint trajectory provided. Exit.")
    #         return

    #     num_of_pose = len(pts_finger_q_traj)
    #     print(f"Total poses in trajectory: {num_of_pose}")

    #     # Optional velocity limit
    #     qd_limit_local = self._qd_limit_input

    #     # --------------------------------------------------------
    #     # Main loop over the joint trajectory
    #     # --------------------------------------------------------
    #     for idx in range(num_of_pose):

    #         print(f"\n---- Executing waypoint {idx+1}/{num_of_pose} ----")

    #         # Update global index for downstream processes
    #         self.idx_global_oct_data = idx

    #         # Get the desired joint configuration
    #         q_target = pts_finger_q_traj[idx]

    #         # ------------------------------
    #         # Move robot to the IK solution
    #         # ------------------------------
    #         self._ptp_exp.move_joint(q_target, qd_limits=qd_limit_local)

    #         # ------------------------------
    #         # Update robot state
    #         # ------------------------------
    #         state_now = self._ptp_exp._controller.state
    #         q_now     = state_now.actual_q
    #         self._q_current = q_now

    #         # Forward kinematics (EE frame in world)
    #         self._ee_tform = self._robot.kinematics.forward(q_now)

    #         # Update ultrasound / sensor visualization frame
    #         tform_us_world = self._ee_tform @ self._tform_ee_to_us_opt_res
    #         self._sensor_widget.load([tform_us_world])

    #         # (Optional) update US frame / OCT imaging
    #         # self.us_img_vis_from_robot_config()
    #         self._img_vis_for_us_sim = "./data_raus/img_tmp.png"
    #         self.us_img_vis_from_robot_config() 

    #     # end for-loop
    #     self._stop_vis = True
    #     print("\n✓ Finished executing IK-based scanning trajectory")

    def main_loop_test2(self):
        """
        Execute IK-based trajectory.
        Valid OCT frame index updates only when idx is inside a real (non-interpolated) range
        defined by self.safe_group_ranges.
        """

        print("===================================================")
        print("        main loop test 2 (IK-based trajectory)      ")
        print("===================================================")

        pts_finger_q_traj = self.pts_finger_q_traj

        if pts_finger_q_traj is None or len(pts_finger_q_traj) == 0:
            print("❌ No joint trajectory provided. Exit.")
            return

        num_of_pose = len(pts_finger_q_traj)
        print(f"Total poses in trajectory: {num_of_pose}")

        # Retrieve saved ranges
        safe_group_ranges = self.safe_group_ranges     # list of (start_after, end_after)

        # Precompute a list mapping each idx → valid_index (or -1 for interpolated)
        valid_index_for_each_point = np.full(num_of_pose, -1, dtype=int)

        counter = 0
        for (s, e) in safe_group_ranges:
            for i in range(s, e + 1):
                valid_index_for_each_point[i] = counter
                counter += 1

        # Optional velocity limit
        qd_limit_local = self._qd_limit_input

        # --------------------------------------------------------
        # Main execution loop
        # --------------------------------------------------------
        for idx in range(num_of_pose):

            print(f"\n---- Executing waypoint {idx+1}/{num_of_pose} ----")

            # Determine whether idx is inside any safe (real-group) range
            valid_frame_index = valid_index_for_each_point[idx]
            is_valid = valid_frame_index != -1

            # Update only when valid
            if is_valid:
                self.idx_global_oct_data = valid_frame_index

            print(f"ValidFrameIndex = {self.idx_global_oct_data}  |  valid={is_valid}")

            # Desired joint configuration
            q_target = pts_finger_q_traj[idx]

            # Move robot
            self._ptp_exp.move_joint(q_target, qd_limits=qd_limit_local)

            # Update robot state
            state_now = self._ptp_exp._controller.state
            q_now = state_now.actual_q
            self._q_current = q_now

            # Forward kinematics
            self._ee_tform = self._robot.kinematics.forward(q_now)

            # Visualization
            tform_us_world = self._ee_tform @ self._tform_ee_to_us_opt_res
            self._sensor_widget.load([tform_us_world])

            # Imaging
            # self._img_vis_for_us_sim = "./data_raus/img_tmp.png"
            
            # Build path to the npy frame
            npy_path = f"./20251129_images_6/{self.idx_global_oct_data}_channel_0.npy"

            # Load B-mode npy
            img = np.load(npy_path)        # shape (H, W)

            # Normalize 0–255 for PNG
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_uint8 = img_norm.astype(np.uint8)

            # Convert grayscale → RGB (optional, safer for some viewers)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

            # Folder for PNG export
            png_dir = "./20251129_images_6_png"
            os.makedirs(png_dir, exist_ok=True)

            # Save PNG
            png_path = f"{png_dir}/{self.idx_global_oct_data}.png"
            cv2.imwrite(png_path, img_rgb)

            # Pass PNG path to visualization pipeline
            self._img_vis_for_us_sim = png_path

            self.us_img_vis_from_robot_config()

        self._stop_vis = True
        print("\n✓ Finished executing IK-based scanning trajectory")


    def main_loop_test(self):

        """solve the main loop testing problem"""
        print("main loop test")
        print("first official data collection platform")

        # run the robot (sim or exp modes)
        # consider the prev-and-next
        if len( self._scan_q_path ) == 0: 
            print("no robot traj exit")
            exit()
        num_of_pose_check = len( self._scan_q_path ) 

        for idx_tmp in range( num_of_pose_check ): 

            # update the global index
            self.idx_global_oct_data = idx_tmp
            print("idx_tmp = ", idx_tmp)

            # # step-1: update the robot movement given the current state and target tform 
            self.move_to_waypoint_from_traj() 

            # update the imaging plane 
            # 2D-ultrasound imaging 
            # self.us_img_vis_from_robot_config() 

            # update the {oct} frame 
            state_tmp                   = self._ptp_exp._controller.state # self._ptp_exp.state 
            self._ee_tform              = self._robot.kinematics.forward( state_tmp.actual_q ) 
            # tform_us_in_world_tmp       = self._ee_tform @ self._tcp_xform
            self._sensor_widget.load( [ self._robot.kinematics.forward( state_tmp.actual_q )  @ self._tform_ee_to_us_opt_res] )
            self._q_current             = state_tmp.actual_q

        self._stop_vis = True

        print("finished the current scanning pattern")

    def ik_hand_traj(self):

        # solve the IK: {base} -> {ee} -> {oct}
        pts_cone_in_world   = self._pts_finger_joint
        ee_path             = []  
        for idx_tmp in range( pts_cone_in_world.shape[0] ): 

            # get the local geometry = [ vec_x, vec_y, vec_z ]
            vec_orientation_use     =  np.asanyarray([0.0, 0.0, +1.0])
            vec_z                   = -vec_orientation_use / np.linalg.norm( vec_orientation_use )
            vec_ref_y               =  [0.0, 1.0, 0.0]
            vec_ref_x               = -np.cross(vec_z, vec_ref_y)
            vec_x                   =  vec_ref_x / np.linalg.norm( vec_ref_x )            
            vec_y                   = -np.cross(vec_x, vec_z) 
            vec_y                   =  vec_y / np.linalg.norm( vec_y )

            # xform_target
            xform_target            = np.eye(4)
            xform_target[0:3, 0]    = vec_x
            xform_target[0:3, 1]    = vec_y
            xform_target[0:3, 2]    = vec_z     
            xform_target[0:3, 3]    = pts_cone_in_world[idx_tmp,:] 
            
            # get the IK-target frame for {ee}
            ee_tform_target                         = np.matmul( xform_target, np.linalg.inv( self._ee_to_tcp_tform ) ) 
            ee_path.append( ee_tform_target )

        # solve the Ik-based trajectory 
        ref_q                           = self._robot.kinematics.inverse_nearest( ee_path[0], self._qs_home  )
        val_check                       = self._robot.kinematics.forward( ref_q ) 
        ee_path_new                     = [val_check] + ee_path
        q_path                          = ik_path( ref_q, ee_path_new, self._robot.kinematics )
        q_path_pivot_pts                = q_path[1:]

        return q_path_pivot_pts

    def traj_linear_and_arc_from_pivot_point(self):
        """move the center probe to the target pivot point configuration
        1. test the IK trajectory system with the proposed models 
        2. test the IK testing system 
        """

        # TODO: define the home position
        # 1.0 cm above the surfacee
        # local start point
        # local end point

        # solve the IK: {base} -> {ee} -> {oct}
        pts_cone_in_world           = self._pts_finger_joint
        list_of_pts_start_traj      = [] 

        # define the trajectory 
        ee_path                     = []  
        for idx_tmp in range( pts_cone_in_world.shape[0] ): 

            # get the local geometry = [ vec_x, vec_y, vec_z ]
            vec_orientation_use     =  np.asanyarray([0.0, 0.0, +1.0])
            vec_z                   = -vec_orientation_use / np.linalg.norm( vec_orientation_use )
            vec_ref_y               =  [0.0, 1.0, 0.0]
            vec_ref_x               = -np.cross(vec_z, vec_ref_y)
            vec_x                   =  vec_ref_x / np.linalg.norm( vec_ref_x )            
            vec_y                   = -np.cross(vec_x, vec_z) 
            vec_y                   =  vec_y / np.linalg.norm( vec_y )

            # xform_target
            xform_target            = np.eye(4)
            xform_target[0:3, 0]    = vec_x
            xform_target[0:3, 1]    = vec_y
            xform_target[0:3, 2]    = vec_z     
            xform_target[0:3, 3]    = pts_cone_in_world[idx_tmp,:] 

            # get the IK-target frame for {ee}
            ee_tform_target         = np.matmul( xform_target, np.linalg.inv( self._ee_to_tcp_tform ) ) 
            ee_path.append( ee_tform_target )

        # solve the Ik-based trajectory 
        ref_q                           = self._robot.kinematics.inverse_nearest( ee_path[0], self._qs_home  )
        val_check                       = self._robot.kinematics.forward( ref_q ) 
        ee_path_new                     = [val_check] + ee_path
        q_path                          = ik_path( ref_q, ee_path_new, self._robot.kinematics )
        q_path_pivot_pts                = q_path[1:]

        """linear-scanning"""
        self._scan_q_path_pivot                 = q_path_pivot_pts
        self._scan_q_path                       = [] 
        self._scan_mode                         = "linear"
        self._global_robot_config_init          = []
        self._linear_scan_num_of_line           = 70
        self._linear_scan_len_line_half         = 0.02
        self._list_of_folder_auto_linear        = []
        self._list_of_robot_traj_auto_linear    = []

        for idx_tmp in range( len(self._scan_q_path_pivot) ): 

            print("idx_tmp = ", idx_tmp)

            # obtain the features 
            q_path_linear = self.linear_scan_from_pivot_point( q_tmp = self._scan_q_path_pivot[idx_tmp] )

            # obtain the starting point
            # self._list_of_robot_traj_auto_linear.append( list_of_pts_start_traj[idx_tmp] )

            # obtain the q-path 
            self._list_of_robot_traj_auto_linear.append( q_path_linear )

            # obtain the ending point 
            # self._list_of_robot_traj_auto_linear.append( list_of_pts_start_traj[idx_tmp] )

            # update the folders
            idx_folder_unique_name             = "calib_auto_linear_" + str(idx_tmp)
            self._path_data_unique_folder      = "./data_raus/fast_scan_1/" + idx_folder_unique_name +  "/"
            idx_exist_folder                   = os.path.isdir(self._path_data_unique_folder)
            if idx_exist_folder == False:
                print("The folder does not exist")
                os.mkdir(self._path_data_unique_folder)

            # save the 
            self._list_of_folder_auto_linear.append( self._path_data_unique_folder )

    def arc_scan_from_pivot_point(self, q_tmp = [] ): 
        """arc-scan around the centerized pivot points"""

        self._global_robot_config_init  = []
        num_of_arc_scan_pts             = self._num_of_arc_scan_pts 
        ang_scan_half_range             = self._ang_scan_half_range 
        self._qs_pivot_global           = q_tmp

        # sample rotation frames
        slerp    = Slerp([0, 1], Rotation.from_euler('xyz', [[-ang_scan_half_range, 0, 0], [+ang_scan_half_range, 0, 0]], degrees = True))
        r        = np.linspace(0, 1, num_of_arc_scan_pts)
        steps    = slerp(r)
        ee_xform = self._robot.kinematics.forward( self._qs_pivot_global )
        ee_path  = [ee_xform @ self._tcp_xform @ _T(s) @ hinv( self._tcp_xform ) for s in steps]

        # inverse kinematics
        ref                     = self._qs_pivot_global
        val_check               = self._robot.kinematics.forward( ref ) 
        ee_path_new             = [val_check] + ee_path
        q_path                  = ik_path(ref, ee_path_new, self._robot.kinematics )
        q_path                  = q_path[1:]

        # summarize the scanning path 
        self._scan_q_path       = q_path

        # initialize the scanning index
        self._scan_q_path_index = 0

        return q_path

    def linear_scan_from_pivot_point(self, q_tmp = [] ): 

        self._global_robot_config_init  = []
        var_data                        = self._scan_mode  
        num_of_line                     = self._linear_scan_num_of_line
        len_line_half                   = self._linear_scan_len_line_half
        tcp_xform                       = self._tcp_xform 
        self._qs_pivot_global           = q_tmp

        # ee-tform
        ee_xform = self._robot.kinematics.forward( self._qs_pivot_global )

        # compute line translational offsets
        offsets         = np.linspace([0, -len_line_half, 0], [0, +len_line_half, 0], num_of_line)
        steps           = []
        for offset in offsets:
            xform = np.eye(4)
            xform[:3, 3] = offset
            steps.append(xform)

        # transform in to EE frame
        ee_path = [ee_xform @ tcp_xform @ s @ hinv(tcp_xform) for s in steps]

        # inverse kinematics
        ref     = self._robot.kinematics.inverse_nearest( ee_path[0], self._qs_home )
        q_path  = ik_path(ref, ee_path, self._robot.kinematics)

        return q_path

    def move_to_waypoint_from_traj(self): 

        # move to the first waypoint 
        # get the single waypoint 
        qd_limit_input_local = self._qd_limit_input

        # move to the current waypoint of the trajectory
        q_tmp                = self._scan_q_path[ self.idx_global_oct_data ]
        # print("q_tmp = ", q_tmp)

        # move to the target joint angle
        self._ptp_exp.move_joint( q_tmp, qd_limits = qd_limit_input_local )

        # # save the joint to the local fodler
        # np.save( self._path_data_unique_folder + "robot_config_" + str( self.idx_global_oct_data) + ".npy", q_tmp )

    def us_img_vis_from_robot_config(self): 

        # image scaling factors (this im)
        img_proj_scale                      = 0.05
        ratio_meter_to_mm                   = 0.001
        len_pxl_height                      = int( 1083 * img_proj_scale )
        len_pxl_width                       = int( 903  * img_proj_scale ) 
        len_mm_height                       = 30.0
        len_mm_width                        = 25.0
        ratio_of_height                     = (len_mm_height) / (len_pxl_height - 1)        
        ratio_of_width                      = (len_mm_width)  / (len_pxl_width  - 1) 

        # save the image = bmode + cfm + pat 
        path_img_tmp                        = self._img_vis_for_us_sim  # self._folder_tmp_use + "./database/img_bmode_test.png" # self._path_data_unique_folder + str(self.idx_global_oct_data) + "_bmode.png"  # 
        img_us_circle_tmp = cv2.imread(path_img_tmp)
        # Flip left ↔ right
        img_us_circle_tmp = cv2.flip(img_us_circle_tmp, 1)

        img_us_circle_rescale               = cv2.resize( img_us_circle_tmp, (len_pxl_width, len_pxl_height)) 

        # display the 3d sensor fusion point cloud 
        pts_line_width_in_mm                = np.linspace( 0, len_pxl_width  - 1, len_pxl_width ) * ratio_of_width 
        pts_line_height_in_mm               = np.linspace( 0, len_pxl_height - 1, len_pxl_height ) * ratio_of_height
        [x_mesh_width, y_mesh_width]        = np.meshgrid( pts_line_width_in_mm, pts_line_height_in_mm )
        pts_xyz                             = np.zeros( [len_pxl_width * len_pxl_height, 3] )
        pts_xyz[:,0]                        = x_mesh_width.ravel() * ratio_meter_to_mm
        pts_xyz[:,1]                        = y_mesh_width.ravel() * ratio_meter_to_mm
        pts_xyz[:,2]                        = np.zeros( [ len(y_mesh_width.ravel()), 1] ).reshape(-1)
        delta_xyz_offset                    = np.asarray([-0.0, -0.0, 0.0])
        pts_xyz                             = pts_xyz + delta_xyz_offset
    
        # projection to the 3D local coordinates
        tform_us_in_world                   = self._robot.kinematics.forward( self._q_current ) @ self._tcp_xform 
        vec_x_us_in_world                   = tform_us_in_world[0:3,0]
        vec_y_us_in_world                   = tform_us_in_world[0:3,1]
        vec_z_us_in_world                   = tform_us_in_world[0:3,2]
        pts_us_in_world_org                 = np.transpose( tform_us_in_world[0:3,3] ) + (-1) * vec_x_us_in_world * len_mm_width * ratio_meter_to_mm * 0.5 
        pts_xyz_proj_in_world               = pts_xyz[:,0].reshape(-1,1) * vec_x_us_in_world + pts_xyz[:,1].reshape(-1,1) * vec_z_us_in_world + pts_us_in_world_org
        pts_rgb                             = img_us_circle_rescale.reshape( [len_pxl_width * len_pxl_height, 3] )
        pts_rgb                             = np.uint8( pts_rgb )
        pts_rgbs                            = np.hstack([ pts_rgb, np.zeros([pts_rgb.shape[0], 1]) + 255 ])

        """save thc object xyz and rgb to the system"""
        # self._path_demo_finger_in_base
        pts_us_xyz_in_base_for_vis = pts_xyz_proj_in_world
        pts_us_rgb_in_base_for_vis = pts_rgb
        # np.save( self._path_demo_finger_in_base + str(self.idx_global_oct_data) + "_xyz.npy", pts_us_xyz_in_base_for_vis )
        # np.save( self._path_demo_finger_in_base + str(self.idx_global_oct_data) + "_rgb.npy", pts_us_rgb_in_base_for_vis )

        # show the model 
        self._pc_us_widget.load( np.ascontiguousarray( pts_xyz_proj_in_world ), np.ascontiguousarray( pts_rgbs ) ) 

    def main_loop_for_calib_auto(self): 

        """time to set the vciewpoint"""
        # time.sleep(3.0)

        for idx_case in range( len( self._list_of_robot_traj_auto_linear )): 

            """firstly: linear-scan"""
            traj_robot           = self._list_of_robot_traj_auto_linear[idx_case]
            self._scan_q_path    = traj_robot
            folder_robot         = self._list_of_folder_auto_linear[idx_case]
            self._path_data_unique_folder = folder_robot

            # save the information
            if len( self._scan_q_path ) == 0: 
                print("no robot traj exit")
                exit()

            # current image folder 
            # self._folder_tmp_use = self._img_folder_in_sim_us + "calib_auto_linear_" + str(idx_case) + r"\\"
            # print("folder_tmp_use = ", self._folder_tmp_use)
            # exit()

            # """data visualization presetting folders + system configurations"""
            # self._path_demo_finger_in_base  = self._folder_tmp_use + r"img_finger_in_base\\"
            # if os.path.isdir( self._path_demo_finger_in_base ) == False:
            #     os.mkdir( self._path_demo_finger_in_base )
            # np.save( self._path_demo_finger_in_base + "xyz_finger_in_base.npy", self._xyz_finger_in_base)
            # np.save( self._path_demo_finger_in_base + "rgb_finger_in_base.npy", self._rgb_finger_in_base)
            # np.save( self._path_demo_finger_in_base + "fid_finger_in_base.npy", self._finger_fid_in_base)

            # continue

            # move to the home position 
            # self._ptp_exp.move_joint( q = self._qs_home, qd_limits = self._qd_limit_input ) 

            num_of_pose_check = len( self._scan_q_path ) 
            for idx_tmp in range( num_of_pose_check ): 

                # update the global index
                # # step-1: update the robot movement given the current state and target tform 
                # time.sleep(0.5)
                self.idx_global_oct_data = idx_tmp
                self.move_to_waypoint_from_traj() 

                # set the image folder for the system configuration
                # self._img_vis_for_us_sim = self._folder_tmp_use + str(idx_tmp) + "_bmode.png"
                self._img_vis_for_us_sim = "./data_raus/img_tmp.png"  # self._folder_tmp_use + str(idx_tmp) + "_bmode.png"

                # update the {oct} frame 
                state_tmp                   = self._ptp_exp._controller.state # self._ptp_exp.state
                self._ee_tform              = self._robot.kinematics.forward( state_tmp.actual_q ) 
                self._sensor_widget.load( [ self._robot.kinematics.forward( state_tmp.actual_q )  @ self._tform_ee_to_us_opt_res] )
                self._q_current             = state_tmp.actual_q

                if self._is_ultrasound_mode:

                    # time step setpoint
                    # save the image = bmode + cfm + pat 
                    if idx_tmp == 0:
                        print("moving to the first position")
                        time.sleep(0.10)
                    else:
                        time.sleep(0.10)

                    # # obtain the image
                    # # TODO: ultrasound image model 
                    self.idx_in_loop = idx_tmp
                    if self._is_ultrasound_mode:
                        self.save_us_img_tmp()
                        print("finished image saving")

                # TODO: vis the image
                # update the imaging plane 
                # 2D-ultrasound imaging 
                self.us_img_vis_from_robot_config() 

            # move to the home position 
            # self._ptp_exp.move_joint( q = self._qs_home, qd_limits = self._qd_limit_input ) 

        """stop the data collections"""
        # stop the us
        if self._is_ultrasound_mode:
            time.sleep(0.5)
            self._obj_ultrasound.stop() 
            exit()
            print("finished the current scanning pattern") 

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
                    if np.max(img_save_bmode.ravel()) > 220:
                        is_exit_bmode = True
                        cv2.imwrite( self._path_data_unique_folder + str( self.idx_in_loop ) + "_bmode.png", img_save_bmode)

                """cfm config"""
                if self._is_collect_cfm:
                    # save the cfm image 
                    if frame.format == FORMAT_CFM:
                        img_save_cfm = img
                        if np.max(img_save_cfm.ravel()) > 10:
                            is_exit_cfm = True
                            cv2.imwrite( self._path_data_unique_folder + str( self.idx_in_loop ) + "_cfm.png", img_save_cfm)
                            np.save( self._path_data_unique_folder + str( self.idx_in_loop ) + "_cfm.npy", img_raw_for_npy ) 
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

    def cone_scan_in_world(self, pts_pivot_cen = [] ):
        """cone-based scanning patterns in {world}
        1. points at the surface cone  
        2. orientation vector defined in {world}
        """

        is_x_rotate_only                = self._para_input_cone_scan["is_x_rotate_only"]
        is_y_rotate_only                = self._para_input_cone_scan["is_y_rotate_only"]
        list_of_angle_x_raw             = self._para_input_cone_scan["list_of_angle_x_raw"]
        list_of_angle_y_raw             = self._para_input_cone_scan["list_of_angle_y_raw"]
        flag_vis                        = self._para_input_cone_scan["flag_vis"]
        idx_pts_local                   = self._para_input_cone_scan["idx_pts_local"]
        z_height_scanning_grid          = self._para_input_cone_scan["z_height_scanning_grid"]
        num_of_waypts                   = self._para_input_cone_scan["num_of_waypts"]
        step_size_in_line               = self._para_input_cone_scan["step_size_in_line"]
        idx_skip                        = self._para_input_cone_scan["idx_skip"]
        is_remove_first_index           = self._para_input_cone_scan["is_remove_first_index"]

        # print("list_of_angle_y_raw = ", list_of_angle_y_raw)
        # exit() 

        if is_x_rotate_only == "true": 
            list_of_angle_x            = list_of_angle_x_raw
            x_angle_use                = list_of_angle_x.ravel() 
            y_angle_use                = np.zeros(x_angle_use.shape)
        elif is_y_rotate_only == "true": 
            list_of_angle_y            = list_of_angle_y_raw
            y_angle_use                = list_of_angle_y.ravel() 
            x_angle_use                = np.zeros(y_angle_use.shape)
        else:
            list_of_angle_x            = list_of_angle_x_raw
            list_of_angle_y            = list_of_angle_y_raw
            [x_mesh, y_mesh]           = np.meshgrid(list_of_angle_x, list_of_angle_y)
            x_angle_use                = x_mesh.ravel()
            y_angle_use                = y_mesh.ravel()

        # print("x_angle_use = ", x_angle_use)
        # print("y_angle_use = ", y_angle_use)
        # print("list_of_angle_x = ", list_of_angle_x)
        # print("list_of_angle_y = ", list_of_angle_y)
        # print("list_of_angle_y_raw = ", list_of_angle_y_raw)
        # exit() 

        pts_grid                        = np.zeros((1, 3)) 
        arr_vec_x                       = np.zeros([1, 3])
        arr_vec_y                       = np.zeros([1, 3])
        arr_vec_z                       = np.zeros([1, 3])
        num_of_pose                     = len(x_angle_use)

        # update the vector orientation 
        # vec_perpendicular = [] 
        for idx_ang in range( num_of_pose ):
            
            # orientation cone configuration 
            theta_orientation_x         = 0.0 / 180.0 * np.math.pi
            theta_orientation_y         = x_angle_use[idx_ang] / 180.0 * np.math.pi
            theta_orientation_z         = y_angle_use[idx_ang] / 180.0 * np.math.pi
            matrix_rotation             = so3.from_rpy( [theta_orientation_x, theta_orientation_y, theta_orientation_z] )
            vec_orientation             = np.asarray([1.0, 0.0, 0.0])
            vec_orientation_use         = so3.apply(matrix_rotation, vec_orientation)
            vec_orientation_use         = vec_orientation_use / np.linalg.norm(vec_orientation_use)    

            # waypoints
            for id_step in range(num_of_waypts): 

                # start from the second index 
                if id_step < idx_skip:
                    continue

                # update the cone-based trajectory 
                pts_local_tmp           = pts_pivot_cen + id_step * step_size_in_line * vec_orientation_use
                pts_grid                = np.vstack([pts_grid, np.asarray([pts_local_tmp[0], pts_local_tmp[1], pts_local_tmp[2]])])

                # get the local geometry = [ vec_x, vec_y, vec_z ]
                vec_z                   = -vec_orientation_use / np.linalg.norm( vec_orientation_use )
                vec_ref_y               =  [0.0, 1.0, 0.0]
                vec_ref_x               = -np.cross(vec_z, vec_ref_y)
                vec_x                   =  vec_ref_x / np.linalg.norm( vec_ref_x )            
                vec_y                   = -np.cross(vec_x, vec_z) 
                vec_y                   =  vec_y / np.linalg.norm( vec_y )
                arr_vec_x               =  np.vstack([arr_vec_x, np.asanyarray(vec_x)])
                arr_vec_y               =  np.vstack([arr_vec_y, np.asanyarray(vec_y)])
                arr_vec_z               =  np.vstack([arr_vec_z, np.asanyarray(vec_z)])

        # remove the first index
        if is_remove_first_index: 
            arr_vec_x                       = arr_vec_x[1:,:]
            arr_vec_y                       = arr_vec_y[1:,:]
            arr_vec_z                       = arr_vec_z[1:,:]
            pts_grid                        = pts_grid[1:,:]
        else:
            arr_vec_x                       = arr_vec_x[0:,:]
            arr_vec_y                       = arr_vec_y[0:,:]
            arr_vec_z                       = arr_vec_z[0:,:]
            pts_grid                        = pts_grid[0:,:]

        pts_cone_in_world               = pts_grid

        # vis 
        if flag_vis == "true":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pts_cone_in_world[0,0], pts_cone_in_world[0,1], pts_cone_in_world[0,2], c = 'r', marker='o')
            ax.scatter(pts_cone_in_world[:,0], pts_cone_in_world[:,1], pts_cone_in_world[:,2], c = 'b', marker='o')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            plt.show()

        # summarize the outputs 
        para_output = {}
        para_output["pts_cone_in_world"]    = pts_cone_in_world
        para_output["arr_vec_x"]            = arr_vec_x
        para_output["arr_vec_y"]            = arr_vec_y
        para_output["arr_vec_z"]            = arr_vec_z

        return para_output 

    def load_sphere_phantom(self, pts_pivot_for_cone = [] ):
        """load the 3D point cloud of the sphere target"""

        # sample the vectors towards the radius and the center
        # pts_pivot_for_cone                                      = np.asanyarray([-0.35, -0.10, 0.10])
        self._para_input_cone_scan                              = {}
        self._para_input_cone_scan["is_x_rotate_only"]          = "false"
        self._para_input_cone_scan["is_y_rotate_only"]          = "false"
        self._para_input_cone_scan["list_of_angle_x_raw"]       = np.linspace( -180, +180, 20 )
        self._para_input_cone_scan["list_of_angle_y_raw"]       = np.linspace( -180, +180, 20 )
        self._para_input_cone_scan["flag_vis"]                  = "false"
        self._para_input_cone_scan["idx_pts_local"]             = 0.0
        self._para_input_cone_scan["z_height_scanning_grid"]    = 0.5
        self._para_input_cone_scan["num_of_waypts"]             = 2
        self._para_input_cone_scan["step_size_in_line"]         = 0.005 * 1
        self._para_input_cone_scan["idx_skip"]                  = 1 
        self._para_input_cone_scan["is_remove_first_index"]     = True

        pts_cen_cone_input_pivot                                = pts_pivot_for_cone # + dis_cen_offset_for_cone_scan
        para_cone_scan_output                                   = self.cone_scan_in_world(pts_pivot_cen = pts_cen_cone_input_pivot)
        pts_cone_in_world                                       = para_cone_scan_output["pts_cone_in_world"] 
        pts_traj_in_cone                                        = pts_cone_in_world 
        
        # show the point cloud widgets
        pts_rgbt_raster                                         = np.zeros(pts_traj_in_cone.shape) * 255
        pts_rgbt_raster[:,0]                                    = 255
        pts_rgbt_raster                                         = np.hstack([pts_rgbt_raster, np.ones([pts_rgbt_raster.shape[0], 1]) * 255 ])
        PointCloudWidget(parent = self._root).load( np.ascontiguousarray(pts_traj_in_cone), np.ascontiguousarray(pts_rgbt_raster) )

        # center of referencese
        pts_center_cluster                                      = np.vstack([pts_cen_cone_input_pivot, pts_cen_cone_input_pivot])
        pts_rgbt_raster                                         = np.zeros(pts_center_cluster.shape) * 255
        pts_rgbt_raster[:,0]                                    = 255
        pts_rgbt_raster                                         = np.hstack([pts_rgbt_raster, np.ones([pts_rgbt_raster.shape[0], 1]) * 255 ])
        PointCloudWidget(parent = self._root).load( np.ascontiguousarray(pts_center_cluster), np.ascontiguousarray(pts_rgbt_raster) )

        # define the local frame 
        tform_sphere_center             = np.eye(4)
        tform_sphere_center[0:3,3]      = pts_pivot_for_cone[:]
        tform_sphere_center[0:3,0]      = np.asanyarray([1.0, 0.0, 0.0])
        tform_sphere_center[0:3,1]      = np.asanyarray([0.0, 1.0, 0.0])
        tform_sphere_center[0:3,2]      = np.asanyarray([0.0, 0.0, 1.0])
        tform_sphere_center_widget      = TransformListWidget( parent = self._root, scale = 0.10 )
        tform_sphere_center_widget.load( [tform_sphere_center] )

        return para_cone_scan_output
    
if __name__ == "__main__": 

    test_class = raus_fast_scanning()

    # test-1: testing exp 
    # test_class.integration_testing_exp() 

    # test-2: fast scanning mode
    test_class.integration_finger_exp() 
