import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
from enum import Enum, auto
from collections import deque

UR5E_D1 = 0.1625    # base to shoulder link
UR5E_D2 = 0.425     # shoulder to elbow link
UR5E_D3 = 0.39225   # elbow to wrist1 link
UR5E_D4 = 0.1333    # shoulder lateral offset
UR5E_D5 = 0.0997    # wrist1 to wrist2 link
UR5E_D6 = 0.0996    # wrist2 to tool flange

INIT_UR5E_POSE = [0.0, -2.14, 2.11, -3.14, -1.57, 0.0]
INIT_ALLEGRO_POSE = [0.0, 0.6, 0.5, 0.33, 0.0, 0.6, 0.5, 0.33, 0.0, 0.6, 0.5, 0.33, 0.0, 1.0, 0.0, 0.0]
PAN_POSE = [0.0, -2.14, 2.11, -3.14, -1.57, 0.0]
STRETCH_POSE = [0.0, 0.0, 0.0, 0.0, -1.57, 0.0]
ALLEGRO_GRASP = [0.0, 1.3, 0.8, 0.8, 0.0, 1.3, 0.8, 0.8, 0.0, 1.3, 0.8, 0.8, 0.0, 1.15, 0.9, 1.0]
BACK = [0.0, -1.57, 1.57, -1.13, -1.57, 0.0]
BASKET = [-1.88, -1.57, 1.57, -1.13, -1.57, 0.0]
ALLEGRO_PUT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

class RobotState(Enum):
    INIT = auto()
    SEARCH_APPLE = auto()
    APPROACH = auto()
    FINE_TUNING = auto()
    RETRACT = auto()
    GRASP = auto()
    PUT = auto()

MENAGERIE = os.path.expanduser("~/sim_model/mujoco_menagerie")
XML = os.path.join(MENAGERIE, "universal_robots_ur5e", "ur5e_V1.0.xml")

class Ur5eControl(Node):
    def __init__(self):
        super().__init__("ur5e_control_node")

        self.bridge = CvBridge()

        self.head_cam_sub = self.create_subscription(Image, '/head_cam/image_raw', self.head_cam_cb, 10)
        self.head_cam_depth_sub = self.create_subscription(Image, '/head_cam/image_depth', self.head_cam_depth_cb, 10)
        self.joint_status_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.site_pose_sub = self.create_subscription(Point, '/palm_center_pose', self.palm_center_pose_cb, 10)

        self.ur5e_joint_pub = self.create_publisher(JointState, '/ur5e_joint', 10)
        self.allegro_joint_pub = self.create_publisher(JointState, '/allegro_joint', 10)
        self.head_cam_joint_pub = self.create_publisher(JointState, '/head_cam_joint', 10)
        self.detect_pub = self.create_publisher(String, '/detect', 10)

        self.control_timer = self.create_timer(0.1, self.main_control_loop)

        self.state = RobotState.INIT
        self.init_count = 0
        self.init_timeout = 100

        self.pan_count = 0
        self.pan_timeout = 30

        self.ur5e_joint = [0.0] * 6
        self.allegro_joint = [0.0] * 16
        self.head_cam_joint = [0.0] * 3

        self.check_obj = False
        self.latest_mask = None
        self.static_obj_pose = False
        self.axis_x = 0
        self.axis_y = 0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.static_target = 0.0

        self.palm_center_pose_x = 0.0
        self.palm_center_pose_y = 0.0
        self.palm_center_pose_z = 0.0

        self.r = 0.0
        self.h = 0.0
        self.d = 0.0

        self.count = 0.0
        self.max_count = 100

    def transform_function(self, alpha, beta, gamma, x, y ,z):
        Rx = np.array([[1,            0,             0],
                       [0,np.cos(alpha),-np.sin(alpha)],
                       [0,np.sin(alpha), np.cos(alpha)]])
        
        Ry = np.array([[ np.cos(beta),0,np.sin(beta)],
                       [0,            1,           0],
                       [-np.sin(beta),0,np.cos(beta)]])
        
        Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                       [np.sin(gamma), np.cos(gamma),0],
                       [0,            0,             1]])
        
        R = Rx @ Ry @ Rz
        P_Cam = np.array([x, y, -z])
        P_Offset = np.array([0.0, -0.3, 1.0])

        P_World = (R @ P_Cam) + P_Offset

        return P_World
    
    def shoulder_theta(self):
        dist_sq = self.target_x ** 2 + self.target_y ** 2
        phi = np.arctan2(self.target_y, self.target_x)
        alpha = np.arccos(np.clip(UR5E_D4 / np.sqrt(dist_sq), -1, 1))
        raw_target = phi + alpha + np.pi / 2
        current_pan = self.ur5e_joint[0]
        diff = (raw_target - current_pan + np.pi) % (2 * np.pi) - np.pi
        final_theta = current_pan + diff

        return final_theta
    
    def lift_theta(self):
        alpha = np.arctan2(self.h, self.r)

        cos_beta = (UR5E_D2 ** 2 + self.d ** 2 - UR5E_D3 ** 2) / (2 * UR5E_D2 * self.d)
        beta = np.arccos(np.clip(cos_beta, -1, 1))

        return -(alpha + beta)

    def elbow_theta(self, LIFT = UR5E_D2, ELBOW = UR5E_D3):
        r_base = np.sqrt(self.target_x ** 2 + self.target_y ** 2)
        r_plane = np.sqrt(np.maximum(r_base ** 2 - UR5E_D4 ** 2, 0))
        self.r = r_plane - UR5E_D6
        self.h = self.target_z - 0.1625
        self.d = np.sqrt(self.r ** 2 + self.h ** 2)
        cos_theta3 = (LIFT ** 2 + ELBOW ** 2 - self.d ** 2) / (2 * LIFT * ELBOW)
        theta3 = (np.pi - np.arccos(cos_theta3))

        return theta3
    
    def depth_error(self):
        gripper_center = np.array([self.palm_center_pose_x,
                            self.palm_center_pose_y,
                            self.palm_center_pose_z])
        
        if self.static_obj_pose == True:
            self.static_target = np.array([self.target_x,
                self.target_y,
                self.target_z])
        else:
            pass
        
        static_error = np.array([self.static_target[0] - gripper_center[0],
                                self.static_target[1] - gripper_center[1],
                                self.static_target[2] - gripper_center[2]])

        static_dist = np.linalg.norm(static_error)

        return static_dist

    def head_cam_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        roi = frame[200:280, 280:360]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_ranges = [
            ([0, 70, 50], [10, 255, 255]),
            ([160, 70, 50], [180, 255, 255]),
        ]
        
        final_mask = None
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            if final_mask is None:
                final_mask = mask
            else:
                final_mask = cv2.bitwise_or(final_mask, mask)

        self.latest_mask = final_mask
        self.check_obj = cv2.countNonZero(final_mask) > 20

        detect_msg = String()
        detect_msg.data = str(self.check_obj)
        self.detect_pub.publish(detect_msg)

        cv2.rectangle(frame, (280, 200), (360, 280), (0, 255, 0), 3)
        cv2.imshow('head_cam', frame)
        cv2.waitKey(1)
    
    def palm_center_pose_cb(self, msg):
        self.palm_center_pose_x = msg.x
        self.palm_center_pose_y = msg.y
        self.palm_center_pose_z = msg.z

    def head_cam_depth_cb(self, msg):
        head_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        
        M = cv2.moments(self.latest_mask)
        if M['m00'] > 0:
            cX = int(M['m10'] / M['m00']) + 280
            cY = int(M['m01'] / M['m00']) + 200
            
            head_axis_x = cX - 320
            head_axis_y = cY - 240
            
            z_c = float(head_depth[cY, cX])

            if not np.isnan(z_c) and z_c > 0:
                x_c = (head_axis_x) * z_c / 415.7
                y_c = (head_axis_y) * z_c / 415.7

                value = self.transform_function(self.head_cam_joint[0], self.head_cam_joint[1], self.head_cam_joint[2], x_c, y_c, z_c)

                self.target_x = value[0]
                self.target_y = value[1]
                self.target_z = value[2]
                
                print("object position : %.2fX, %.2fY, %.2fZ" %(self.target_x, self.target_y, self.target_z))

    def joint_state_cb(self, msg):
        self.ur5e_joint = msg.position[3:9]
        self.allegro_joint = msg.position[9:]
        self.head_cam_joint = msg.position[:3]

    def main_control_loop(self):
        if self.state == RobotState.INIT:
            if self.init_count < self.init_timeout:
                init_ur5e = INIT_UR5E_POSE
                ur5e_msg = JointState()
                ur5e_msg.position = init_ur5e
                self.ur5e_joint_pub.publish(ur5e_msg)

                init_allegro = INIT_ALLEGRO_POSE
                allegro_msg = JointState()
                allegro_msg.position = init_allegro
                self.allegro_joint_pub.publish(allegro_msg)

                self.init_count += 1
                if self.init_count % 10 == 0:
                    print(f"Initializing... {self.init_count / 10:.0f} sec")
            else:
                self.get_logger().info("INIT -> SEARCH_APPLE")
                self.init_count = 0
                self.state = RobotState.SEARCH_APPLE               

        elif self.state == RobotState.SEARCH_APPLE:
            if self.check_obj:
                self.get_logger().info("APPLE CHECK -> APPROACH")
                self.state = RobotState.APPROACH

        elif self.state == RobotState.APPROACH:
            theta1 = self.shoulder_theta()
            theta3 = self.elbow_theta()
            theta2 = self.lift_theta()
            theta4 = -np.pi/2 - theta2 - theta3

            current_pan = self.ur5e_joint[0]
            pan_error = abs(theta1 - current_pan)

            ur5e_msg = JointState()
            print(self.static_obj_pose)
            self.static_obj_pose = True

            self.depth_error()

            if pan_error > 0.02:
                self.get_logger().info(f"Aligning Pan... Error: {pan_error:.4f}")
                pan_pose = list(PAN_POSE)
                pan_pose[0] = theta1
                ur5e_msg.position = pan_pose

            else:
                self.get_logger().info("Stabilization Complete! Lowering Arm...")
                stretch_pose = list(STRETCH_POSE)
                stretch_pose[0] = theta1
                stretch_pose[1] = theta2
                stretch_pose[2] = theta3
                stretch_pose[3] = theta4

                ur5e_msg.position = stretch_pose
                self.static_obj_pose = False
                self.state = RobotState.GRASP
                          
            self.ur5e_joint_pub.publish(ur5e_msg)

        elif self.state == RobotState.GRASP:
            
            allegro_msg = JointState()
            ur5e_msg = JointState()

            dist = self.depth_error()

            if dist < 0.06:
                allegro_msg.position = ALLEGRO_GRASP
                self.allegro_joint_pub.publish(allegro_msg)

                current_pan = self.ur5e_joint[0]

                back = list(BACK)
                back[0] = current_pan
                self.retract_target = back

                self.state = RobotState.RETRACT
        
        elif self.state == RobotState.RETRACT:
            ur5e_msg = JointState()
            ur5e_msg.position = self.retract_target
            self.ur5e_joint_pub.publish(ur5e_msg)

            error = 0.0
            for i in range(1, 4):
                error += abs(self.ur5e_joint[i] - self.retract_target[i])
                
            if error < 0.1:
                self.get_logger().info("MOVE TO BASKET")
                self.state = RobotState.PUT
                
        elif self.state == RobotState.PUT:
            ur5e_msg = JointState()
            ur5e_msg.position = BASKET
            self.ur5e_joint_pub.publish(ur5e_msg)
            self.count += 1

            if self.count == self.max_count:
                allegro_msg = JointState()
                allegro_msg.position = ALLEGRO_PUT
                self.allegro_joint_pub.publish(allegro_msg)
                self.count = 0
                self.state = RobotState.INIT


def main():
    rclpy.init()
    node = Ur5eControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()