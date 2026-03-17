import os
import time
import numpy as np
import mujoco
from mujoco import viewer
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge

XML_PATH = os.path.expanduser("~/sim_model/mujoco_menagerie/universal_robots_ur5e/ur5e_V1.0.xml")

class Ur5eMain(Node):
    def __init__(self, model, data):
        super().__init__('ur5e_main')

        self.model = model
        self.data = data

        self.bridge = CvBridge()

        self.head_cam_pub = self.create_publisher(Image, '/head_cam/image_raw', 10)
        self.wrist_cam_pub = self.create_publisher(Image, '/wrist_cam/image_raw', 10)
        self.head_cam_depth_pub = self.create_publisher(Image, '/head_cam/image_depth', 10)
        self.wrist_cam_depth_pub = self.create_publisher(Image, '/wrist_cam/image_depth', 10)
        self.joint_status_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.site_pose_pub = self.create_publisher(Point, '/palm_center_pose', 10)

        self.ur5e_joint_sub = self.create_subscription(JointState, '/ur5e_joint', self.ur5e_joint_cb, 10)
        self.allegro_joint_sub = self.create_subscription(JointState, '/allegro_joint', self.allegro_joint_cb, 10)
        self.head_cam_joint_sub = self.create_subscription(JointState, '/head_cam_joint', self.head_cam_joint_cb, 10)
        self.detect_sub = self.create_subscription(String, '/detect', self.detect_cb, 10)
        

    def publish_joint_states(self):

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        names = []
        positions = []

        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name
            
            qpos_addr = self.model.jnt_qposadr[i]

            if self.model.jnt_type[i] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                names.append(joint_name)
                positions.append(float(self.data.qpos[qpos_addr]))
       

        msg.name = names
        msg.position = positions
        
        self.joint_status_pub.publish(msg)
 
    def pub_frames(self, renderer):
        renderer.update_scene(self.data, camera="head_cam")
        renderer.disable_depth_rendering()
        head_rgb = renderer.render()
        head_msg = self.bridge.cv2_to_imgmsg(head_rgb, encoding="rgb8")
        self.head_cam_pub.publish(head_msg)

        renderer.enable_depth_rendering()
        head_depth_img = renderer.render()
        head_depth_msg = self.bridge.cv2_to_imgmsg(head_depth_img.astype(np.float32), encoding='32FC1')
        self.head_cam_depth_pub.publish(head_depth_msg)

    def ur5e_joint_cb(self, msg):

        if len(msg.position) >= 6:
            self.data.ctrl[:6] = msg.position[:6]
            
        else:
            self.get_logger().warn('Received JointState message with insufficient position data.')

    def allegro_joint_cb(self, msg):
         
        if len(msg.position) >= 16:
             self.data.ctrl[6:22] = msg.position[:16]

    def head_cam_joint_cb(self, msg):

        if len(msg.position) >= 3:
            self.data.ctrl[22:25] = msg.position[:3]

    def detect_cb(self, msg):
        print(msg.data)

    def publish_site_pos(self):
        pos = self.data.site('palm_center').xpos
        
        msg = Point()
        msg.x = float(pos[0])
        msg.y = float(pos[1])
        msg.z = float(pos[2])
        
        self.site_pose_pub.publish(msg)
        
def main():

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    renderer.enable_depth_rendering()

    rclpy.init()
    node = Ur5eMain(model, data)

    apple_names = ["apple_1", "apple_2", "apple_3"]
    apples_data = []

    for name in apple_names:
        weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{name}_weld")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{name}")

        if weld_id != -1 and joint_id != -1:
            apples_data.append({
                "name": name,
                "weld_id": weld_id,
                "dof_adr": model.jnt_dofadr[joint_id],
                "is_detached": False
            })

    with viewer.launch_passive(model, data) as v:
        while rclpy.ok() and v.is_running():
            rclpy.spin_once(node, timeout_sec=0)

            mujoco.mj_step(model, data)

            state_changed = False
            for apple in apples_data:
                if not apple["is_detached"]:
                    dof_adr = apple["dof_adr"]
                    
                    force_vec = data.qfrc_constraint[dof_adr : dof_adr + 3]
                    force_mag = np.linalg.norm(force_vec)
                    
                    if force_mag > 5.0:
                        data.eq_active[apple["weld_id"]] = 0
                        apple["is_detached"] = True
                        state_changed = True
                        print(f"!!! {apple['name']} detached (Force: {force_mag:.2f}N) !!!")

            if state_changed:
                mujoco.mj_forward(model, data)
            
            v.sync()
            node.pub_frames(renderer)
            node.publish_joint_states()
            node.publish_site_pos()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()