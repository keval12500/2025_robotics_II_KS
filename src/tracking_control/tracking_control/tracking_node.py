import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
def hat(k):
    khat = np.zeros((3,3))
    khat[0,1] = -k[2]
    khat[0,2] = k[1]
    khat[1,0] = k[2]
    khat[1,2] = -k[0]
    khat[2,0] = -k[1]
    khat[2,1] = k[0]
    return khat

def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2

def euler_from_quaternion(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        self.obs_pose = None
        self.goal_pose = None
        
        self.declare_parameter('world_frame_id', 'odom')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_update)
    
    def detected_obs_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        try:
            transform = self.tf_buffer.lookup_transform(odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error(f'Transform Error: {e}')
            return
        
        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        try:
            transform = self.tf_buffer.lookup_transform(odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error(f'Transform Error: {e}')
            return
        
        self.goal_pose = cp_world
    
    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            obstacle_pose = robot_world_R @ self.obs_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
            goal_pose = robot_world_R @ self.goal_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
        except TransformException as e:
            self.get_logger().error(f'Transform Error: {e}')
            return None, None
        
        return obstacle_pose, goal_pose
    
    def timer_update(self):
        if self.goal_pose is None:
            cmd_vel = Twist()
            self.pub_control_cmd.publish(cmd_vel)
            return

        current_obs_pose, current_goal_pose = self.get_current_poses()
        if current_obs_pose is None or current_goal_pose is None:
            return

        goal_distance = np.linalg.norm(current_goal_pose[:2])
        if goal_distance < 0.3:
            cmd_vel = Twist()
            self.pub_control_cmd.publish(cmd_vel)
            return

        cmd_vel = self.controller(current_obs_pose, current_goal_pose)
        self.pub_control_cmd.publish(cmd_vel)

    def controller(self, current_obs_pose, current_goal_pose):
        cmd_vel = Twist()
        safe_distance = 0.5
        linear_speed = 0.2
        angular_speed = 0.5

        goal_vector = np.array([current_goal_pose[0], current_goal_pose[1]])
        if current_obs_pose is not None:
            obs_vector = np.array([current_obs_pose[0], current_obs_pose[1]])
            obs_distance = np.linalg.norm(obs_vector)
        else:
            obs_distance = float('inf')

        if obs_distance < safe_distance:
            angle_to_obstacle = math.atan2(obs_vector[1], obs_vector[0])
            cmd_vel.angular.z = angular_speed if angle_to_obstacle < 0 else -angular_speed
        else:
            angle_to_goal = math.atan2(goal_vector[1], goal_vector[0])
            cmd_vel.linear.x = linear_speed
            cmd_vel.angular.z = angle_to_goal * 0.5

        return cmd_vel

def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
