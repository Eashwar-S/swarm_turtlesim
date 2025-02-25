#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen
from turtlesim.msg import Pose
import math

class TurtlePatternDrawer(Node):
    def __init__(self):
        super().__init__('turtle_pattern_drawer')
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Subscriber
        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        

        self.pen_client = self.create_client(SetPen, '/turtle1/set_pen')

        while not self.pen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /turtle1/set_pen not available, waiting...')


        # Store current pose
        self.current_pose = None
        
        # Waypoints for SWARM ROBOTICS (as defined above)
        # self.waypoints = [
        #     [(0.5, 8), (1.4, 8), (0.5, 7), (1.4, 6), (0.5, 6)],  # S
        #     [(2.3, 8), (2.75, 6), (3.1, 8), (3.55, 6), (4.1, 8)],  # W
        #     [(4.9, 6), (5.8, 8), (6.7, 6), (6.25, 7), (5.35, 7)],  # A
        #     [(7.5, 6), (7.5, 8), (8.4, 8), (8.4, 7), (7.5, 7), (8.4, 6)],  # R
        #     [(9.1, 6), (9.1, 8), (9.9, 7), (10.7, 8), (9.9, 6), (10.7, 6)],  # M
        #     [(0.5, 3), (0.5, 5), (1.1, 5), (1.1, 3), (0.5, 3)],  # R
        #     [(1.7, 3), (1.7, 5), (2.3, 5), (2.3, 3), (1.7, 3)],  # O
        #     [(2.9, 3), (2.9, 5), (2.3, 5), (3.5, 5), (3.5, 3)],  # B
        #     [(4.1, 3), (4.1, 5), (4.7, 5), (4.7, 3), (4.1, 3)],  # O
        #     [(5.3, 3), (5.3, 5), (4.7, 5), (5.9, 5), (5.9, 3)],  # T
        #     [(6.5, 3), (6.5, 5)],  # I
        #     [(6.7, 5), (6.1, 5), (6.1, 3), (6.7, 3)],  # C
        #     [(7.3, 3), (7.3, 5), (7.9, 5), (7.9, 3), (7.3, 3)],  # S
        #     [(8.5, 3), (8.5, 5), (9.1, 5), (9.1, 3), (8.5, 3)]   # S
        # ]
        
        self.waypoints = [[(2.4, 8), (1.5, 8), (1.5, 7), (2.4, 7), (2.4, 6), (1.5, 6)],
                          [(2.8, 8), (3.25, 6), (3.6999999999999997, 8), (4.15, 6), (4.6, 8)],
                          [(4.6, 6), (5.3, 8), (6.0, 6), (5.6499999999999995, 7), (4.949999999999999, 7)],
                          [(6.4, 6), (6.4, 8), (7.300000000000001, 8), (7.300000000000001, 7), (6.4, 7), (7.300000000000001, 6)],
                          [(8.2, 6), (8.2, 8), (9.1, 7), (10.0, 8), (10.0, 6)],
                          [(1.0, 3), (1.0, 5), (1.9, 5), (1.9, 4), (1.0, 4), (1.9, 3)],
                          [(2.2, 3), (2.2, 5), (3.2, 5), (3.2, 3), (2.2, 3)],
                          [(3.4, 3), (3.4, 5), (4.4, 5), (4.4, 4), (3.4, 4), (4.4, 4), (4.4, 3), (3.4, 3)],
                          [(4.6, 3), (4.6, 5), (5.6, 5), (5.6, 3), (4.6, 3)],
                          [(6.3999999999999995, 3), (6.3999999999999995, 5), (5.8, 5), (7.0, 5)],
                          [(7.6, 3), (7.6, 5)],
                          [(8.799999999999999, 5), (8.2, 5), (8.2, 3), (8.799999999999999, 3)],
                          [(10.3, 5), (9.4, 5), (9.4, 4), (10.3, 4), (10.3, 3), (9.4, 3)]]
        # State variables
        self.current_letter_idx = 0  # Index of the current letter
        self.current_waypoint_idx = 0  # Index within the current letter's waypoints
        self.is_drawing = False  # Track if pen is drawing
        self.pen_down = False  # Track pen state
        self.waypoint_distance_threshold = 0.15 # Threshold to determine a waypoint is reached
        self.angular_velocity_set = False
        # Background color (blue: RGB = 0, 0, 255)
        self.background_r = 0
        self.background_g = 0
        self.background_b = 255
        
        # Start drawing after receiving initial pose
        self.timer = self.create_timer(0.1, self.draw_pattern)  # Check every 0.1 seconds

    def pose_callback(self, msg):
        """Callback for /turtle1/pose to track turtle position."""
        self.current_pose = msg
        # Check if turtle reached the next waypoint
        if self.current_pose and self.is_drawing:
            target_x, target_y = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
            self.get_logger().info(f'target_x - {target_x}, target_y - {target_y}')
            distance = math.sqrt((self.current_pose.x - target_x)**2 + (self.current_pose.y - target_y)**2)
            self.get_logger().info(f'distance - {distance}')
            if distance < self.waypoint_distance_threshold:  # Threshold for reaching waypoint
                self.current_waypoint_idx += 1
                self.angular_velocity_set = False
                if self.current_waypoint_idx >= len(self.waypoints[self.current_letter_idx]):
                    self.current_waypoint_idx = 0
                    self.current_letter_idx += 1
                    self.is_drawing = False  # Stop drawing after finishing a letter
                    if self.current_letter_idx >= len(self.waypoints):
                        self.get_logger().info("Pattern completed!")
                        self.destroy_timer(self.timer)  # Stop the timer
                        return
                self.is_drawing = True  # Start drawing the next segment

    def calculate_velocities(self, target_x, target_y):
        """Calculate linear and angular velocities to move to target."""
        if not self.current_pose:
            return None, None
        
        linear_vel = 0.0 
        angular_vel = 0.0
        # Current position and orientation
        current_x = self.current_pose.x
        current_y = self.current_pose.y
        current_theta = self.current_pose.theta
        
        # Target position
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Distance to target
        distance = math.sqrt(dx**2 + dy**2)
        
        # Desired angle to target
        desired_theta = math.atan(dy, dx)
        
        # Angular velocity (turn towards target)
        # if not self.angular_velocity_set:
        angular_vel = 0.1 * (desired_theta - current_theta)
        
        # if abs(angular_vel) < 0.03:
            # self.angular_velocity_set = True
        # if angular_vel > math.pi:
        #     angular_vel -= 2 * math.pi
        # elif angular_vel < -math.pi:
        #     angular_vel += 2 * math.pi
        
        # Linear velocity (move forward)
        # if self.angular_velocity_set:
        linear_vel = 0.1 * distance  if abs(angular_vel) < 0.1 else 0.0  # Only move forward if nearly aligned
        self.get_logger().info(f'linear velocity - {linear_vel}, angualar velocity - {angular_vel}')
        return linear_vel, angular_vel

    def set_pen(self, r, g, b, width=2, off=False):
        """Set pen color and state via /turtle1/set_pen."""
        request = SetPen.Request()
        request.r = r
        request.g = g
        request.b = b
        request.width = width
        request.off = off

        
        future = self.pen_client.call_async(request)
        # future.add_done_callback(self.pen_callback)
        return future

    # def pen_callback(self, future):
    #     try:

    #         response = future.result()
    #         self.get_logger().info(f'Pen settings applied successfully')
    #     except Exception as e:
    #         self.get_logger().error(f'Service call failed: {e}')
        

    def draw_pattern(self):
        """Main logic to draw the pattern letter by letter."""
        if not self.current_pose or self.current_letter_idx >= len(self.waypoints):
            return
        
        # Determine if we should draw (pen down) or move (pen up)
        target_x, target_y = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
        
        if self.is_drawing:
            # Drawing phase: Set pen to white (255, 255, 255)
            self.set_pen(255, 255, 255) # White while drawing
        else:
            # Moving phase: Set pen to match background (blue: 0, 0, 255)
             self.set_pen(self.background_r, self.background_g, self.background_b, off=True)
        
        # Calculate and publish velocities to move to target
        linear_vel, angular_vel = self.calculate_velocities(target_x, target_y)
        if linear_vel is not None and angular_vel is not None:
            twist_msg = Twist()
            twist_msg.linear.x = linear_vel
            twist_msg.angular.z = angular_vel
            self.cmd_vel_pub.publish(twist_msg)
        
        # Start drawing if not already drawing and at the start of a letter
        if not self.is_drawing and self.current_waypoint_idx == 0:
            distance = math.sqrt((self.current_pose.x - target_x)**2 + (self.current_pose.y - target_y)**2)
            self.get_logger().info(f'target_x - {target_x}, target_y - {target_y}')
            self.get_logger().info(f'current_x - {self.current_pose.x}, current_y - {self.current_pose.y}')
            self.get_logger().info(f'distance - {distance}')
            if distance < self.waypoint_distance_threshold:
                self.is_drawing = True

def main(args=None):
    rclpy.init(args=args)
    node = TurtlePatternDrawer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()