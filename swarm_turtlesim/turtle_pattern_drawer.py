# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from turtlesim.srv import SetPen
# from turtlesim.msg import Pose
# import math
# import time

# class TurtlePatternDrawer(Node):
#     def __init__(self):
#         super().__init__('turtle_pattern_drawer')
        
#         self.start_time = time.time()
#         # Publisher
#         self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
#         # Subscriber
#         self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        

#         self.pen_client = self.create_client(SetPen, '/turtle1/set_pen')

#         while not self.pen_client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('Service /turtle1/set_pen not available, waiting...')

#         # Store current pose
#         self.current_pose = None
        
#         # Waypoints for SWARM ROBOTICS (as defined above)
#         # self.waypoints = [
#         #     [(0.5, 8), (1.4, 8), (0.5, 7), (1.4, 6), (0.5, 6)],  # S
#         #     [(2.3, 8), (2.75, 6), (3.1, 8), (3.55, 6), (4.1, 8)],  # W
#         #     [(4.9, 6), (5.8, 8), (6.7, 6), (6.25, 7), (5.35, 7)],  # A
#         #     [(7.5, 6), (7.5, 8), (8.4, 8), (8.4, 7), (7.5, 7), (8.4, 6)],  # R
#         #     [(9.1, 6), (9.1, 8), (9.9, 7), (10.7, 8), (9.9, 6), (10.7, 6)],  # M
#         #     [(0.5, 3), (0.5, 5), (1.1, 5), (1.1, 3), (0.5, 3)],  # R
#         #     [(1.7, 3), (1.7, 5), (2.3, 5), (2.3, 3), (1.7, 3)],  # O
#         #     [(2.9, 3), (2.9, 5), (2.3, 5), (3.5, 5), (3.5, 3)],  # B
#         #     [(4.1, 3), (4.1, 5), (4.7, 5), (4.7, 3), (4.1, 3)],  # O
#         #     [(5.3, 3), (5.3, 5), (4.7, 5), (5.9, 5), (5.9, 3)],  # T
#         #     [(6.5, 3), (6.5, 5)],  # I
#         #     [(6.7, 5), (6.1, 5), (6.1, 3), (6.7, 3)],  # C
#         #     [(7.3, 3), (7.3, 5), (7.9, 5), (7.9, 3), (7.3, 3)],  # S
#         #     [(8.5, 3), (8.5, 5), (9.1, 5), (9.1, 3), (8.5, 3)]   # S
#         # ]
        
        # self.waypoints = [[(2.4, 8), (1.5, 8), (1.5, 7), (2.4, 7), (2.4, 6), (1.5, 6)],
        #                   [(2.8, 8), (3.25, 6), (3.6999999999999997, 8), (4.15, 6), (4.6, 8)],
        #                   [(4.6, 6), (5.3, 8), (6.0, 6), (5.6499999999999995, 7), (4.949999999999999, 7)],
        #                   [(6.4, 6), (6.4, 8), (7.300000000000001, 8), (7.300000000000001, 7), (6.4, 7), (7.300000000000001, 6)],
        #                   [(8.2, 6), (8.2, 8), (9.1, 7), (10.0, 8), (10.0, 6)],
        #                   [(1.0, 3), (1.0, 5), (1.9, 5), (1.9, 4), (1.0, 4), (1.9, 3)],
        #                   [(2.2, 3), (2.2, 5), (3.2, 5), (3.2, 3), (2.2, 3)],
        #                   [(3.4, 3), (3.4, 5), (4.4, 5), (4.4, 4), (3.4, 4), (4.4, 4), (4.4, 3), (3.4, 3)],
        #                   [(4.6, 3), (4.6, 5), (5.6, 5), (5.6, 3), (4.6, 3)],
        #                   [(6.3999999999999995, 3), (6.3999999999999995, 5), (5.8, 5), (7.0, 5)],
        #                   [(7.6, 3), (7.6, 5)],
        #                   [(8.799999999999999, 5), (8.2, 5), (8.2, 3), (8.799999999999999, 3)],
        #                   [(10.3, 5), (9.4, 5), (9.4, 4), (10.3, 4), (10.3, 3), (9.4, 3)]]
#         # State variables
#         self.current_letter_idx = 0  # Index of the current letter
#         self.current_waypoint_idx = 0  # Index within the current letter's waypoints
#         self.is_drawing = False  # Track if pen is drawing
#         self.pen_down = False  # Track pen state
#         self.waypoint_distance_threshold = 0.05 # Threshold to determine a waypoint is reached
#         self.angular_velocity_set = False
#         # Background color (blue: RGB = 0, 0, 255)
#         self.background_r = 0
#         self.background_g = 0
#         self.background_b = 255
        
#         # Start drawing after receiving initial pose
#         self.timer = self.create_timer(0.1, self.draw_pattern)  # Check every 0.1 seconds

#     def pose_callback(self, msg):
#         """Callback for /turtle1/pose to track turtle position."""
#         self.current_pose = msg
#         # Check if turtle reached the next waypoint
#         if self.current_pose and self.current_letter_idx < len(self.waypoints):
#             target_x, target_y = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
#             self.get_logger().info(f'target_x - {target_x}, target_y - {target_y}')
#             distance = math.sqrt((self.current_pose.x - target_x)**2 + (self.current_pose.y - target_y)**2)
#             self.get_logger().info(f'distance - {distance}')

#             if self.current_waypoint_idx  == 0:
#                 self.is_drawing = False
#             else:
#                 self.is_drawing = True  # Start drawing the next segment

#             if distance < self.waypoint_distance_threshold:  # Threshold for reaching waypoint
#                 self.current_waypoint_idx += 1
#                 if self.current_waypoint_idx >= len(self.waypoints[self.current_letter_idx]):
#                     self.current_waypoint_idx = 0
#                     self.current_letter_idx += 1
#                     self.is_drawing = False  # Stop drawing after finishing a letter
#                     if self.current_letter_idx >= len(self.waypoints):
#                         self.get_logger().info("Pattern completed!")
#                         end = time.time()
#                         self.get_logger().info(f'Total time to draw the pattern - {end - self.start_time} seconds')
#                         self.destroy_timer(self.timer)  # Stop the timer
#                         return
                

#     def calculate_velocities(self, target_x, target_y):
#         """Calculate linear and angular velocities to move to target."""
#         if not self.current_pose:
#             return None, None
        
#         linear_vel = 0.0 
#         angular_vel = 0.0
#         # Current position and orientation
#         current_x = self.current_pose.x
#         current_y = self.current_pose.y
#         current_theta = self.current_pose.theta
        
#         # Target position
#         dx = target_x - current_x
#         dy = target_y - current_y
        
#         # Distance to target
#         distance = math.sqrt(dx**2 + dy**2)
        
#         # Desired angle to target
#         desired_theta = math.atan2(dy, dx)
        
#         # Angular velocity (turn towards target)
#         # if not self.angular_velocity_set:
#         angle_error = desired_theta - current_theta
#         angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

#         # Proportional gains (tune to your needs)
#         Kp_angular = 0.5
#         Kp_linear = 0.2
    
#         # Linear velocity (move forward)
#         # if self.angular_velocity_set:

#         # Calculate angular velocity
#         angular_vel = Kp_angular * angle_error

#         linear_vel = Kp_linear * distance  if abs(angle_error) < 0.1 else 0.0  # Only move forward if nearly aligned
#         self.get_logger().info(f'linear velocity - {linear_vel}, angualar velocity - {angular_vel}, angle error - {angle_error}')
#         return linear_vel, angular_vel

#     def set_pen(self, r, g, b, width=2, off=False):
#         """Set pen color and state via /turtle1/set_pen."""
#         request = SetPen.Request()
#         request.r = r
#         request.g = g
#         request.b = b
#         request.width = width
#         request.off = off

        
#         future = self.pen_client.call_async(request)
#         # future.add_done_callback(self.pen_callback)
#         return future

#     # def pen_callback(self, future):
#     #     try:

#     #         response = future.result()
#     #         self.get_logger().info(f'Pen settings applied successfully')
#     #     except Exception as e:
#     #         self.get_logger().error(f'Service call failed: {e}')
        

#     def draw_pattern(self):
#         """Main logic to draw the pattern letter by letter."""
#         if not self.current_pose or self.current_letter_idx >= len(self.waypoints):
#             return
        
#         # Determine if we should draw (pen down) or move (pen up)
#         target_x, target_y = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
        
#         if self.is_drawing:
#             # Drawing phase: Set pen to white (255, 255, 255)
#             self.set_pen(255, 255, 255) # White while drawing
#         else:
#             # Moving phase: Set pen to match background (blue: 0, 0, 255)
#              self.set_pen(self.background_r, self.background_g, self.background_b, off=True)
        
#         # Calculate and publish velocities to move to target
#         linear_vel, angular_vel = self.calculate_velocities(target_x, target_y)
#         if linear_vel is not None and angular_vel is not None:
#             twist_msg = Twist()
#             twist_msg.linear.x = linear_vel
#             twist_msg.angular.z = angular_vel
#             self.cmd_vel_pub.publish(twist_msg)
        
#         # Start drawing if not already drawing and at the start of a letter
#         # if not self.is_drawing and self.current_waypoint_idx == 0:
#         #     distance = math.sqrt((self.current_pose.x - target_x)**2 + (self.current_pose.y - target_y)**2)
#         #     self.get_logger().info(f'target_x - {target_x}, target_y - {target_y}')
#         #     self.get_logger().info(f'current_x - {self.current_pose.x}, current_y - {self.current_pose.y}')
#         #     self.get_logger().info(f'distance - {distance}')
#         #     if distance < self.waypoint_distance_threshold:
#         #         self.is_drawing = True

# def main(args=None):
    
#     rclpy.init(args=args)
#     node = TurtlePatternDrawer()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3

# import sys
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from turtlesim.srv import SetPen, Spawn
# from turtlesim.msg import Pose
# import math
# import time

# class TurtlePatternDrawer(Node):
#     def __init__(self, waypoints, turtle_name='turtle1'):
#         super().__init__(f'turtle_pattern_drawer_{turtle_name}')
#         self.turtle_name = turtle_name
        
#         self.start_time = time.time()
#         # Publisher for cmd_vel specific to this turtle
#         self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.turtle_name}/cmd_vel', 10)
        
#         # Subscriber for pose specific to this turtle
#         self.pose_sub = self.create_subscription(Pose, f'/{self.turtle_name}/pose', self.pose_callback, 10)
        
#         # Client for the set_pen service specific to this turtle
#         self.pen_client = self.create_client(SetPen, f'/{self.turtle_name}/set_pen')
#         while not self.pen_client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info(f'Service /{self.turtle_name}/set_pen not available, waiting...')
        
#         # Store current pose
#         self.current_pose = None
        
#         # Waypoints for pattern drawing (this is an example)
#         self.waypoints = waypoints
#         # self.waypoints = [
#         #     [(2.4, 8), (1.5, 8), (1.5, 7), (2.4, 7), (2.4, 6), (1.5, 6)],
#         #     [(2.8, 8), (3.25, 6), (3.7, 8), (4.15, 6), (4.6, 8)]
#         # ]
        
#         # State variables for drawing
#         self.current_letter_idx = 0  # Index of the current letter
#         self.current_waypoint_idx = 0  # Index within the current letter's waypoints
#         self.is_drawing = False  # Track if pen is drawing
#         self.pen_down = False  # Track pen state
#         self.waypoint_distance_threshold = 0.05 # Threshold to determine a waypoint is reached
        
#         # Background color (blue: RGB = 0, 0, 255)
#         self.background_r = 0
#         self.background_g = 0
#         self.background_b = 255
        
#         # Start drawing after receiving initial pose
#         self.timer = self.create_timer(0.1, self.draw_pattern)  # Check every 0.1 seconds

#     def pose_callback(self, msg):
#         """Callback for pose to track turtle position."""
#         self.current_pose = msg
#         if self.current_pose and self.current_letter_idx < len(self.waypoints):
#             target_x, target_y = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
#             distance = math.sqrt((self.current_pose.x - target_x)**2 + (self.current_pose.y - target_y)**2)
#             self.get_logger().info(f'[{self.turtle_name}] target: ({target_x}, {target_y}) | current: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f}) | distance: {distance:.2f}')

#             # Decide whether to draw (pen down) or move (pen up)
#             if self.current_waypoint_idx == 0:
#                 self.is_drawing = False
#             else:
#                 self.is_drawing = True

#             if distance < self.waypoint_distance_threshold:  # Reached waypoint
#                 self.current_waypoint_idx += 1
#                 if self.current_waypoint_idx >= len(self.waypoints[self.current_letter_idx]):
#                     self.current_waypoint_idx = 0
#                     self.current_letter_idx += 1
#                     self.is_drawing = False
#                     if self.current_letter_idx >= len(self.waypoints):
#                         self.get_logger().info(f"[{self.turtle_name}] Pattern completed in {time.time() - self.start_time:.2f} seconds!")
#                         self.destroy_timer(self.timer)
#                         return

#     def calculate_velocities(self, target_x, target_y):
#         """Calculate linear and angular velocities to move to target."""
#         if not self.current_pose:
#             return 0.0, 0.0

#         current_x = self.current_pose.x
#         current_y = self.current_pose.y
#         current_theta = self.current_pose.theta

#         dx = target_x - current_x
#         dy = target_y - current_y

#         distance = math.sqrt(dx**2 + dy**2)
#         desired_theta = math.atan2(dy, dx)
#         angle_error = desired_theta - current_theta
#         angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

#         # Here you could use dynamic gains as needed.
#         Kp_angular = 0.5
#         Kp_linear = 0.2

#         angular_vel = Kp_angular * angle_error
#         linear_vel = Kp_linear * distance if abs(angle_error) < 0.1 else 0.0
#         self.get_logger().info(f"[{self.turtle_name}] linear_vel: {linear_vel:.2f}, angular_vel: {angular_vel:.2f}")
#         return linear_vel, angular_vel

#     def set_pen(self, r, g, b, width=2, off=False):
#         """Set pen color and state."""
#         request = SetPen.Request()
#         request.r = r
#         request.g = g
#         request.b = b
#         request.width = width
#         request.off = off
#         future = self.pen_client.call_async(request)
#         return future

#     def draw_pattern(self):
#         """Main logic to draw the pattern letter by letter."""
#         if not self.current_pose or self.current_letter_idx >= len(self.waypoints):
#             return
        
#         target_x, target_y = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
        
#         if self.is_drawing:
#             self.set_pen(255, 255, 255)  # White pen when drawing
#         else:
#             self.set_pen(self.background_r, self.background_g, self.background_b, off=True)
        
#         linear_vel, angular_vel = self.calculate_velocities(target_x, target_y)
#         twist_msg = Twist()
#         twist_msg.linear.x = linear_vel
#         twist_msg.angular.z = angular_vel
#         self.cmd_vel_pub.publish(twist_msg)

# def spawn_turtles(num_turtles):
#     """
#     Spawns additional turtles in turtlesim (if num_turtles > 1).
#     turtle1 exists by default.
#     """
#     # Create a temporary node to call the spawn service
#     node = rclpy.create_node('turtle_spawner')
#     spawn_client = node.create_client(Spawn, 'spawn')
#     while not spawn_client.wait_for_service(timeout_sec=1.0):
#         node.get_logger().info('Spawn service not available, waiting...')
    
#     # Spawn turtles from turtle2 up to turtleN
#     for i in range(2, num_turtles + 1):
#         request = Spawn.Request()
#         # You can choose spawn positions arbitrarily; here we space them out
#         request.x = 2.0 * i
#         request.y = 2.0
#         request.theta = 0.0
#         request.name = f"turtle{i}"
#         future = spawn_client.call_async(request)
#         rclpy.spin_until_future_complete(node, future)
#         if future.result() is not None:
#             node.get_logger().info(f"Spawned turtle: {future.result().name}")
#         else:
#             node.get_logger().error("Failed to spawn turtle")
#     node.destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     # Get number of turtles from command-line arguments (default: 1)
#     num_turtles = 1
#     waypoints = [[(2.4, 8), (1.5, 8), (1.5, 7), (2.4, 7), (2.4, 6), (1.5, 6)],
#                           [(2.8, 8), (3.25, 6), (3.6999999999999997, 8), (4.15, 6), (4.6, 8)],
#                           [(4.6, 6), (5.3, 8), (6.0, 6), (5.6499999999999995, 7), (4.949999999999999, 7)],
#                           [(6.4, 6), (6.4, 8), (7.300000000000001, 8), (7.300000000000001, 7), (6.4, 7), (7.300000000000001, 6)],
#                           [(8.2, 6), (8.2, 8), (9.1, 7), (10.0, 8), (10.0, 6)],
#                           [(1.0, 3), (1.0, 5), (1.9, 5), (1.9, 4), (1.0, 4), (1.9, 3)],
#                           [(2.2, 3), (2.2, 5), (3.2, 5), (3.2, 3), (2.2, 3)],
#                           [(3.4, 3), (3.4, 5), (4.4, 5), (4.4, 4), (3.4, 4), (4.4, 4), (4.4, 3), (3.4, 3)],
#                           [(4.6, 3), (4.6, 5), (5.6, 5), (5.6, 3), (4.6, 3)],
#                           [(6.3999999999999995, 3), (6.3999999999999995, 5), (5.8, 5), (7.0, 5)],
#                           [(7.6, 3), (7.6, 5)],
#                           [(8.799999999999999, 5), (8.2, 5), (8.2, 3), (8.799999999999999, 3)],
#                           [(10.3, 5), (9.4, 5), (9.4, 4), (10.3, 4), (10.3, 3), (9.4, 3)]]
#     if len(sys.argv) > 1:
#         try:
#             num_turtles = int(sys.argv[1])
#         except ValueError:
#             print("Invalid num_turtles argument. Using default value 1.")
#             num_turtles = 1

#     # Spawn additional turtles if needed (turtle1 is already present)
#     if num_turtles > 1:
#         spawn_turtles(num_turtles)

#     # For demonstration, we create a pattern drawer for each turtle.
#     # In a more complex system you might run them in separate threads or nodes.
#     drawers = []
#     for i in range(1, num_turtles + 1):
#         turtle_name = f"turtle{i}"
#         # Assign waypoints based on auction algorithm
#         node = TurtlePatternDrawer(waypoints=waypoints[0] ,turtle_name=turtle_name)
#         drawers.append(node)

#     # Use a multi-threaded executor to spin all the nodes concurrently.
#     executor = rclpy.executors.MultiThreadedExecutor()
#     for node in drawers:
#         executor.add_node(node)

#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass

#     # Clean up
#     for node in drawers:
#         node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3

import sys
import math
import time
import random
from typing import List, Dict, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen, Spawn
from turtlesim.msg import Pose

############################
# Auction Classes / Logic #
############################

class Task:
    """
    Represents a single 'letter' in SWARM ROBOTICS,
    storing:
      - letter_id (int)
      - waypoints (List[Tuple[x,y]]) for that letter
      - precomputed total_path_length (float)
      - reference_x, reference_y (float) as some representative point (e.g. first waypoint)
    Auction states:
      - assigned_robot (None or robot_id)
      - price (float)
    """
    def __init__(self, letter_id, waypoints):
        self.id = letter_id
        self.waypoints = waypoints
        self.price = 0.0
        self.assigned_robot = None

        # We can pick the first waypoint for cost reference
        self.reference_x, self.reference_y = waypoints[0]

        # Precompute total path length of this letter
        dist = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            dist += math.sqrt(dx*dx + dy*dy)
        self.path_length = dist

class RobotInfo:
    """
    Represents the robot for the auction.  (not the actual turtle node)
    """
    def __init__(self, robot_id, init_x, init_y):
        self.id = robot_id
        self.x = init_x
        self.y = init_y
        self.assigned_tasks = []

    def cost_to_do_task(self, task: Task) -> float:
        """
        Cost = distance from (robot.x, robot.y) to letter's reference point + path_length
        """
        dx = task.reference_x - self.x
        dy = task.reference_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        return dist + task.path_length

def auction_allocation(robots: List[RobotInfo], tasks: List[Task],
                       epsilon=0.01, max_rounds=200):
    """
    Synchronous Auction Algorithm for tasks -> robots.
    1) Each robot calculates cost + price for all tasks,
       picks best and 2nd-best, places a bid.
    2) For each task, pick the highest bid -> assign to that robot, update price.
    3) Repeat until tasks are assigned or max_rounds is reached.
    """
    unassigned = set(t.id for t in tasks)
    task_dict = {t.id : t for t in tasks}

    for t in tasks:
        t.assigned_robot = None
        t.price = 0.0

    for r in robots:
        r.assigned_tasks.clear()

    round_num = 0
    while unassigned:
        round_num += 1
        robot_bids = []  # (robot_id, task_id, bid_value)

        # Step 1: Each robot picks best task, second best
        for robot in robots:
            best_task = None
            best_score = float('inf')
            second_best_score = float('inf')

            # Evaluate cost + price for each task
            for task in tasks:
                cost = robot.cost_to_do_task(task)
                score = cost + task.price
                if score < best_score:
                    second_best_score = best_score
                    best_score = score
                    best_task = task
                elif score < second_best_score:
                    second_best_score = score

            if best_task is not None:
                increment = (second_best_score - best_score) + epsilon
                new_bid = best_task.price + increment
                robot_bids.append((robot.id, best_task.id, new_bid))

        if not robot_bids:
            break

        # Step 2: Group bids by task, pick highest
        bids_by_task = {}
        for (r_id, t_id, bid_val) in robot_bids:
            if t_id not in bids_by_task:
                bids_by_task[t_id] = []
            bids_by_task[t_id].append((r_id, bid_val))

        changed = False
        for t_id, bid_list in bids_by_task.items():
            best_bid_val = task_dict[t_id].price
            best_bidder = None
            for (r_id, b_val) in bid_list:
                if b_val > best_bid_val:
                    best_bid_val = b_val
                    best_bidder = r_id
            if best_bidder is not None:
                # Reassign
                task_dict[t_id].assigned_robot = best_bidder
                task_dict[t_id].price = best_bid_val
                changed = True

        # Remove assigned tasks from unassigned
        for t in tasks:
            if t.assigned_robot is not None and t.id in unassigned:
                unassigned.remove(t.id)

        if not changed:
            # stable
            break

    # At end, fill assigned_tasks in RobotInfo
    for t in tasks:
        if t.assigned_robot is not None:
            for r in robots:
                if r.id == t.assigned_robot:
                    r.assigned_tasks.append(t)
                    break

    return tasks

##############################
# The Turtle Drawing Node(s) #
##############################

class TurtlePatternDrawer(Node):
    def __init__(self, turtle_name: str, waypoints: List[List[Tuple[float,float]]]):
        """
        :param waypoints: a list of letters, each letter is a list of (x,y).
        """
        super().__init__(f'turtle_pattern_drawer_{turtle_name}')
        self.turtle_name = turtle_name
        self.waypoints = waypoints

        self.start_time = time.time()

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.turtle_name}/cmd_vel', 10)

        # Subscriber
        self.pose_sub = self.create_subscription(Pose, f'/{self.turtle_name}/pose', self.pose_callback, 10)

        # Pen service
        self.pen_client = self.create_client(SetPen, f'/{self.turtle_name}/set_pen')
        while not self.pen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Service /{self.turtle_name}/set_pen not available, waiting...')

        # Current pose
        self.current_pose = None

        # State for drawing
        self.current_letter_idx = 0
        self.current_waypoint_idx = 0
        self.is_drawing = False
        self.waypoint_distance_threshold = 0.05

        # Timer
        self.timer = self.create_timer(0.1, self.draw_pattern)

        # Pen / background color
        self.bg_r = 0
        self.bg_g = 0
        self.bg_b = 255

    def pose_callback(self, msg: Pose):
        self.current_pose = msg

        # Check if reached current waypoint
        if (self.current_pose and
            self.current_letter_idx < len(self.waypoints)):
            tx, ty = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
            dist = math.sqrt((self.current_pose.x - tx)**2 + (self.current_pose.y - ty)**2)
            if dist < self.waypoint_distance_threshold:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints[self.current_letter_idx]):
                    self.current_waypoint_idx = 0
                    self.current_letter_idx += 1
                    self.is_drawing = False
                    if self.current_letter_idx >= len(self.waypoints):
                        self.get_logger().info(f"[{self.turtle_name}] Completed pattern in {time.time() - self.start_time:.2f} sec.")
                        self.destroy_timer(self.timer)

    def set_pen(self, r, g, b, width=2, off=False):
        request = SetPen.Request()
        request.r = r
        request.g = g
        request.b = b
        request.width = width
        request.off = off
        future = self.pen_client.call_async(request)
        return future

    def calculate_velocities(self, tx, ty) -> Tuple[float, float]:
        if not self.current_pose:
            return 0.0, 0.0
        dx = tx - self.current_pose.x
        dy = ty - self.current_pose.y
        dist = math.sqrt(dx**2 + dy**2)
        desired_theta = math.atan2(dy, dx)
        angle_error = desired_theta - self.current_pose.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        Kp_lin = 0.2
        Kp_ang = 0.5
        angular_z = Kp_ang * angle_error
        linear_x = Kp_lin * dist if abs(angle_error) < 0.1 else 0.0
        return linear_x, angular_z

    def draw_pattern(self):
        # Nothing to do if done or no pose
        if (not self.current_pose or
            self.current_letter_idx >= len(self.waypoints)):
            return
        # Are we drawing or moving?
        self.is_drawing = (self.current_waypoint_idx != 0)

        if self.is_drawing:
            self.set_pen(255, 255, 255, off=False)  # pen down
        else:
            self.set_pen(self.bg_r, self.bg_g, self.bg_b, off=True)  # pen up

        tx, ty = self.waypoints[self.current_letter_idx][self.current_waypoint_idx]
        lin, ang = self.calculate_velocities(tx, ty)
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.cmd_vel_pub.publish(twist)

#######################
# Main Program (ROS2) #
#######################

def spawn_turtles(num_robots: int):
    """
    If user wants multiple robots in Turtlesim,
    spawn from turtle2.. up to turtleN
    """
    if num_robots <= 1:
        return
    node = rclpy.create_node('spawn_client')
    cli = node.create_client(Spawn, 'spawn')
    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info("Waiting for Spawn service...")
    for i in range(2, num_robots + 1):
        req = Spawn.Request()
        # place them at random
        req.x = float(random.randint(1, 10))
        req.y = float(random.randint(1, 8))
        req.theta = 0.0
        req.name = f"turtle{i}"
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        if future.result():
            node.get_logger().info(f"Spawned {future.result().name}")
        else:
            node.get_logger().error(f"Failed to spawn turtle{i}")
    node.destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # 1) Setup letter tasks, each letter is a list of waypoints
    # For demonstration, we use the entire set of SWARM ROBOTICS letters
    # as a list-of-lists. We'll treat each letter as a single 'Task'.
    # (We've truncated to 5 letters for brevity, but you can use all.)
    full_waypoints = [
        [(2.4, 8), (1.5, 8), (1.5, 7), (2.4, 7), (2.4, 6), (1.5, 6)],  # S
        [(2.8, 8), (3.25, 6), (3.7, 8), (4.15, 6), (4.6, 8)],          # W
        [(4.6, 6), (5.3, 8), (6.0, 6), (5.65, 7), (4.95, 7)],          # A
        [(6.4, 6), (6.4, 8), (7.3, 8), (7.3, 7), (6.4, 7), (7.3, 6)],  # R
        [(8.2, 6), (8.2, 8), (9.1, 7), (10.0, 8), (10.0, 6)],          # M
        [(1.0, 3), (1.0, 5), (1.9, 5), (1.9, 4), (1.0, 4), (1.9, 3)],  # R
        [(2.2, 3), (2.2, 5), (3.2, 5), (3.2, 3), (2.2, 3)],            # O
        [(3.4, 3), (3.4, 5), (4.4, 5), (4.4, 4), (3.4, 4), (4.4, 4), (4.4, 3), (3.4, 3)],  # B
        [(4.6, 3), (4.6, 5), (5.6, 5), (5.6, 3), (4.6, 3)],            # O
        [(6.4, 3), (6.4, 5), (5.8, 5), (7.0, 5)],                      # T
        [(7.6, 3), (7.6, 5)],                                         # I
        [(8.8, 5), (8.2, 5), (8.2, 3), (8.8, 3)],                      # C
        [(10.3, 5), (9.4, 5), (9.4, 4), (10.3, 4), (10.3, 3), (9.4, 3)], # S
    ]

    tasks = []
    for i, wpts in enumerate(full_waypoints):
        tasks.append(Task(letter_id=i, waypoints=wpts))

    # 2) Determine number of robots from cmd line
    num_robots = 1
    if len(sys.argv) > 1:
        try:
            num_robots = int(sys.argv[1])
        except:
            print("Invalid argument for number of turtles, defaulting to 1")

    # 3) Spawn extra turtles if needed
    spawn_turtles(num_robots)

    # 4) Collect the actual positions of each spawned turtle from /turtleX/pose
    #    We do a quick measurement by creating a node, sub to each /pose once
    #    This is a minimal approach.
    robot_positions = []
    temp_node = rclpy.create_node('position_reader')
    poses_received = [False]*num_robots

    def make_callback(idx):
        def cb(p: Pose):
            robot_positions[idx] = (p.x, p.y)
            poses_received[idx] = True
        return cb

    for i in range(num_robots):
        robot_positions.append((0.0, 0.0))  # placeholder
        turtle_name = f"turtle{i+1}"
        temp_node.create_subscription(Pose, f'/{turtle_name}/pose', make_callback(i), 1)

    # Spin a bit until we get at least one pose from each
    spin_start = time.time()
    while not all(poses_received) and (time.time() - spin_start < 5.0):
        rclpy.spin_once(temp_node, timeout_sec=0.1)
    temp_node.destroy_node()

    for i in range(num_robots):
        print(f"[INFO] Robot {i+1} at {robot_positions[i]}")

    # 5) Build RobotInfo for each turtle
    robots = []
    for i in range(num_robots):
        rx, ry = robot_positions[i]
        robots.append(RobotInfo(robot_id=i, init_x=rx, init_y=ry))

    # 6) Run Auction
    assigned_tasks = auction_allocation(robots, tasks, epsilon=0.01, max_rounds=200)

    print("\n*** Final Auction Results ***")
    for r in robots:
        print(f"Robot {r.id+1} -> Letters {[t.id for t in r.assigned_tasks]}")

    # 7) Start TurtlePatternDrawer for each robot with assigned letters
    #    This means we pass only the waypoints for the tasks it won
    #    We'll do it in a single process with MultiThreadedExecutor
    drawers = []
    for r in robots:
        # Combine all letter tasks assigned to r
        # We can just chain them, or do them in sequence letter by letter
        # E.g. if you want them to do letter 0, then letter 5, etc. in that order
        # We'll do it in ascending task.id order for convenience
        r.assigned_tasks.sort(key=lambda t: t.id)
        combined_waypoints = []
        for t in r.assigned_tasks:
            combined_waypoints.append(t.waypoints)

        # If the robot got no tasks, then it will have an empty pattern
        turtle_name = f"turtle{r.id+1}"
        print(turtle_name, combined_waypoints)
        drawer_node = TurtlePatternDrawer(turtle_name, combined_waypoints)
        drawers.append(drawer_node)

    # 8) MultiThreadedExecutor for all drawers
    executor = MultiThreadedExecutor()
    for d in drawers:
        executor.add_node(d)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for d in drawers:
            d.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

