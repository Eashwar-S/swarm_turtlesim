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
        self.robot_cost_so_far = 0.0

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
                # if task.id in unassigned:
                    cost = robot.cost_to_do_task(task)
                    score =  cost + task.price + robot.robot_cost_so_far
                    if score < best_score:
                        second_best_score = best_score
                        best_score = score
                        best_task = task
                    elif score < second_best_score:
                        second_best_score = score

            if best_task is not None:
                if robot.robot_cost_so_far == 0.0:
                    increment = (second_best_score - best_score) + epsilon
                else:
                    increment = ((second_best_score - best_score) + epsilon)/robot.robot_cost_so_far
                # if robot.robot_cost_so_far == 0.0:
                #     new_bid = best_task.price + increment
                # else:
                new_bid = (best_task.price + increment) #- robot.robot_cost_so_far
                robot_bids.append((robot.id, best_task.id, new_bid))

        print(f'round - {round_num}')
        # print(robot_bids)
        if not robot_bids:
            break
        
        # Step 2: Group bids by task, pick highest
        bids_by_task = {}
        for (r_id, t_id, bid_val) in robot_bids:
            if t_id not in bids_by_task:
                bids_by_task[t_id] = []
            bids_by_task[t_id].append((r_id, bid_val))

        # print(f'bids by task - {bids_by_task}')
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
                assigned_task = task_dict[t_id]
                assigned_task.assigned_robot = best_bidder
                assigned_task.price = best_bid_val

                winner_robot = robots[best_bidder]

                # print(f'Assigning task {assigned_task.id} to robot {best_bidder}')
                # ACCUMULATE the new cost
                winner_robot.robot_cost_so_far += winner_robot.cost_to_do_task(assigned_task)
                # Update the robot's position to the last waypoint in the letter
                last_x, last_y = assigned_task.waypoints[-1]
                winner_robot.x = last_x
                winner_robot.y = last_y
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
        req.x = float(random.randint(2, 10))
        req.y = float(random.randint(3, 8))
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

    letters_list = ['S', 'W', 'A', 'R', 'M', 'R', 'O', 'B', 'O', 'T', 'I', 'C', 'S']
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
    assigned_tasks = auction_allocation(robots, tasks, epsilon=100.0, max_rounds=200)

    print("\n*** Final Auction Results ***")
    
    for r in robots:
        print(f"Robot {r.id+1} -> Letters {[letters_list[t.id]for t in r.assigned_tasks]} -> Distance travelled {r.robot_cost_so_far}")

    print()    
    print(f'Max of all distances - {max([r.robot_cost_so_far for r in robots])}')
    # 7) Start TurtlePatternDrawer for each robot with assigned letters
    #    This means we pass only the waypoints for the tasks it won
    #    We'll do it in a single process with MultiThreadedExecutor
    drawers = []
    for r in robots:
        # Combine all letter tasks assigned to r
        # We can just chain them, or do them in sequence letter by letter
        # E.g. if you want them to do letter 0, then letter 5, etc. in that order
        # We'll do it in ascending task.id order for convenience
        print(f'robot id - {r.id}, assigned tasks - {[t.id for t in r.assigned_tasks]}')
        # r.assigned_tasks.sort(key=lambda t: t.id)
        combined_waypoints = []
        for t in r.assigned_tasks:
            combined_waypoints.append(t.waypoints)

        # If the robot got no tasks, then it will have an empty pattern
        turtle_name = f"turtle{r.id+1}"
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

