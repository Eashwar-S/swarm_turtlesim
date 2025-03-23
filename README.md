# Swarm Turtlesim

This repository contains a **ROS 2** package called `swarm_turtlesim`, showcasing **multi-robot task allocation** in the Turtlesim simulator using an **auction algorithm**. Each turtle is assigned specific letters in the phrase **“SWARM ROBOTICS”**, then draws them in parallel—resulting in faster completion compared to a single turtle approach.

## Dependencies

- **ROS 2 (Jammy)** or compatible distribution  
- **Python 3**  
- **turtlesim** package (installed by default with many ROS 2 installations)  
- **colcon** for building (standard in ROS 2)

## Installation & Setup

1. **Clone** this repository into your ROS 2 workspace:

    ```bash
    cd ~/ros2_ws/src
    git clone https://github.com/Eashwar-S/swarm_turtlesim.git
    ```

2. **Build** your workspace:

    ```bash
    cd ~/ros2_ws
    colcon build
    ```

3. **Source** your workspace so ROS 2 sees the new package:

    ```bash
    source ~/ros2_ws/install/setup.bash
    ```

   *(You can add this line to your shell startup script if desired.)*

## Usage

1. **Launch Turtlesim** (if not auto-launched by your node):

    ```bash
    ros2 run turtlesim turtlesim_node
    ```

2. **Run the `turtle_pattern_drawer` node** with a specified number of robots in a separate terminal:

    ```bash
    ros2 run swarm_turtlesim turtle_pattern_drawer <number_of_robots>
    ```

   For example:

    ```bash
    ros2 run swarm_turtlesim turtle_pattern_drawer 4
    ```

   This example spawns **4** turtles, each assigned different letters by the auction algorithm. They will then draw **“SWARM ROBOTICS”** in parallel.

## Results

<img align="center" width="1000" height="1000" src="result.gif">

## Blog Reference

For a detailed explanation of the **auction-based** multi-robot approach, performance graphs, and examples, check out the accompanying blog post:

[**Swarm Robots – A Gentle Introduction**](https://www.eashwarsathyamurthy.com/post/swarm-robotics-a-gentle-introduction)



