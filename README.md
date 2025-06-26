<p align="center">
  <h1 align="center">GS-NBV: a Geometry-based, Semantics-aware Viewpoint Planning Algorithm for Avocado Harvesting under Occlusions</h1>
  <p align="center">
    <strong>Xiao'ao Song</strong>
    ·
    <strong>Konstantinos Karydis</strong>
  </p>
</p>
<h2 align="center">
  Paper: 
  <a href="https://arxiv.org/pdf/2311.16759" target="_blank">IEEE</a> | 
  <a href="https://arxiv.org/pdf/2311.16759" target="_blank">ArXiv</a>
</h2>

[GSNBV Demo](https://github.com/lineojcd/GSNBV/blob/main/assets/gsnbv_group1_demo.mp4)


This is the offical repository for the CASE 2025 paper: "GS-NBV: a Geometry-based, Semantics-aware Viewpoint Planning Algorithm for Avocado Harvesting under Occlusions"

## Installation
**Prerequisites**

This project is build on Ubuntu20.04, ROS noetic platform and python 3.8 environment. Please install the platform before any installation steps. When you install ROS, please deactivate conda environment. You will also need at least 8GB of GPU VRAM to run this project.

**Create a ROS Workspace**
```
sudo apt-get install ros-noetic-catkin python3-catkin-tools 
mkdir -p ~/gsnbv_ws/src
cd ~/gsnbv_ws/
catkin build
echo "source ~/gsnbv_ws/devel/setup.bash" >> ~/.bashrc
source ~/gsnbv_ws/devel/setup.bash
```

**Clone the repository**
```
cd ~/Downloads
git clone git@github.com:lineojcd/GSNBV.git
```
**ROS dependencies**
```
sudo apt-get install ros-noetic-gazebo-ros* ros-noetic-ros-controllers 
sudo apt-get install ros-noetic-trac-ik ros-noetic-trac-ik-kinematics-plugin -y
sudo apt-get install ros-noetic-moveit
sudo apt install python3-pip python3-defusedxml
```
**Python packages**
```
conda create -n gsnbv python==3.8.10
conda activate gsnbv
conda install libtiff=4.0.8 libffi==3.3
pip install Pillow==2.2.2
pip install opencv-python==4.9.0.80
pip install pyyaml==6.0.1 rospkg==1.5.1
pip install scipy==1.10.1 pytransform3d==3.5.0 open3d==0.18.0 empy==3.3.4
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

#Install Python packages for yolov8
cd ~/Downloads/GSNBV/viewpoint_planning/src/perception/ultralytics_yolov8
pip install -e .
```
**Compile**
```
cd ~/Downloads
cp -r GSNBV/* ~/gsnbv_ws/src
rm -rf GSNBV
cd ~/gsnbv_ws/
catkin build
# you might need to re-run catkin build multiple times to sucessfully complete the build
```
## Execute

Bring up the robot (Kinova robotic arm + Realsense D435i camera):
```
# In the 1st terminal: launch Kinova arm.
roslaunch kinova_gazebo my_robot_launch.launch
```
Start the GSNBV planner in a new terminal:
```
# In the 2nd terminal: launch simulated avocado tree and run GSNBV algo
conda activate gsnbv
roslaunch viewpoint_planning viewpoint_planning.launch
```
## Citation
If you use this repository, please cite below:
```bibtex
@article{burusa2023gradient,
  title={Gradient-based Local Next-best-view Planning for Improved Perception of Targeted Plant Nodes},
  author={Burusa, Akshay K and van Henten, Eldert J and Kootstra, Gert},
  journal={arXiv preprint arXiv:2311.16759},
  year={2024}
}
```
