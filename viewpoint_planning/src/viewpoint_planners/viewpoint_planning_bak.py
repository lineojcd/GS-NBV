import rospy
import numpy as np

from abb_control.arm_control_client import ArmControlClient
from perception.perceiver import Perceiver
from viewpoint_planners.my_viewpoint_sampler import MyViewpointSampler
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from viewpoint_planners.gradientnbv_planner import GradientNBVPlanner
from utils.my_sdf_spawner import MySDFSpawner
from utils.sdf_spawner import SDFSpawner
from utils.py_utils import numpy_to_pose


class ViewpointPlanning:
    def __init__(self):
        # change this one to Jaco2
        self.arm_control = ArmControlClient()
        
        self.perceiver = Perceiver()
        self.viewpoint_sampler = ViewpointSampler()
        self.sdf_spawner = SDFSpawner()
        print("start sleep 3 s...")
        rospy.sleep(1.0)
        print("done sleeping")
        
        # self.experiment = 'gnbv_exp'
        
        
        if self.experiment == 'gnbv_exp':
            self.model_init_pose= [0.9, 0.15 , 0.70] #param for spawn models in Gazebo
            self.sdf_spawner = MySDFSpawner(init_pose = self.model_init_pose)
            # VP Sampled player
            self.viewpoint_sampler = MyViewpointSampler()
            self.init_target_align_position = np.array([0.5, -0.4, 1.18])
            self.init_target_distance = 0.35
            self.fruit_sampling_radius = 0.27
            self.tree_position = [0.5 - self.fruit_sampling_radius, -0.4 - self.fruit_sampling_radius, 0.84]   
            self.avo_height_offset = 0.029
        else:
            self.sdf_spawner = SDFSpawner()
            self.viewpoint_sampler = ViewpointSampler()
        
        
        self.config()
        # Gradient-based planner
        self.gradient_planner = GradientNBVPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=1,
        )

    def run(self):
        self.run_gradient_nbv()
        # self.run_random()

    def config(self):
        # Configure target
        self.target_position = np.array([0.5, -0.4, 1.1])
        # occlusion_position = np.array([0.5, -0.3, 1.25])  # top occlusion
        # occlusion_position = np.array([0.5, -0.3, 0.95])  # bottom occlusion
        occlusion_position = np.array([0.65, -0.3, 1.1])  # left occlusion
        # occlusion_position = np.array([0.35, -0.3, 1.1])  # right occlusion
        self.sdf_spawner.spawn_box(occlusion_position)
        # Configure initial camera viewpoint
        self.camera_pose = self.viewpoint_sampler.predefine_start_pose(self.target_position)
        
        # Configure scene
        self.grid_size = np.array([0.3, 0.6, 0.3])
        self.grid_center = self.target_position
        
        if self.experiment == 'gnbv_exp':
            self.camera_pose = self.viewpoint_sampler.predefine_start_pose(self.init_target_align_position, self.init_target_distance)
            self.grid_size = np.array([0.42, 0.45, 0.84]) #Configure scene region
            self.grid_size = np.array([0.42, 0.45, 0.5]) #Configure scene region
            self.grid_center = self.tree_position
            
        print("Init camera_pose is:", self.camera_pose)
        self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        # Configure camera
        camera_info = self.perceiver.get_camera_info()
        self.image_size = np.array([camera_info.width, camera_info.height])
        self.intrinsics = np.array(camera_info.K).reshape(3, 3)

    def run_gradient_nbv(self):
        self.camera_pose, loss, iters = self.gradient_planner.next_best_view()
        is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        rospy.sleep(1.0)
        if is_success:
            depth_image, points, semantics = self.perceiver.run()
            coverage = self.gradient_planner.update_voxel_grid(depth_image, semantics, self.camera_pose)
            print("Target coverage: ", coverage)
            self.gradient_planner.visualize()