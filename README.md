# EquiContact-Experiment

This is a repo to provide a "raw" experimental code.

We have uploaded the raw ROS2 codes, each folder represents an individual project in the ``ros2_ws/src`` folder. 

Note that the objective of this codebase is to provide a glimpse of how it is formulated.

The detailed implementation to the robot experiment varies a lot by its own conditions, for example, which camera and robots they use, etc. Thus, we will not provide any supports to this codebase.  

The naming conventions and codes implementations are somewhat ad-hoc, and the codes are quite nasty - sorry for that.

The datasets and trained neural network is not included in this codebase.

The code is Written by Joohwan Seo, Arvind Kruthiventy, and Soomi Lee, Ph.D. stduent at Mechanical Engineering, UC Berkeley, in Autonomy, Robotics, and Control lab (Professor Roberto Horowitz). 

## Anaconda Environment
ROS2 is not well compatible with the Anaconda Environment, and it can only work with the single python version. In our codebase, all codes are implemented with ``python == 3.10``. If the training of the submodules (such as Diff-EDF or ACT) cannot be performed, we just trained the submodules in the recommended setting (ex. python 3.8 for Diff-EDF) in the other computer and just loaded the weights of the neural network. 

## Default Experiment Running Process
You first need to launch 7 terminals. 3 to activate cameras and 4 to run experiment modules.

**Anaconda Environments** 
- ``neuromeka`` environment is for default environment where indy7 sdk is installed.
- ``diff_inference`` environment is to run Diffusion-EDF. https://github.com/tomato1mule/diffusion_edf
- ``aloha_inference`` environment is to run ACT. https://github.com/tonyzhaozh/act 

### Launch Cameras
Activate RealSense Camera (wrist camera #1)
```
conda activate neuromeka
cd ros2_ws
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30 camera_namespace:=realsense
```

Activate ArduCam Camera (wrist camera #2)
```
conda activate neuromeka
python ros2_ws_clean/src/camera_processing/camera_processing_new/camera_processing_new/camera_utils/usb_node.py
```

Activate Orbbec Cameras (two external cameras)
```
conda activate neuromeka
cd ros2_ws_clean
ros2 launch orbbec_camera multi_camera.launch.py
```

### Running Experiment Modules 
Run ``DiffEDFServer.py``.
```
conda activate diff_inference
cd ros2_ws_clean/src/pipeline/pipeline/diffusion_edf
python DiffEDFService.py
```

### On the beginning of the every episode of experiments:
Run ``robot_gac.py``
```
conda activate neuromeka
cd ros2_ws_clean/src/indy_utils/indy_utils
python robot_gac.py
```

Run ``ACT Server``
```
conda activate aloha_inference
cd ros2_ws_clean/src/inference_pipeline/inference_pipeline
python ACTInferenceServer.py
```

Run ``Full Pipeline Client``
```
conda activate neuromeka
cd ros2_ws_clean/src/pipeline/pipeline/diffusion_edf
python FullPipelineClient.py
```

## ``camera_processing``
This project have a camera-related codes, such as, how to process point cloud. This code depends on the several camera modules, in our case is from ORBBEC and RealSense. We refer to the repos as follows.

Orbbec: https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main

RealSense: https://github.com/IntelRealSense/realsense-ros

In ``camera_interfaces`` folder, the required message files are defined.

## ``indy_utils``
This project have a does related to the robot movement, teleoperation and geometric admittance control. 

## ``inference_pipeline``
This project is to run an ACT pipeline. Please refer to our simulation implementation (https://github.com/Joohwan-Seo/EquiContact-Simulation.git) for the test of this code and detailed implementation.

## ``pipeline``
This project is to run Diffusion-EDF pipeline.