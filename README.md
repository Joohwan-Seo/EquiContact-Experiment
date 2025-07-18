# EquiContact-Experiment

This is a repo to provide a "raw" experimental code.

We have uploaded the raw ROS2 codes, each folder represents an individual project in the ``ros2_ws/src`` folder. 

Note that the objective of this codebase is to provide a glimpse of how it is formulated.

The detailed implementation to the robot experiment varies a lot by its own conditions, for example, which camera and robots they use, etc. Thus, we will not provide any supports to this codebase; again, we just post the code "as it is".

The code is Written by Joohwan Seo, Arvind Kruthiventy, Soomi Lee, Ph.D. stduent at Mechanical Engineering, UC Berkeley, in Autonomy, Robotics, and Control lab (Professor Roberto Horowitz). 

## Anaconda Environment
ROS2 is not well compatible with the Anaconda Environment, and it can only work with the single python version. In our codebase, all codes are implemented with ``python == 3.10``. If the training of the submodules (such as Diff-EDF or ACT) cannot be performed, we just trained the submodules in the recommended setting (ex. python 3.8 for Diff-EDF) in the other computer and just loaded the weights of the neural network. 

## ``camera_processing``
This project have a camera-related codes, such as, how to process point cloud, 