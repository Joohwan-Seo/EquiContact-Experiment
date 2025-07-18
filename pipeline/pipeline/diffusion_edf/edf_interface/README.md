<<<<<<< HEAD
# edf_interface
EDF interface

# Installation
Execute the following command in your terminal.
```shell
pip install -e .
```
Install PyTorch if you don't have it.
```shell
pip install torch==1.13.1
```

# Example
Please run the following notebooks in order with Jupyter:
1. 'env_server.ipynb'
2. 'agent_server.ipynb'
3. 'client.ipynb'

# Usage
Use @expose decorator to share server's class methods with clients.

## Environment Server Example
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroServer, expose

class EnvService():
    def __init__(self): ...

    @expose
    def get_current_poses(self) -> SE3: 
        <YOUR CODE HERE>

    @expose
    def observe_scene(self) -> PointCloud: 
        <YOUR CODE HERE>

    @expose
    def observe_grasp(self) -> PointCloud: 
        <YOUR CODE HERE>

    @expose
    def move_se3(self, target_poses: SE3) -> bool: 
        <YOUR CODE HERE>

service = EnvService()
server = PyroServer(server_name='env', init_nameserver=True)
server.register_service(service=service)
server.run(nonblocking=False) # set nonblocking = True if you want to run server in another thread.

server.close()
```

## Agent Server Example
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroServer, expose

class AgentService():
    def __init__(self):
        pass

    @expose
    def infer_target_poses(self, scene_pcd: PointCloud, 
                           task_name: str,
                           grasp_pcd: PointCloud,
                           current_poses: SE3) -> SE3: 
        <YOUR CODE HERE>

service = AgentService()
server = PyroServer(server_name='agent', init_nameserver=False)
server.register_service(service=service)
server.run(nonblocking=False) # set nonblocking = True if you want to run server in another thread.

server.close()
```

## Client Example
Methods are only for type hinting. You do not have to write the codes.
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroClientBase

class ExampleClient(PyroClientBase):
    def __init__(self, env_server_name: str = 'env',
                 agent_sever_name: str = 'agent'):
        super().__init__(service_names=[env_server_name, agent_sever_name])

    def get_current_poses(self, **kwargs) -> SE3: ... 
    
    def observe_scene(self, **kwargs) -> PointCloud: ...
    
    def observe_grasp(self, **kwargs) -> PointCloud: ...

    def move_se3(self, target_poses: SE3, **kwargs) -> bool: ...

    def infer_target_poses(self, scene_pcd: PointCloud, 
                           task_name: str,
                           grasp_pcd: Optional[PointCloud] = None,
                           current_poses: Optional[SE3] = None, 
                           **kwargs) -> SE3: ...

client = ExampleClient(env_server_name='env', agent_sever_name='agent')
```
=======
# [ICLR 2023] Equivariant Descriptor Fields (EDFs)

Official PyTorch implementation of Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning (ICLR 2023 Poster).

The paper can be found at: https://arxiv.org/abs/2206.08321

Project Website: https://sites.google.com/view/edf-robotics

> [!TIP]
> Please also find our new work, Diffusion-EDFs at: https://sites.google.com/view/diffusion-edfs/home

> This is a standalone implementation of EDFs without PyBullet simulation environments. To reproduce our experimental results in the paper, please check the following branch:  https://github.com/tomato1mule/edf/tree/iclr2023_rebuttal_ver

> EDF+ROS MoveIt example (unstable): https://github.com/tomato1mule/edf_pybullet_ros_experiment


## Installation

**Step 1.** Clone Github repository.
```shell
git clone https://github.com/tomato1mule/edf
```

**Step 2.** Setup Conda environment.
```shell
conda create -n edf python=3.8
conda activate edf
```

**Step 3.** Install Dependencies
```shell
CUDA=cu113
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install iopath fvcore
pip install --no-index --no-cache-dir pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_${CUDA}_pyt1110/download.html
pip install -e .
```

# Usage
## Train
```shell
python pick_train.py
python place_train.py
```

If you want to load already trained checkpoints, please rename 'checkpoint_example' folder to 'checkpoint'.
## Evaluate
Please run the example notebook codes for visualizing sampled poses from trained models (evaluate_pick.ipynb and evaluate_place.ipynb)

## View train log
```shell
python train_log_viewer.py --logdir="checkpoint/mug_10_demo/ {pick or place} /trainlog_iter_{iter}.gzip"
```

# Citing
If you find our paper useful, please consider citing our paper:
```
@inproceedings{
ryu2023equivariant,
title={Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning},
author={Hyunwoo Ryu and Hong-in Lee and Jeong-Hoon Lee and Jongeun Choi},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=dnjZSPGmY5O}
}
```



>>>>>>> 5c10bf464c759558958836f38c11831a7943f4a4
