import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/horowitzlab/ros2_ws_clean/src/pipeline/pipeline/diffusion_edf/install/diffusion_edf'
