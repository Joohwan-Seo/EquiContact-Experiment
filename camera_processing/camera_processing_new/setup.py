from setuptools import find_packages, setup

package_name = 'camera_processing_new'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='horowitzlab',
    maintainer_email='arvindkruthiventy@berkeley.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'PointCloudSub = camera_processing_new.camera_utils.get_pcd:main',
            'ImageSub = camera_processing_new.camera_utils.collect_image_realsense:main',
            'ImageSubOrbbec = camera_processing_new.camera_utils.collect_image_orbbec:main',
            'PointCloudSubRGBD = camera_processing_new.camera_utils.get_pcd_from_rgbd_gpt:main',
            'ImgService = camera_processing_new.camera_utils.get_img_service:main',
            'ImgClientAsync = camera_processing_new.camera_utils.get_img_client:main',
            'ImageSubOrbbecMulti = camera_processing_new.camera_utils.collect_image_orbbec_dual:main',
            'PCDService = camera_processing_new.camera_utils.get_pcd_service:main',
            'PCDClient = camera_processing_new.camera_utils.get_pcd_client:main',
            'DataRecorder = camera_processing_new.camera_utils.get_edf_dataset:main',
            'TSDFMerger = camera_processing_new.camera_utils.tsdf_pcd:main',
            'HandEye = camera_processing_new.camera_utils.calibrate_hand_eye:main'
            'DataLogger = camera_processing_new.camera_utils.data_logger:main'
            'ACTDataLogger = camera_processing_new.camera_utils.data_logger_act:main'
        ],
    },
)
