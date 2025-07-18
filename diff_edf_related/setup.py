from setuptools import find_packages, setup

package_name = 'pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
    'torch==1.13.1'
    	],
    zip_safe=True,
    maintainer='horowitzlab',
    maintainer_email='arvindkruthiventy@berkeley.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'DiffEDFService = pipeline.diffusion_edf.diff_edf_service:main',
        ],
    },
)
