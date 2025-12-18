from setuptools import setup

package_name = 'robot_examples'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    py_modules=[
        'example_publisher',
        'example_subscriber'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Textbook Maintainer',
    maintainer_email='example@university.edu',
    description='Example ROS 2 packages for the Physical AI & Humanoid Robotics textbook',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'example_publisher = example_publisher:main',
            'example_subscriber = example_subscriber:main',
        ],
    },
)