# Ground-FALS
 a novel Ground-based Fast Approximate Least Squares (Ground-FALS) estimator for ground normal vectors
## Related Works

1. [NA-LOAM](https://github.com/BuaaYfl/NA-LOAM):  NA-LOAM: Normal-based Adaptive LiDAR Odometry and Mapping
 
![normals.jpg](Figure%2FFig4.jpg)
## How to use
Tested under Ubuntu 18.04 with opencv4.0. 
Output point cloud with the topic "cloud_normal".

```angular2html
mkdir -p ground_fals_ws/src
cd ground_fals_ws/src
git clone https://github.com/BuaaYfl/Ground-FALS.git
```
set _**OpenCV_DIR**_ in the CMakeLists.txt to your local path, and please compile the  _**opencv-contrib**_ module in advance.
```
cd .. & catkin_make
source devel/setup.bash
```
### run with our pre-built lookup table
For the [M2DGR](https://github.com/SJTU-ViSYS/M2DGR) dataset
```
roslaunch ground_fals normal_m2dgr.launch 
```
For the [KITTI datasets](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset
```
roslaunch ground_fals normal_kitti.launch 
```
So far, we have only provided launch files for [M2DGR](https://github.com/SJTU-ViSYS/M2DGR) and [KITTI datasets](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
### build a lookup table for a spinning LiDAR
Set the parameters in the _**yaml**_ file for the LiDAR carefully.
**_compute_table_** must be set to true.
```angular2html
    compute_table: true               # true: compute only the lookup table
    ring_table_dir: "/table_dir"      # lookup table path, read or write
```
The lookup table will be saved to the path specified by  **_ring_table_dir_** upon ros shutdown.
```
roslaunch ground_fals *.launch
rosbag play *.bag
```
Finally, run with your lookup table
```
roslaunch ground_fals *.launch 
```
