#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <point_type.h>

#include "range_image.h"
#include "patchwork.h"


using namespace std;
enum LID_TYPE{AVIA = 1, VELODYNE, OUSTER}; //{1, 2, 3}

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Preprocess(ros::NodeHandle *nh);
    ~Preprocess();

    int lidar_type, point_filter_num, N_SCANS, time_unit;
    double blind;
    bool given_offset_time;

    bool compute_table = false;
    int Horizon_SCAN, image_cols;
    string ring_table_dir;
    bool useDepth;
    bool runtime_log = false;
    bool has_ring = true;
    bool ringfals_en = false;
    int  min_table_cloud;
    void initNormalEstimator();
    void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

    RangeImage range_image;

    PointCloudXYZI pl_surf;
    private:
        void ouster_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void estimateNormals(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud);
        void projectPointCloudFromDepth(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud,pcl::PointCloud<PointXYZIRT>::Ptr& cloud_out);
        void projectPointCloudFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud,pcl::PointCloud<PointXYZIRT>::Ptr& cloud_out);

        bool pointInImage(const PointXYZIRT& point,int& row, int& col);

        void computeRingNormalsFromDepth();
        void computeRingNormalsFromHeight();

        void extractCloudAndNormalsFromDepth(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud);
        void extractCloudAndNormalsFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud);

        void flipNormalsFromDepth(const cv::Vec3f& center, const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, cv::Mat& image_normals);
        
        void flipNormalsFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, cv::Mat& image_normals);

        void NormalizeNormalsFromDepth(cv::Mat& image_normals);
        void NormalizeNormalsFromHeight(cv::Mat& image_normals);
        void addNormalInfoFromDepth(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, PointCloudXYZI & cloud_out);
        void addNormalInfoFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, PointCloudXYZI & cloud_out);

        sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<PointXYZIRT> cloud);
        cv::Mat rangeMat;//cloud projection，value default: FLT_MAX
        cv::Mat heightMat;//cloud projection，value default: FLT_MAX

        cv::Mat normalsFromdepth; //record ring normals
        cv::Mat normalsFromheight; //record ring normals

        ros::Publisher  GroundPublisher, NonGroundPublisher;
        ros::NodeHandle nh_;

        boost::shared_ptr<PatchWork<PointXYZIRT> > PatchworkGroundSeg;

        pcl::PointCloud<PointXYZIRT>::Ptr nonground_points , ground_points ;
        pcl::PointCloud<PointXYZIRT>::Ptr new_nonground_points, new_ground_points;

        double aver_normal_time = 0.0, aver_seg_time = 0.0,aver_proj_time = 0.0, aver_compu_time = 0.0, aver_smooth_time = 0.0;
        double seg_time = 0.0,proj_time = 0.0, compu_time = 0.0, smooth_time = 0.0;
        
        int num_scans = 0;
        float ang_res_x; //angle resolution 360 / 1800 = 0.2

};