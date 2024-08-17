#include "preprocess.h"
#include "tic_toc.h"


#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess(ros::NodeHandle *nh) : nh_(*nh), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
    given_offset_time = false;
    nonground_points.reset(new pcl::PointCloud<PointXYZIRT>());
    ground_points.reset(new pcl::PointCloud<PointXYZIRT>());
    new_nonground_points.reset(new pcl::PointCloud<PointXYZIRT>());
    new_ground_points.reset(new pcl::PointCloud<PointXYZIRT>());

    GroundPublisher = nh_.advertise<sensor_msgs::PointCloud2>("/patchwork/ground", 100, true);
    NonGroundPublisher = nh_.advertise<sensor_msgs::PointCloud2>("/patchwork/non_ground", 100, true);

}

Preprocess::~Preprocess() {}


void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{

    switch (lidar_type)
    {
    case OUSTER:
        // ouster_handler(msg);
        break;

    case VELODYNE:
        velodyne_handler(msg);
        break;
    
    default:
        printf("Error LiDAR Type");
        break;
    }


    *pcl_out = pl_surf;

}


sensor_msgs::PointCloud2 Preprocess::cloud2msg(pcl::PointCloud<PointXYZIRT> cloud)
{
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.frame_id = "camera_init";
    return cloud_ROS;
}


void Preprocess::estimateNormals(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud)
{
    if (compute_table)
    {
        TicToc t_range_image;
        bool systemInited;
   
        systemInited = range_image.buildTableFromcloud(cloud);
    
        if(systemInited)
        {
            printf(".....Computing M inverse....\n");
            if(useDepth){
                range_image.computeMInverseFromdepth();
                printf("Computing M inverse matrix from depth.\n");
                printf(".....Saving range image lookup table....\n");
                range_image.saveMInverseFromdepth(ring_table_dir, "ring" + std::to_string(N_SCANS));
                ROS_WARN("build range image from rings cost: %fms", t_range_image.toc());
            }else{
                range_image.computeMInverseFromheight();
                printf("Computing M inverse matrix from height.\n");
                printf(".....Saving range image lookup table....\n");
                range_image.saveMInverseFromheight(ring_table_dir, "ring" + std::to_string(N_SCANS));
                ROS_WARN("build range image from rings cost: %fms", t_range_image.toc());
            }

        }       

        return;
    }
    pcl::PointCloud<PointXYZIRT> pc_curr;
    pcl::PointCloud<PointXYZIRT> pc_ground;
    pcl::PointCloud<PointXYZIRT> pc_non_ground;

    

    static double time_taken;
    pc_curr = *cloud;
    TicToc t_0;
    PatchworkGroundSeg->estimate_ground(pc_curr, pc_ground, pc_non_ground, time_taken);
    seg_time = t_0.toc();

    GroundPublisher.publish(cloud2msg(pc_ground));
    NonGroundPublisher.publish(cloud2msg(pc_non_ground));

    ground_points = pc_ground.makeShared();
    nonground_points = pc_non_ground.makeShared();
    
    TicToc t_1;
    projectPointCloudFromDepth(nonground_points,new_nonground_points);
    projectPointCloudFromHeight(ground_points,new_ground_points);
    proj_time = t_1.toc();

    // std::cout << "  before: " << nonground_points->points.size()  <<  std::endl;
    // std::cout << "  after: " << new_nonground_points->points.size()  <<  std::endl;

    std::cout << "  before: " << ground_points->points.size()  <<  std::endl;
    std::cout << "  after: " << new_ground_points->points.size()  <<  std::endl;

    TicToc t_2;
    computeRingNormalsFromDepth();
    computeRingNormalsFromHeight();
    compu_time = t_2.toc();

    TicToc t_3;
    extractCloudAndNormalsFromDepth(new_nonground_points);
    extractCloudAndNormalsFromHeight(new_ground_points);
    smooth_time = t_3.toc();
}


void Preprocess::projectPointCloudFromDepth(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, pcl::PointCloud<PointXYZIRT>::Ptr& cloud_out)
{

    rangeMat = cv::Mat(N_SCANS, image_cols, CV_32F, cv::Scalar::all(FLT_MAX));

    cloud_out->clear();
    
    TicToc t_range_image;
    int cloudSize = (int)cloud->points.size();


    // range image projection
    for (int i = 0; i < cloudSize; ++i)
    {
        const PointXYZIRT& thisPoint = cloud->points[i];

        int rowIdn, columnIdn;
        if (!pointInImage(thisPoint, rowIdn, columnIdn))
          continue;
        // std::cout <<" rowIdn  "  << rowIdn << " columnIdn " << columnIdn << std::endl;

        float range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);//  sqrt(x^2 + y^2 + z^2)

//        if (range < 1.0)
//            continue;

       if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
            continue;
  

        rangeMat.at<float>(rowIdn, columnIdn) = range; //record range after correcting

        int index = columnIdn  + rowIdn * image_cols; //index

        PointXYZIRT newPoint;

        newPoint = thisPoint;

        newPoint.index = index;
 
        cloud_out->push_back(newPoint);
    }
}

void Preprocess::projectPointCloudFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, pcl::PointCloud<PointXYZIRT>::Ptr& cloud_out)
{
    heightMat = cv::Mat(N_SCANS, image_cols, CV_32F, cv::Scalar::all(FLT_MAX));

    cloud_out->clear();
    
    TicToc t_range_image;
    int cloudSize = (int)cloud->points.size();

    // height image projection
    for (int i = 0; i < cloudSize; ++i)
    {
        const PointXYZIRT& thisPoint = cloud->points[i];

        int rowIdn, columnIdn;

        if (!pointInImage(thisPoint, rowIdn, columnIdn))
            continue;

        float height = thisPoint.z;// 


        if (heightMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
           continue;

        heightMat.at<float>(rowIdn, columnIdn) = height; //record range after correcting

        int index = columnIdn  + rowIdn * image_cols; //index

        PointXYZIRT newPoint;

        newPoint = thisPoint;

        newPoint.index = index;
 
        cloud_out->push_back(newPoint);
    }
}


void Preprocess::computeRingNormalsFromDepth()
{
    range_image.computeNormalsFromdepth(rangeMat, normalsFromdepth); ///output: normalized normals
}

void Preprocess::computeRingNormalsFromHeight()
{
    range_image.computeNormalsFromheight(heightMat, normalsFromheight); ///output: normalized normals
}



void Preprocess::extractCloudAndNormalsFromDepth(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud)
{
    TicToc t_flip;
    cv::Vec3f lidar(0, 0, 0);
    flipNormalsFromDepth(lidar, cloud, normalsFromdepth);
//    ROS_WARN("flip normals cost: %fms", t_flip.toc());
//    saveNormalPCD("/tmp/normal_flipped.pcd", cloud, normals);

    // smooth normals after flipping
    TicToc t_smo;
//    cv::Mat normal_smoothed;
//        cv::GaussianBlur(normals, normal_smoothed, cv::Size(3, 3), 0);
    cv::medianBlur(normalsFromdepth, normalsFromdepth, 5);
//    saveNormalPCD("/tmp/normal_median_blur.pcd", cloud, normals);
//        TicToc t_nor;
    NormalizeNormalsFromDepth(normalsFromdepth);
}

void Preprocess::flipNormalsFromDepth(const cv::Vec3f& center, const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, cv::Mat& image_normals)
{
    ROS_ASSERT(rangeMat.size == image_normals.size);
    int p_size = cloud->size();
    for (int i = 0; i < p_size; ++i)
    {
        const PointXYZIRT &thisPoint = cloud->points[i];                    
        // vector: from center to point
        cv::Vec3f vc2p(thisPoint.x - center(0), thisPoint.y - center(1), thisPoint.z - center(2));
        vc2p /= norm(vc2p);
        int index;
        int rowIdn, columnIdn;
        
        index = thisPoint.index;
        rowIdn = index / image_cols;
        columnIdn = index - rowIdn * image_cols;
        cv::Vec3f &n = image_normals.at<cv::Vec3f>(rowIdn, columnIdn); ///already normalized
        // std::cout << "rowIdn: " << rowIdn << " columnIdn: " << columnIdn << std::endl;

        if (vc2p.dot(n) > 0)
            n *= -1;
    }
}

void Preprocess::NormalizeNormalsFromDepth(cv::Mat& image_normals)
{
    ROS_ASSERT(rangeMat.size == image_normals.size);
    for (int i = 0; i < N_SCANS; ++i)//
        for (int j = 0; j < image_cols; ++j)
            if (rangeMat.at<float>(i, j) != FLT_MAX)//
            {
                cv::Vec3f &n = image_normals.at<cv::Vec3f>(i, j);
                
                n /= norm(n);
            }
}

void Preprocess::extractCloudAndNormalsFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud)
{
    TicToc t_flip;
    flipNormalsFromHeight(cloud, normalsFromheight);

    // smooth normals after flipping
    TicToc t_smo;
//    cv::Mat normal_smoothed;
//        cv::GaussianBlur(normals, normal_smoothed, cv::Size(3, 3), 0);
    cv::medianBlur(normalsFromheight, normalsFromheight, 5);
//    saveNormalPCD("/tmp/normal_median_blur.pcd", cloud, normals);
//        TicToc t_nor;
    NormalizeNormalsFromHeight(normalsFromheight);
}


void Preprocess::flipNormalsFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, cv::Mat& image_normals)
{
    ROS_ASSERT(heightMat.size == image_normals.size);
    int p_size = cloud->size();
    for (int i = 0; i < p_size; ++i)
    {
        const PointXYZIRT &thisPoint = cloud->points[i];                    
        int index;
        int rowIdn, columnIdn;
        
        index = thisPoint.index;
        rowIdn = index / image_cols;
        columnIdn = index - rowIdn * image_cols;
        cv::Vec3f &n = image_normals.at<cv::Vec3f>(rowIdn, columnIdn); ///already normalized
        // std::cout << "rowIdn: " << rowIdn << " columnIdn: " << columnIdn << std::endl;
        if (!std::isfinite(n(0)) || !std::isfinite(n(1)) || !std::isfinite(n(2)))
            continue;

        if (n(2) < 0)
            n *= -1;
    }

}

void Preprocess::NormalizeNormalsFromHeight(cv::Mat& image_normals)
{
    ROS_ASSERT(heightMat.size == image_normals.size);
    for (int i = 0; i < N_SCANS; ++i)//
        for (int j = 0; j < image_cols; ++j)
            if (heightMat.at<float>(i, j) != FLT_MAX)//
            {
                cv::Vec3f &n = image_normals.at<cv::Vec3f>(i, j);
                
                n /= norm(n);
            }
}


bool Preprocess::pointInImage(const PointXYZIRT& point,int& rowIdn, int& columnIdn)
{

    if (has_ring == true){
        rowIdn = (int)point.ring;
    }
    else{
        float verticalAngle;
        verticalAngle = atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        double ang_bottom = 24.8;                           
        double ang_res_y  = 0.4253968;

        rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
    }
    if (rowIdn < 0 || rowIdn >= N_SCANS)
        return false;
    static float ang_res_x = 360.0 / float(image_cols); 
    float horizonAngle = atan2(point.y, -point.x);
    if (horizonAngle < 0)
        horizonAngle += 2 * M_PI;
    horizonAngle = horizonAngle  * 180 / M_PI;
    columnIdn = round(horizonAngle/ ang_res_x);

    if (columnIdn < 0 || columnIdn >= image_cols)
        return false;
    return true;
}


void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();

    pcl::PointCloud<PointXYZIRT>::Ptr pl_orig(new pcl::PointCloud<PointXYZIRT>());
//    pcl::PointCloud<PointXYZIRT>::Ptr pl_orig;
    pcl::fromROSMsg(*msg, *pl_orig);
   int plsize = pl_orig->points.size();
   ROS_INFO("cloud input size: %d", plsize);

    if (ringfals_en)
    {
        TicToc t_nor;
        estimateNormals(pl_orig); // get cloud_with_normal
        if (runtime_log)
        {
            num_scans++;
            aver_normal_time = aver_normal_time * (num_scans - 1) / num_scans + t_nor.toc() / num_scans;
            aver_proj_time = aver_proj_time * (num_scans - 1) / num_scans + proj_time / num_scans;
            aver_seg_time = aver_seg_time * (num_scans - 1) / num_scans + seg_time / num_scans;

            aver_compu_time = aver_compu_time * (num_scans - 1) / num_scans + compu_time / num_scans;
            aver_smooth_time = aver_smooth_time * (num_scans - 1) / num_scans + smooth_time / num_scans;
            ROS_INFO("[Ring FALS] mean seg %0.3fms , project %0.3fms, compute %0.3fms, smooth %0.3fms, total %0.3fms",
                     aver_seg_time, aver_proj_time, aver_compu_time, aver_smooth_time, aver_normal_time);
        }
    }

    PointCloudXYZI ground_points_out, non_ground_points_out;

    bool normal_valid = ringfals_en && !compute_table;

    if (normal_valid)
    {
        addNormalInfoFromDepth(new_nonground_points,non_ground_points_out);
        addNormalInfoFromHeight(new_ground_points,ground_points_out);
        std::cout << "  afterafter: " << ground_points_out.points.size()  <<  std::endl;

        // pl_surf += non_ground_points_out;
        pl_surf += ground_points_out;
    }else{

        int plsize = pl_orig->points.size();
        if (plsize == 0) return;
        pl_surf.reserve(plsize);
        for (int i = 0; i < plsize; i++)
        {
            PointType added_pt;
            const PointXYZIRT& orig_pt = pl_orig->points[i];
            // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
            added_pt.x = orig_pt.x;
            added_pt.y = orig_pt.y;
            added_pt.z = orig_pt.z;
            added_pt.intensity = orig_pt.intensity;
            pl_surf.points.push_back(added_pt);

        }
    }
}

void Preprocess::addNormalInfoFromDepth(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, PointCloudXYZI & cloud_out) {

    int plsize = cloud->points.size();
    if (plsize == 0) return;

    cloud_out.clear();
    cloud_out.reserve(plsize);

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        // const PointXYZIRT& orig_pt = pl_orig->points[i];
        const PointXYZIRT& orig_pt = cloud->points[i];
        added_pt.x = orig_pt.x;
        added_pt.y = orig_pt.y;
        added_pt.z = orig_pt.z;
        added_pt.intensity = orig_pt.intensity;

        int index = orig_pt.index;
        if (i % point_filter_num == 0)
        {
            if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
            {
                int rowIdn, columnIdn;
                rowIdn = index / image_cols;
                columnIdn = index - rowIdn * image_cols;
                const cv::Vec3f &n_cv = normalsFromdepth.at<cv::Vec3f>(rowIdn, columnIdn);
                

                if (!std::isfinite(n_cv(0)) || !std::isfinite(n_cv(1)) || !std::isfinite(n_cv(2)))
                    continue;
                added_pt.normal_x = n_cv(0);
                added_pt.normal_y = n_cv(1);
                added_pt.normal_z = n_cv(2);
                cloud_out.points.push_back(added_pt);
            }
        }
    }

}


void Preprocess::addNormalInfoFromHeight(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud, PointCloudXYZI & cloud_out) {

    int plsize = cloud->points.size();
    if (plsize == 0) return;

    cloud_out.clear();
    cloud_out.reserve(plsize);
    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        // const PointXYZIRT& orig_pt = pl_orig->points[i];
        const PointXYZIRT& orig_pt = cloud->points[i];
        added_pt.x = orig_pt.x;
        added_pt.y = orig_pt.y;
        added_pt.z = orig_pt.z;
        added_pt.intensity = orig_pt.intensity;
        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            int index = orig_pt.index;

            int rowIdn, columnIdn;
            rowIdn = index / image_cols;
            columnIdn = index - rowIdn * image_cols;
            const cv::Vec3f &n_cv = normalsFromheight.at<cv::Vec3f>(rowIdn, columnIdn);
            
            if (!std::isfinite(n_cv(0)) || !std::isfinite(n_cv(1)) || !std::isfinite(n_cv(2)))
                continue;
 
            added_pt.normal_x = n_cv(0);
            added_pt.normal_y = n_cv(1);
            added_pt.normal_z = n_cv(2);

            cloud_out.points.push_back(added_pt);
          }
        }
   
    }
}


void Preprocess::initNormalEstimator() {

    image_cols = Horizon_SCAN;
    range_image = RangeImage(N_SCANS, image_cols,has_ring,min_table_cloud);

    PatchworkGroundSeg.reset(new PatchWork<PointXYZIRT>(&nh_));


    if (!compute_table) {
        if (!range_image.loadMInverseFromdepth(ring_table_dir, "ring" + std::to_string(N_SCANS))) {
            ROS_ERROR("Wrong path to ring normal M depth file, EXIT.");
        }
        if (!range_image.loadMInverseFromheight(ring_table_dir, "ring" + std::to_string(N_SCANS))) {
            ROS_ERROR("Wrong path to ring normal M height file, EXIT.");
        }
    }
}