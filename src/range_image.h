#include <vector>
#include <array>
#include <opencv2/core.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>


#include "point_type.h"

/** Just compute the norm of a vector
 * @param vec a vector of size 3 and any type T
 * @return
 */
template<typename T>
T
inline
norm_vec(const cv::Vec<T, 3> &vec)
{
return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

/** Modify normals to make sure they point towards the camera
 * @param normals
 */
template<typename T>
inline
void
signNormal(const cv::Vec<T, 3> & normal_in, cv::Vec<T, 3> & normal_out)
{
cv::Vec<T, 3> res;
if (normal_in[2] > 0)
    res = -normal_in / norm_vec(normal_in);
else
    res = normal_in / norm_vec(normal_in);

normal_out[0] = res[0];
normal_out[1] = res[1];
normal_out[2] = res[2];
}
/** Normalized normals
* @param normals
*/
template<typename T>
inline
void
normalizedNormal(const cv::Vec<T, 3> & normal_in, cv::Vec<T, 3> & normal_out)
{
    normal_out = normal_in / norm_vec(normal_in);
}

class RangeImage
{

private:
    typedef cv::Matx<float, 3, 3> Mat33T;
    typedef cv::Vec<float, 9> Vec9T;
    typedef cv::Vec<float, 3> Vec3T;
    cv::Mat cos_theta;
    cv::Mat sin_theta;
    cv::Mat cos_phi;
    cv::Mat sin_phi;
    cv::Mat sin_phi_inv;
    cv::Mat Ones_mat;
    int rows_, cols_;
    int depth_ = CV_32F;

    bool has_ring_;
    std::vector<int> num_per_pixel; //points in each pixel

    float range_min = 1.0;
    float range_max = 200.0;
    double ang_bottom = 24.8;                           
    double ang_res_y  = 0.4253968;

    int num_cloud = 0;
    int min_table_cloud_;

    int window_size_ = 3;
    cv::Mat_<Vec3T> V_depth; //sin(theta) * cos(phi), sin(phi), cos(theta) * cos(phi)
    cv::Mat_<Vec3T> V_height; //sin(theta) * cos(phi), sin(phi), cos(theta) * cos(phi)
    cv::Mat_<Vec9T> M_inv_depth; //M^-1
    cv::Mat_<Vec9T> M_inv_height; //M^-1

public:
    explicit RangeImage(){}

    explicit RangeImage(const int& rows, const int& cols, const bool& has_ring, const int& min_table_cloud)
            : rows_(rows)
            , cols_(cols)
            , has_ring_ (has_ring)
            , min_table_cloud_(min_table_cloud)

    {
//        range_image = cv::Mat(num_bins_y, num_bins_x, CV_32F, cv::Scalar::all(FLT_MAX));

        //structural information
        cos_theta =  cv::Mat(rows_, cols_, CV_32F);
        sin_theta =  cv::Mat(rows_, cols_, CV_32F);
        cos_phi = cv::Mat(rows_, cols_, CV_32F);
        sin_phi =  cv::Mat(rows_, cols_, CV_32F);
        sin_phi_inv =  cv::Mat(rows_, cols_, CV_32F);
        Ones_mat = cv::Mat::ones(rows_, cols_, CV_32F);
        num_per_pixel.resize(rows_ * cols_, 0);
    }

    bool buildTableFromcloud(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud)
    {
        if (rows_ == 0 || cols_ == 0) {
            std::cout << "rows_ == 0 or cols_ == 0, can't create range image.\n";
            return false;
        }

    //    int num_valid_bins = 0;
    //    range_image = cv::Mat(rows_, cols_, CV_32F, range_init);

        int cloudSize = (int)cloud->points.size();
        static float ang_res_x = 360.0 / float(cols_); //angle resolution 360 / 1800 = 0.2
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            const PointXYZIRT & thisPoint = cloud->points[i];

            int rowIdn ;// ring，

            if (has_ring_ == true){
                rowIdn = cloud->points[i].ring;
            }
            else{
                float verticalAngle, horizonAngle;
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }

            if (rowIdn < 0 || rowIdn >= rows_)
                continue;

            float horizonAngle = atan2(thisPoint.y, -thisPoint.x); //theta
            if (horizonAngle < 0)
                horizonAngle += 2 * M_PI;
            horizonAngle = horizonAngle  * 180 / M_PI;
            int columnIdn = round(horizonAngle/ ang_res_x);

            if (columnIdn < 0 || columnIdn >= cols_)
                continue;

            float range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y +thisPoint.z * thisPoint.z);//  sqrt(x^2 + y^2 + z^2)

            if (range < range_min || range > range_max)
                continue;


            //theta, phi used for building lookup table
            float theta = (float) std::atan2(-thisPoint.y, thisPoint.x); //lidar frame azimuth，-y / x
            float phi = (float) std::asin(thisPoint.z / range); //lidar frame: phi， z/r
            int index = columnIdn  + rowIdn * cols_; //index
            float num_p = static_cast<float>(num_per_pixel[index]);
            float num_p_new = num_p + 1;
            cos_theta.at<float>(rowIdn, columnIdn) = (num_p * cos_theta.at<float>(rowIdn, columnIdn) + std::cos(theta)) / num_p_new;
            sin_theta.at<float>(rowIdn, columnIdn) = (num_p * sin_theta.at<float>(rowIdn, columnIdn) + std::sin(theta)) / num_p_new;
            cos_phi.at<float>(rowIdn, columnIdn) = (num_p * cos_phi.at<float>(rowIdn, columnIdn) + std::cos(phi)) / num_p_new;
            sin_phi.at<float>(rowIdn, columnIdn) = (num_p * sin_phi.at<float>(rowIdn, columnIdn) + std::sin(phi)) / num_p_new;
            sin_phi_inv.at<float>(rowIdn, columnIdn) = (num_p * sin_phi_inv.at<float>(rowIdn, columnIdn) + 1 / std::sin(phi)) / num_p_new;

            ++num_per_pixel[index];
        }

        ++num_cloud;
        //todo compute M matrix and M inverse

        if (num_cloud < min_table_cloud_)
            return false;
        else{
            return true;
        }

    }


    void computeNormalsFromdepth(const cv::Mat &r, const cv::_OutputArray &normals_out) 
    {
        // Get the normals
        normals_out.create(rows_, cols_, CV_MAKETYPE(depth_, 3));


        cv::Mat normals = normals_out.getMat();
    //    cv::Mat res = residual.getMat();
        computeFromdepth(r, normals);
    }

    void computeNormalsFromheight(const cv::Mat &h, const cv::_OutputArray &normals_out) 
    {
        // Get the normals
        normals_out.create(rows_, cols_, CV_MAKETYPE(depth_, 3));


        cv::Mat normals = normals_out.getMat();
    //    cv::Mat res = residual.getMat();
        computeFromheight(h, normals);
    }

    void computeFromdepth(const cv::Mat &r, cv::Mat & normals) const
    {
        // Compute B
        cv::Mat_<Vec3T> B(rows_, cols_);

        const float* row_r = r.ptr < float > (0), *row_r_end = row_r + rows_ * cols_;
        const Vec3T *row_V = V_depth[0];
        Vec3T *row_B = B[0];
        for (; row_r != row_r_end; ++row_r, ++row_B, ++row_V)
        {
            if (*row_r==FLT_MAX)
                *row_B = Vec3T();
            else
                *row_B = (*row_V) / (*row_r); //v_i / r_i
        }

        ///todo BorderTypes::BORDER_TRANSPARENT, error
        // Apply a box filter to B
        boxFilter(B, B, B.depth(), cv::Size(window_size_, window_size_), cv::Point(-1, -1), false);

        // compute the Minv*B products
        row_r = r.ptr < float > (0);
        const Vec3T * B_vec = B[0];
        const Mat33T * M_inv = reinterpret_cast<const Mat33T *>(M_inv_depth.ptr(0));
        Vec3T *normal = normals.ptr<Vec3T>(0);
        for (; row_r != row_r_end; ++row_r, ++B_vec, ++normal, ++M_inv) {
            if (*row_r==FLT_MAX) {
                (*normal)[0] = *row_r;
                (*normal)[1] = *row_r;
                (*normal)[2] = *row_r;
            } else {
                Mat33T Mr = *M_inv;
                Vec3T Br = *B_vec;
                Vec3T MBr(Mr(0, 0) * Br[0] + Mr(0, 1) * Br[1] + Mr(0, 2) * Br[2],
                          Mr(1, 0) * Br[0] + Mr(1, 1) * Br[1] + Mr(1, 2) * Br[2],
                          Mr(2, 0) * Br[0] + Mr(2, 1) * Br[1] + Mr(2, 2) * Br[2]);
//                signNormal(MBr, *normal);
                normalizedNormal(MBr, *normal);
            }
        }
    }


    void computeFromheight(const cv::Mat &h, cv::Mat & normals) const
    {
        // Compute B
        cv::Mat_<Vec3T> B(rows_, cols_);

        const float* row_h = h.ptr < float > (0), *row_h_end = row_h + rows_ * cols_;
        const Vec3T *row_V = V_height[0];
        Vec3T *row_B = B[0];
        for (; row_h != row_h_end; ++row_h, ++row_B, ++row_V)
        {
            if (*row_h==FLT_MAX)
                *row_B = Vec3T();
            else
                *row_B = (*row_V) / (*row_h); //v_i / r_i
        }

        ///todo BorderTypes::BORDER_TRANSPARENT, error
        // Apply a box filter to B
        boxFilter(B, B, B.depth(), cv::Size(window_size_, window_size_), cv::Point(-1, -1), false);

        // compute the Minv*B products
        row_h = h.ptr < float > (0);
        const Vec3T * B_vec = B[0];
        const Mat33T * M_inv = reinterpret_cast<const Mat33T *>(M_inv_height.ptr(0));
        Vec3T *normal = normals.ptr<Vec3T>(0);

        for (; row_h != row_h_end; ++row_h, ++B_vec, ++normal, ++M_inv) {
            if (*row_h==FLT_MAX) {
                (*normal)[0] = *row_h;
                (*normal)[1] = *row_h;
                (*normal)[2] = *row_h;
            } else {
                Mat33T Mr = *M_inv;
                Vec3T Br = *B_vec;
                Vec3T MBr(Mr(0, 0) * Br[0] + Mr(0, 1) * Br[1] + Mr(0, 2) * Br[2],
                          Mr(1, 0) * Br[0] + Mr(1, 1) * Br[1] + Mr(1, 2) * Br[2],
                          Mr(2, 0) * Br[0] + Mr(2, 1) * Br[1] + Mr(2, 2) * Br[2]);
//                signNormal(MBr, *normal);
                normalizedNormal(MBr, *normal);
            }
        }

    }


    void computeMInverseFromdepth()
    {
        std::vector<cv::Mat> channels(3);
        channels[0] = cos_phi.mul(cos_theta);
        channels[1] = -cos_phi.mul(sin_theta);
        channels[2] = sin_phi;
        merge(channels, V_depth);

        // Compute M
        cv::Mat_<Vec9T> M(rows_, cols_);
        Mat33T VVt;
        const Vec3T * vec = V_depth[0];
        Vec9T * M_ptr = M[0], *M_ptr_end = M_ptr + rows_ * cols_;
        for (; M_ptr != M_ptr_end; ++vec, ++M_ptr)
        {
            VVt = (*vec) * vec->t(); //v * v_t
            *M_ptr = Vec9T(VVt.val);
        }

        ///todo BorderTypes::BORDER_TRANSPARENT, error
        int border_type = cv::BorderTypes::BORDER_TRANSPARENT;
        boxFilter(M, M, M.depth(), cv::Size(window_size_, window_size_), cv::Point(-1, -1), false);

        // Compute M's inverse
        Mat33T M_inv;
        M_inv_depth.create(rows_, cols_);
        Vec9T * M_inv_ptr = M_inv_depth[0];
        for (M_ptr = &M(0); M_ptr != M_ptr_end; ++M_inv_ptr, ++M_ptr)
        {
            // We have a semi-definite matrix
            invert(Mat33T(M_ptr->val), M_inv, cv::DECOMP_CHOLESKY);
            *M_inv_ptr = Vec9T(M_inv.val);
        }
    }



    void computeMInverseFromheight()
    {
        std::vector<cv::Mat> channels(3);
        channels[0] = cos_phi.mul(cos_theta);
        channels[1] = -cos_phi.mul(sin_theta);
        channels[2] = Ones_mat;

        channels[0] = channels[0].mul(sin_phi_inv);
        channels[1] = channels[0].mul(sin_phi_inv);

        merge(channels, V_height);

        // Compute M
        cv::Mat_<Vec9T> M(rows_, cols_);
        Mat33T VVt;
        const Vec3T * vec = V_height[0];
        Vec9T * M_ptr = M[0], *M_ptr_end = M_ptr + rows_ * cols_;
        for (; M_ptr != M_ptr_end; ++vec, ++M_ptr)
        {
            VVt = (*vec) * vec->t(); //v * v_t
            *M_ptr = Vec9T(VVt.val);
        }

        ///todo BorderTypes::BORDER_TRANSPARENT, error
        int border_type = cv::BorderTypes::BORDER_TRANSPARENT;
        boxFilter(M, M, M.depth(), cv::Size(window_size_, window_size_), cv::Point(-1, -1), false);

        // Compute M's inverse
        Mat33T M_inv;
        M_inv_height.create(rows_, cols_);
        Vec9T * M_inv_ptr = M_inv_height[0];
        for (M_ptr = &M(0); M_ptr != M_ptr_end; ++M_inv_ptr, ++M_ptr)
        {
            // We have a semi-definite matrix
            invert(Mat33T(M_ptr->val), M_inv, cv::DECOMP_CHOLESKY);
            *M_inv_ptr = Vec9T(M_inv.val);
        }
    }

    void saveMInverseFromdepth(std::string dir, std::string filename)
    {
        if (access(dir.c_str(), 0)) {
            printf("folder dose not exist. Creating a new folder:\n%s\n", dir.c_str());
            if(mkdir(dir.c_str(), 0771) == 0)
                printf("Created successfully.\n");
            else {
                printf("Failed to create, EXIT.\n");
                return;
            }
        }

        std::vector<cv::Mat> mats(9);
        for (int i = 0; i < 9; ++i) {
            mats[i].create(M_inv_depth.rows, M_inv_depth.cols, CV_32F);
        }
        cv::Mat mat_vv(M_inv_depth.rows, M_inv_depth.cols, CV_32FC3);

        for (int i = 0; i < M_inv_depth.rows; ++i) {
            for (int j = 0; j < M_inv_depth.cols; ++j) {
                const Vec9T& tmp(M_inv_depth.at<Vec9T>(i, j));
                for (int k = 0; k < 9; ++k) {
                    mats[k].at<float>(i, j) = tmp(k);
                }
                mat_vv.at<Vec3T>(i, j) = V_depth.at<Vec3T>(i, j);
            }
        }

        //save M inverse
        for (int i = 0; i < 9; ++i) {
            std::string file_id(std::to_string(i));
            cv::FileStorage fs(dir + "/" + filename + "_M_depth_" + file_id + ".xml", cv::FileStorage::WRITE);
            fs << filename + "_" + file_id << mats[i];
            fs.release();
        }
        //save v v_t
        cv::FileStorage fs(dir + "/" + filename + "_v_depth" +  + ".xml", cv::FileStorage::WRITE);
        fs << filename + "_vv"  << mat_vv;
        fs.release();
    }

    void saveMInverseFromheight(std::string dir, std::string filename)
    {
        if (access(dir.c_str(), 0)) {
            printf("folder dose not exist. Creating a new folder:\n%s\n", dir.c_str());
            if(mkdir(dir.c_str(), 0771) == 0)
                printf("Created successfully.\n");
            else {
                printf("Failed to create, EXIT.\n");
                return;
            }
        }

        std::vector<cv::Mat> mats(9);
        for (int i = 0; i < 9; ++i) {
            mats[i].create(M_inv_height.rows, M_inv_height.cols, CV_32F);
        }
        cv::Mat mat_vv(M_inv_height.rows, M_inv_height.cols, CV_32FC3);

        for (int i = 0; i < M_inv_height.rows; ++i) {
            for (int j = 0; j < M_inv_height.cols; ++j) {
                const Vec9T& tmp(M_inv_height.at<Vec9T>(i, j));
                for (int k = 0; k < 9; ++k) {
                    mats[k].at<float>(i, j) = tmp(k);
                }
                mat_vv.at<Vec3T>(i, j) = V_height.at<Vec3T>(i, j);
            }
        }

        //save M inverse
        for (int i = 0; i < 9; ++i) {
            std::string file_id(std::to_string(i));
            cv::FileStorage fs(dir + "/" + filename + "_M_height_" + file_id + ".xml", cv::FileStorage::WRITE);
            fs << filename + "_" + file_id << mats[i];
            fs.release();
        }
        //save v v_t
        cv::FileStorage fs(dir + "/" + filename + "_v_height" +  + ".xml", cv::FileStorage::WRITE);
        fs << filename + "_vv"  << mat_vv;
        fs.release();
    }



    bool loadMInverseFromdepth(std::string dir, std::string filename)
    {
        std::vector<cv::Mat> mats(9);
        for (int i = 0; i < 9; ++i) {
            mats[i].create(rows_, cols_, CV_32F);
        }

        for (int i = 0; i < 9; ++i) {
            std::string file_id(std::to_string(i));
            cv::FileStorage fs(dir + "/" + filename + "_M_depth_" + file_id + ".xml", cv::FileStorage::READ);
            if(!fs.isOpened())
            {
//                std::cerr << "ERROR: Wrong path to ring normal M file" << std::endl;
                return false;
            }
            fs[filename + "_" + file_id] >> mats[i];
            fs.release();
        }

        M_inv_depth.create(rows_, cols_);

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                Vec9T& tmp(M_inv_depth.at<Vec9T>(i, j));
                for (int k = 0; k < 9; ++k) {
                    tmp(k) = mats[k].at<float>(i, j);
                }
            }
        }

        V_depth.create(rows_, cols_);
        cv::FileStorage fs(dir + "/" + filename + "_v_depth" +  + ".xml", cv::FileStorage::READ);
        fs[filename + "_vv"] >> V_depth;
        fs.release();
        ROS_INFO("load depth table success.");

        return true;
    }

    bool loadMInverseFromheight(std::string dir, std::string filename)
    {
        std::vector<cv::Mat> mats(9);
        for (int i = 0; i < 9; ++i) {
            mats[i].create(rows_, cols_, CV_32F);
        }

        for (int i = 0; i < 9; ++i) {
            std::string file_id(std::to_string(i));
            cv::FileStorage fs(dir + "/" + filename + "_M_height_" + file_id + ".xml", cv::FileStorage::READ);
            if(!fs.isOpened())
            {
//                std::cerr << "ERROR: Wrong path to ring normal M file" << std::endl;
                return false;
            }
            fs[filename + "_" + file_id] >> mats[i];
            fs.release();
        }

        M_inv_height.create(rows_, cols_);

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                Vec9T& tmp(M_inv_height.at<Vec9T>(i, j));
                for (int k = 0; k < 9; ++k) {
                    tmp(k) = mats[k].at<float>(i, j);
                }
            }
        }

        V_height.create(rows_, cols_);
        cv::FileStorage fs(dir + "/" + filename + "_v_height" +  + ".xml", cv::FileStorage::READ);
        fs[filename + "_vv"] >> V_height;
        fs.release();
        ROS_INFO("load height table success.");

        return true;
    }

};
