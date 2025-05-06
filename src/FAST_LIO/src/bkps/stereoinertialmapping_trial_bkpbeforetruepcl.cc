/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include<sensor_msgs/Imu.h>

#include<opencv2/core/core.hpp>

#include"ORB_SLAM3/include/System.h"
#include"ORB_SLAM3/include/ImuTypes.h"

////////////////////////////////////////////////////////////////
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver2/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <unordered_map>
#include <cmath>
#include <boost/array.hpp>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unordered_map>
////////////////////////////////////////////////////////////////////
// Define a custom output function for boost::array<double, 3>
std::ostream& operator<<(std::ostream& os, const boost::array<double, 3>& arr) {
    os << "[" << arr[0] << ", " << arr[1] << ", " << arr[2] << "]";
    return os;
}

// Custom hash function for double type
struct DoubleHash {
    std::size_t operator()(const double& d) const {
        return std::hash<double>()(d);
    }
};


struct DataEntry {
    std::vector<Eigen::Vector3d> mvl_WorldCoords;
    std::vector<Eigen::Matrix<double, 2, 1>> mvl_observations;
    std::vector<Eigen::Vector3d> mvr_WorldCoords;
    std::vector<Eigen::Matrix<double, 2, 1>> mvr_observations;
    double mean;
    Sophus::SE3d Tiwi;
    std::vector<double> mvl_invSigma2s;
    state_ikfom prior_s;
    bool frame_or_not;
};


class DataStore {
private:
    std::unordered_map<double, DataEntry, DoubleHash> dataMap; // Using unordered_map for faster access

public:
    void addData(double timestamp, const DataEntry& entry) {
        dataMap[timestamp] = entry;
    }

    const DataEntry& getData(double timestamp) const {
        return dataMap.at(timestamp);
    }

    bool hasData(double timestamp) const {
        return dataMap.find(timestamp) != dataMap.end();
    }
};


using namespace std;
bool bVIBA0;
bool bVIBA1;
bool bVIBA2;
bool bTracker_ikdtree;
bool blidar_processed;
bool bIMU_inited;
Sophus::SE3d Tyc0;
Sophus::SE3d Tc0y;
Sophus::SE3d Tycn;
Sophus::SE3d Tc0cn;
std::string voc_path, settings_path, do_rectify;

//////////////////////////////////////////////////////////////////////////////////
typedef boost::array<double, 3> mainvect3;
mainvect3 globalposition = {0, 0, 0};
float INIT_TIME = 0.1;
float LASER_POINT_COV = 5;
// int MAXN = 720000;
// define the MAXN to be constant 720000
const int MAXN = 720000;
int PUBFRAME_PERIOD = 20;

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
int iterate_num = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

ofstream fout_pre, fout_out, fout_dbg;
FILE *fp;
vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

// define a deque of double to store the time of the lidar to find the initialization time of vslam system
deque<double>                     time_buffer_vslam;
// define a vector of double of 5 to store the time
vector<double>                    time_vector_vslam;
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures_;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;


double frame_time;
// create a queue to store the frame_time
std::deque<double> frame_time_queue;
std::deque<double> lidar_time_queue;
std::deque<double> lidar_time_queue_for_vslam;
// create corresponding mutex for the frame_time_queue
std::mutex frame_time_queue_mutex;
std::mutex lidar_time_queue_mutex;
std::mutex lidar_time_queue_for_vslam_mutex;

std::deque<std::string> frame_time_queue_str;
std::deque<std::string> lidar_time_queue_str;
double frame_time_g;
void insertSorted(std::deque<double>& timestamps, double newTimestamp) {
    // Find the correct position to insert the new timestamp to keep the deque sorted
    auto it = std::lower_bound(timestamps.begin(), timestamps.end(), newTimestamp);
    // Insert the new timestamp at the found position
    timestamps.insert(it, newTimestamp);
}

DataStore dataStore;

// std::vector<Eigen::Vector3d> mvl_WorldCoords;
// std::vector<Eigen::Matrix<double, 2, 1>> mvl_observations;
// std::vector<Eigen::Vector3d> mvr_WorldCoords;
// std::vector<Eigen::Matrix<double, 2, 1>> mvr_observations;
DataEntry globalMeasures;
std::vector<double> mvl_invSigma2s;
std::vector<double> mvr_invSigma2s;

std::vector<double> mv_invSigma2s;
Sophus::SE3f Tcw_f;
Sophus::SE3d Tcleft;
float Parameter0, Parameter1, Parameter2, Parameter3, Parameter4, Parameter5, Parameter6, Parameter7;
Sophus::SE3d mTrl;

Eigen::Matrix3d rotation;
Eigen::Vector3d translation;
Eigen::Matrix3d rotation_lr;
Eigen::Vector3d translation_lr;
Eigen::Matrix3d rotation_imu_left;
Eigen::Matrix3d rotation_imu_right;
Eigen::Vector3d translation_imu_left;
Eigen::Vector3d translation_imu_right;

Sophus::SE3d Tbleft;
Sophus::SE3d Tleftright;
Sophus::SE3d Tleftb;
Sophus::SE3d Trightb;
Sophus::SE3d Tbright;
Sophus::SE3d Trightleft;
Sophus::SE3d Tcw;
Sophus::SE3d Twc;
std::vector<Eigen::Vector3d> mvl_WorldCoords_imucoor;

std::vector<Eigen::Vector3d> mvr_WorldCoords_imucoor;
Eigen::Matrix3d Tleftb_rotation;

Sophus::SE3d Twi;
Sophus::SE3d Tiwi;
shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());
float thHuberMono;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}


inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures_.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}


void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}


void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}


void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}


BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}


void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    double lidarendtime;
    lidarendtime = last_timestamp_lidar + ptr->points.back().curvature / double(1000);
    time_buffer.push_back(last_timestamp_lidar);
    {
        std::lock_guard<std::mutex> lock(lidar_time_queue_mutex);
        // insertSorted(time_queue, lidarendtime);
        lidar_time_queue.push_back(lidarendtime);
        
    }

    {
        std::lock_guard<std::mutex> lock(lidar_time_queue_for_vslam_mutex);
        lidar_time_queue_for_vslam.push_back(lidarendtime);
    }

    lidar_time_queue_str.push_back(std::to_string(lidarendtime));
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    DataEntry Measure_lidar;
    Measure_lidar.frame_or_not = false;
    dataStore.addData(lidarendtime, Measure_lidar);

}


void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    // sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    // if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    // {
    //     msg->header.stamp = \
    //     ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    // }

    double timestamp = msg_in->header.stamp.toSec();
    std::string imu_time_str = std::to_string(timestamp);
    // std::cout<<"IMU got at: "<<imu_time_str<<endl;

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg_in);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages_stereo(MeasureGroup &meas, double frame_time_)
{
    if (imu_buffer.empty()) {
        std::cout << "IMU buffer empty" << std::endl;
        return false;
    }
    /*** push a lidar scan ***/

    // if (last_timestamp_imu < frame_time)
    // {
    //     return false;
    // }
    std::cout << "IMU time 0 " << std::endl;
    meas.lidar_beg_time = frame_time_;
    meas.lidar_end_time = frame_time_;
    if(last_timestamp_imu < frame_time_)
    {
        std::cout << "last IMU time < frame time" << std::endl;
        return false;
    }
    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    std::string imu_time_str = std::to_string(imu_time);
    std::cout << "IMU time front: " << imu_time_str << std::endl;
    std::cout << "IMU time 1 " << std::endl;
    if(imu_time > frame_time_){
        std::cout << "IMU time > frame time" << std::endl;
        // pop the imu data

        // imu_buffer.pop_front();
        return false;
    }
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < frame_time_))
    {
        // std::cout << "IMU time: " << imu_time << std::endl;
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > frame_time_) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    std::cout << "IMU time 2 " << std::endl;
    return true;
}

bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());

        std::string imu_time_str = std::to_string(imu_time);
        // std::cout << "IMU time: " << imu_time_str << std::endl;
        std::string lidar_end_time_str = std::to_string(lidar_end_time);
        // std::cout << "Lidar end time: " << lidar_end_time_str << std::endl;
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}


int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}


PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}


void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}


void h_share_model_stereo_prior(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    if(globalMeasures.frame_or_not)
    {
        // this function should use the input of visual map points, camera pose, then calculate the corresponding Jacobian matrix H and residual vector
        bool activaterobust;
        std::vector<Eigen::Matrix<double, 2, 1>> errors;
        std::vector<Eigen::Matrix<double, 2, 6>> jacobians;
        activaterobust = true;
        iterate_num ++;

        int total_observations = globalMeasures.mvl_observations.size() + globalMeasures.mvr_observations.size();

        ekfom_data.h_x = MatrixXd::Identity(23 , 23); //23
        ekfom_data.h.resize(23);
    }
    else{

        double match_start = omp_get_wtime();
        laserCloudOri->clear(); 
        corr_normvect->clear(); 
        total_residual = 0.0; 

        /** closest surface search and residual computation **/
        #ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
        #endif
        for (int i = 0; i < feats_down_size; i++)
        {
            PointType &point_body  = feats_down_body->points[i]; 
            PointType &point_world = feats_down_world->points[i]; 

            /* transform to world frame */
            V3D p_body(point_body.x, point_body.y, point_body.z);
            V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
            point_world.x = p_global(0);
            point_world.y = p_global(1);
            point_world.z = p_global(2);
            point_world.intensity = point_body.intensity;

            vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

            auto &points_near = Nearest_Points[i];

            if (ekfom_data.converge)
            {
                /** Find the closest surfaces in the map **/
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
            }

            if (!point_selected_surf[i]) continue;

            VF(4) pabcd;
            point_selected_surf[i] = false;
            if (esti_plane(pabcd, points_near, 0.1f))
            {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                if (s > 0.9)
                {
                    point_selected_surf[i] = true;
                    normvec->points[i].x = pabcd(0);
                    normvec->points[i].y = pabcd(1);
                    normvec->points[i].z = pabcd(2);
                    normvec->points[i].intensity = pd2;
                    res_last[i] = abs(pd2);
                }
            }
        }
        
        effct_feat_num = 0;

        for (int i = 0; i < feats_down_size; i++)
        {
            if (point_selected_surf[i])
            {
                laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                corr_normvect->points[effct_feat_num] = normvec->points[i];
                total_residual += res_last[i];
                effct_feat_num ++;
            }
        }

        if (effct_feat_num < 1)
        {
            ekfom_data.valid = false;
            ROS_WARN("No Effective Points! \n");
            return;
        }

        res_mean_last = total_residual / effct_feat_num;
        match_time  += omp_get_wtime() - match_start;
        double solve_start_  = omp_get_wtime();
        
        /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
        ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
        ekfom_data.h.resize(effct_feat_num);

        for (int i = 0; i < effct_feat_num; i++)
        {
            const PointType &laser_p  = laserCloudOri->points[i];
            V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
            M3D point_be_crossmat;
            point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
            V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
            M3D point_crossmat;
            point_crossmat<<SKEW_SYM_MATRX(point_this);

            /*** get the normal vector of closest surface/corner ***/
            const PointType &norm_p = corr_normvect->points[i];
            V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

            /*** calculate the Measuremnt Jacobian matrix H ***/
            V3D C(s.rot.conjugate() *norm_vec);
            V3D A(point_crossmat * C);
            if (extrinsic_est_en)
            {
                V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
                ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            }
            else
            {
                ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }

            /*** Measuremnt: distance to the closest surface/corner ***/
            ekfom_data.h(i) = -norm_p.intensity;
        }
        solve_time += omp_get_wtime() - solve_start_;
    }


}


class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bRect, const bool bClahe): mpSLAM(pSLAM), mpImuGb(pImuGb), do_rectify(bRect), mbClahe(bClahe){}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft,mBufMutexRight;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "Stereo_Inertial");
  ros::start();
  ros::NodeHandle no;
  // retrieve the vocabulary file path by ros parameter
  blidar_processed = false;
  no.param("ORB_SLAM3/VocabularyPath",voc_path,std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Vocabulary/ORBvoc.txt"));
  // retrieve the settings file path by ros parameter
  no.param("ORB_SLAM3/SettingsPath",settings_path,std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Examples/Stereo-Inertial/3001.yaml"));
  no.param("ORB_SLAM3/ImuStereo/DoRectify",do_rectify,std::string("false"));
  //////////////////////////////////////////////////////////////////////////
  no.param<bool>("publish/path_en",path_en, true);
  no.param<bool>("publish/scan_publish_en",scan_pub_en, true);
  no.param<bool>("publish/dense_publish_en",dense_pub_en, true);
  no.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
  no.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
  no.param<string>("map_file_path",map_file_path,"");
  no.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
  no.param<string>("common/imu_topic", imu_topic,"/livox/imu");
  no.param<bool>("common/time_sync_en", time_sync_en, false);
  no.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
  no.param<double>("filter_size_corner",filter_size_corner_min,0.5);
  no.param<double>("filter_size_surf",filter_size_surf_min,0.5);
  no.param<double>("filter_size_map",filter_size_map_min,0.5);
  no.param<double>("cube_side_length",cube_len,200);
  no.param<float>("mapping/det_range",DET_RANGE,300.f);
  no.param<double>("mapping/fov_degree",fov_deg,180);
  no.param<double>("mapping/gyr_cov",gyr_cov,0.1);
  no.param<double>("mapping/acc_cov",acc_cov,0.1);
  no.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
  no.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
  no.param<double>("preprocess/blind", p_pre->blind, 0.01);
  no.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  no.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
  no.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
  no.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
  no.param<int>("point_filter_num", p_pre->point_filter_num, 2);
  no.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
  no.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
  no.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
  no.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  no.param<int>("pcd_save/interval", pcd_save_interval, -1);
  no.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
  no.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
  
  path.header.stamp    = ros::Time::now();
  path.header.frame_id ="camera_init";
  thHuberMono = sqrt(5.991);
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  bool bEqual = false;
  bVIBA2  = false;
  /*** variables definition ***/
  int effect_feat_num = 0, frame_num = 0;
  double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
  bool flg_EKF_converged, EKF_stop_flg = 0;
  
  FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
  HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

  _featsArray.reset(new PointCloudXYZI());

  memset(point_selected_surf, true, sizeof(point_selected_surf));
  memset(res_last, -1000.0f, sizeof(res_last));
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
  memset(point_selected_surf, true, sizeof(point_selected_surf));
  memset(res_last, -1000.0f, sizeof(res_last));

  Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

  double epsi[23] = {0.001};
  fill(epsi, epsi+23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model_stereo_prior, NUM_MAX_ITERATIONS, epsi);

  rotation_lr << 0.9999958, -0.0019376, -0.0021683,
              0.0019764,  0.9998351,  0.0180507,
              0.0021330, -0.0180549,  0.9998347;

  translation_lr<<0.0916374166332839, 0.00045393955462707007, 0.0005929167382999913;

  rotation_imu_left << 0.999956, -0.00617887, -0.00706637,
                      -0.00613937, -0.999965, 0.00559736,
                      -0.00710071, -0.00555373, -0.999959;
  rotation_imu_right << 0.99992443596803171, -0.0079877673237543148, -0.0093449548694623016,
                      -0.0081037536730824422,   -0.99988973281600624,  -0.012440327622059263,
                      -0.0092445542408392117,   0.012515117766320152,   -0.99987892479535034;


  translation_imu_left << -0.0445156, -0.0232076, -0.280637;
  translation_imu_right << 0.047110744386894318, -0.024220757567199817, -0.28188358029462574;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation_imu_left, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R_orthogonal = svd.matrixU() * svd.matrixV().transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd_lr(rotation_lr, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R_orthogonal_lr = svd_lr.matrixU() * svd_lr.matrixV().transpose();

  Tbleft = Sophus::SE3d(R_orthogonal, translation_imu_left);
  Tleftright = Sophus::SE3d(R_orthogonal_lr, translation_lr);
  Tleftb = Tbleft.inverse();
  Trightleft = Tleftright.inverse();
  Tbright = Tbleft * Tleftright;
  Trightb = Tbright.inverse();
  // Trightb = Tbright.inverse();
  /*** debug record ***/
  FILE *fp;
  string pos_log_dir = root_dir + "/Log/pos_log.txt";
  fp = fopen(pos_log_dir.c_str(),"w");


  ofstream fout_pre, fout_out, fout_dbg;
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
  fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
  if (fout_pre && fout_out)
  {
    cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
  }
  else
  {
    cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;
  }

  ros::Subscriber sub_pcl = no.subscribe(lid_topic, 200000, livox_pcl_cbk);
  ros::Subscriber sub_imu_lidar = no.subscribe(imu_topic, 200000, imu_cbk);
  ros::Publisher pubLaserCloudFull = no.advertise<sensor_msgs::PointCloud2>
          ("/cloud_registered", 100000);
  ros::Publisher pubLaserCloudFull_body = no.advertise<sensor_msgs::PointCloud2>
          ("/cloud_registered_body", 100000);
  ros::Publisher pubLaserCloudEffect = no.advertise<sensor_msgs::PointCloud2>
          ("/cloud_effected", 100000);
  ros::Publisher pubLaserCloudMap = no.advertise<sensor_msgs::PointCloud2>
          ("/Laser_map", 100000);
  ros::Publisher pubOdomAftMapped = no.advertise<nav_msgs::Odometry> 
          ("/Odometry", 100000);
  ros::Publisher pubPath          = no.advertise<nav_msgs::Path> 
          ("/path", 100000);
  // if(argc < 4 || argc > 5)
  // {
  //   std::cout << argc << std::endl;
  //   cerr << endl << "Usage: rosrun ORB_SLAM3 Stereo_Inertial path_to_vocabulary path_to_settings do_rectify [do_equalize]" << endl;
  //   ros::shutdown();
  //   return 1;
  // }

  // std::string sbRect(argv[3]);
  // if(argc==5)
  // {
  //   std::string sbEqual(argv[4]);
  //   if(do_rectify == "true")
  //     bEqual = true;
  // }

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM3::System SLAM(voc_path, settings_path,ORB_SLAM3::System::IMU_STEREO,true);

  ImuGrabber imugb;
  ImageGrabber igb(&SLAM,&imugb,do_rectify == "true",bEqual);
  
    if(igb.do_rectify)
    {      
        // std::cout<<"Rectifying images..."<<std::endl;
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,igb.M1l,igb.M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,igb.M1r,igb.M2r);
    }

  // Maximum delay, 5 seconds
  // ros::Subscriber sub_imu = n.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb); 
  // ros::Subscriber sub_img_left = n.subscribe("/camera/left/image_raw", 100, &ImageGrabber::GrabImageLeft,&igb);
  // ros::Subscriber sub_img_right = n.subscribe("/camera/right/image_raw", 100, &ImageGrabber::GrabImageRight,&igb);
  ros::Subscriber sub_imu = no.subscribe("/livox/imu", 1000, &ImuGrabber::GrabImu, &imugb); 
  ros::Subscriber sub_img_left = no.subscribe("/uas_cam2/image", 100, &ImageGrabber::GrabImageLeft,&igb);
  ros::Subscriber sub_img_right = no.subscribe("/uas_cam4/image", 100, &ImageGrabber::GrabImageRight,&igb);

  std::thread sync_thread(&ImageGrabber::SyncWithImu,&igb);
  signal(SIGINT, SigHandle);
  ros::Rate rate(5000);
  // ros::spin();
  bool status = ros::ok();
  bool data_ready = true;
  while(status)
  {
    if (flg_exit) break;
    ros::spinOnce();
    
    {
        
        // if(!time_queue.empty() && data_ready)
        // {
            
        //     {
        //         std::lock_guard<std::mutex> lock(time_queue_mutex);
        //         frame_time_g = time_queue.front();
        //         time_queue.pop_front();
        //     }
        // }
        // else if(time_queue.empty() && data_ready)
        // {
        //     continue;
        // }
        // else if(!data_ready)
        // {
        //     frame_time_g = frame_time_g;
        // }
        
        if(!frame_time_queue.empty() && !lidar_time_queue.empty())
        {
            double frame_time = frame_time_queue.front();
            double lidar_time = lidar_time_queue.front();
            if (frame_time < lidar_time)
            {
                frame_time_g = frame_time;
                {
                    std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                    frame_time_queue.pop_front();
                }
                
            }
            else
            {
                frame_time_g = lidar_time;
                {
                    std::lock_guard<std::mutex> lock(lidar_time_queue_mutex);
                    lidar_time_queue.pop_front();
                }
            }
            std::string frame_time_str = std::to_string(frame_time_g);
            std::cout << "frame_timestr_ = " << frame_time_str << std::endl;
        }
        else
        {
            continue;
        }
    }
    while(!dataStore.hasData(frame_time_g))
    {
        ros::spinOnce();
        // std::cout << "waiting for data" << std::endl;
        continue;
    }
    data_ready = true;
    std::string frame_time_str1 = std::to_string(frame_time_g);
    std::cout<<"syncing frame_time = "<<frame_time_str1<<std::endl;
    globalMeasures = dataStore.getData(frame_time_g);
    std::cout << "globalMeasures.frame_or_not = " << globalMeasures.frame_or_not << std::endl;
    Eigen::Matrix3d Riwi;
    Eigen::Vector3d tiwi;
    if(globalMeasures.frame_or_not)
    {
         std::string frame_time_str = std::to_string(frame_time_g);
         std::cout << " syncing frame_time = " << frame_time_str << std::endl;
         if(sync_packages_stereo(Measures_, frame_time_g))
         {
             cout << "sync_packages_stereo(Meausures, frame_time) succeed" << endl;
             if(flg_first_scan)
             {
                 flg_first_scan = false;
                 p_imu->first_lidar_time = frame_time_g;
                 continue;
             }
             Riwi = globalMeasures.Tiwi.so3().matrix();
             tiwi = globalMeasures.Tiwi.translation();
             double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;
             match_time = 0;
             kdtree_search_time = 0.0;
             solve_time = 0;
             solve_const_H_time = 0;
             svd_time   = 0;
             t0 = omp_get_wtime();
             p_imu->Process_stereo(Measures_, kf);
             t2 = omp_get_wtime();
             state_point = kf.get_x();
             euler_cur = SO3ToEuler(state_point.rot);
             MTK::SubManifold<MTK::vect<3, double>, 0, 0> state_pos = state_point.pos;
             boost::array<double, 3> converted_pos;
             for (int i = 0; i < 3; ++i)
             {
                 converted_pos[i] = state_pos[i];
             }
             globalposition = converted_pos;
             Eigen::Quaterniond quaternion(Riwi);
             state_point.rot = MTK::SO3<double>(quaternion.coeffs());
             state_point.pos = tiwi;
             globalMeasures.prior_s = state_point;
             // kf.change_x(state_point);
     
             /*** iterated state estimation ***/
             double t_update_start = omp_get_wtime();
             double solve_H_time = 0;
             kf.update_iterated_dyn_share_modified_stereo(0.01, state_point, solve_H_time);
             state_point = kf.get_x();
             euler_cur = SO3ToEuler(state_point.rot);
             state_pos = state_point.pos;
             for (int i = 0; i < 3; ++i)
             {
                 converted_pos[i] = state_pos[i];
             }
             globalposition = converted_pos;
             pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
             geoQuat.x = state_point.rot.coeffs()[0];
             geoQuat.y = state_point.rot.coeffs()[1];
             geoQuat.z = state_point.rot.coeffs()[2];
             geoQuat.w = state_point.rot.coeffs()[3];
             publish_odometry(pubOdomAftMapped);
         }
      
    }
    else{
      std::string frame_time_str = std::to_string(frame_time_g);
      std::cout << " syncing lidar_time = " << frame_time_str << std::endl;
      if(sync_packages(Measures_))
      {
        blidar_processed = false;
        if (flg_first_scan)
        {
          first_lidar_time = Measures_.lidar_beg_time;
          p_imu->first_lidar_time = first_lidar_time;
          flg_first_scan = false;
          blidar_processed = true;
          continue;
        }
        double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;
        match_time = 0;
        kdtree_search_time = 0.0;
        solve_time = 0;
        solve_const_H_time = 0;
        svd_time   = 0;
        t0 = omp_get_wtime();
        p_imu->Process(Measures_, kf, feats_undistort);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
          ROS_WARN("No point, skip this scan!\n");
          blidar_processed = true;
          continue;
        }
        flg_EKF_inited = (Measures_.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                                  false : true;
        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        t1 = omp_get_wtime();
        feats_down_size = feats_down_body->points.size();
        /*** initialize the map kdtree ***/
        if(ikdtree.Root_Node == nullptr)
        {
          if(feats_down_size > 5)
          {
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size);
            for(int i = 0; i < feats_down_size; i++)
            {
              pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ikdtree.Build(feats_down_world->points);
          }
          blidar_processed = true;
          continue;
        }
        int featsFromMapNum = ikdtree.validnum();
        kdtree_size_st = ikdtree.size();
        /*** ICP and iterated Kalman filter update ***/
        if (feats_down_size < 5)
        {
          ROS_WARN("No enough points, skip this scan!\n");
          blidar_processed = true;
          continue;
        
        }
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
        fout_pre<<setw(20)<<Measures_.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
        <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;
        pointSearchInd_surf.resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        int  rematch_num = 0;
        bool nearest_search_en = true; //
        t2 = omp_get_wtime();
        /*** iterated state estimation ***/
        double t_update_start = omp_get_wtime();
        double solve_H_time = 0;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
        state_point = kf.get_x();
        euler_cur = SO3ToEuler(state_point.rot);
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];
        double t_update_end = omp_get_wtime();
        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped);
        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();
        map_incremental();
        t5 = omp_get_wtime();
        
        /******* Publish points *******/
        if (path_en)                         publish_path(pubPath);
        if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
        if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
        blidar_processed = true;
      }
    }
    // else{
    //     if(!bVIBA2)
    //     {
    //         data_ready = true;
    //         std::cout << "data not usable for fastlio" << std::endl;
    //     }
    //     else
    //     {
    //         data_ready = false;
    //         std::cout << "No data at this time" << std::endl;
    //         // sleep for 10ms

    //         continue;
    //     }
    // }
    status = ros::ok();
    rate.sleep();
  }
  // Stop all threads
  SLAM.Shutdown();
  // log out the strings in the frame_time_queue_str into a txt file
  std::ofstream frame_time_file;
  frame_time_file.open("/home/uas/a_files/frame_time.txt");
  for (int i = 0; i < frame_time_queue_str.size(); i++)
  {
    frame_time_file << frame_time_queue_str[i] << std::endl;
  }
  frame_time_file.close();
  std::ofstream lidar_time_file;
  lidar_time_file.open("/home/uas/a_files/lidar_time.txt");
  for (int i = 0; i < lidar_time_queue_str.size(); i++)
  {
    lidar_time_file << lidar_time_queue_str[i] << std::endl;
  }
  lidar_time_file.close();
  // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
  SLAM.SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");
  SLAM.SaveTrajectoryKITTI("FrameTrajectory_KITTI_Format.txt");
  ros::shutdown();

  return 0;
}



void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
  // std::cout<<"GrabImageLeft"<<std::endl;
  mBufMutexLeft.lock();
  if (!imgLeftBuf.empty())
    imgLeftBuf.pop();
  imgLeftBuf.push(img_msg);
  mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg)
{
  // std::cout<<"GrabImageRight"<<std::endl;
  mBufMutexRight.lock();
  if (!imgRightBuf.empty())
    imgRightBuf.pop();
  imgRightBuf.push(img_msg);
  mBufMutexRight.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  
  if(cv_ptr->image.type()==0)
  {
    return cv_ptr->image.clone();
  }
  else
  {
    std::cout << "Error type" << std::endl;
    return cv_ptr->image.clone();
  }
}

void ImageGrabber::SyncWithImu()
{

    const double maxTimeDiff = 0.01;
    while(1)
    {

        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        // if(imgLeftBuf.empty())
        // {
        //   std::cout << "imgLeftBuf empty" << std::endl;
        //   continue;
        // }
        // if(imgRightBuf.empty())
        // {
        //   std::cout << "imgRightBuf empty" << std::endl;
        //   continue;
        // }
        // if(mpImuGb->imuBuf.empty())
        // {
        //   std::cout << "imuBuf empty" << std::endl;
        //   continue;
        // }
        if (!imgLeftBuf.empty()&&!imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())
        {
            // std::cout<<"SyncWithImu"<<std::endl;
            tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            tImRight = imgRightBuf.front()->header.stamp.toSec();
            this->mBufMutexRight.lock();
            while((tImLeft-tImRight)>maxTimeDiff && imgRightBuf.size()>1)
            {
                imgRightBuf.pop();
                tImRight = imgRightBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRight.unlock();
            this->mBufMutexLeft.lock();
            while((tImRight-tImLeft)>maxTimeDiff && imgLeftBuf.size()>1)
            {
                imgLeftBuf.pop();
                tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexLeft.unlock();
            if((tImLeft-tImRight)>maxTimeDiff || (tImRight-tImLeft)>maxTimeDiff)
            {
                // std::cout << "big time difference" << std::endl;
                continue;
            }
            if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;
            this->mBufMutexLeft.lock();
            imLeft = GetImage(imgLeftBuf.front());
            frame_time = imgLeftBuf.front()->header.stamp.toSec();
            frame_time_queue_str.push_back(std::to_string(frame_time));
            {
                std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                // insertSorted(time_queue, frame_time);
                frame_time_queue.push_back(frame_time);
            }
            std::string frame_time_str = std::to_string(frame_time);
            std::cout << "frame_time_here = " << frame_time_str << std::endl;
            imgLeftBuf.pop();
            this->mBufMutexLeft.unlock();
            this->mBufMutexRight.lock();
            imRight = GetImage(imgRightBuf.front());
            imgRightBuf.pop();
            this->mBufMutexRight.unlock();
            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if(!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImLeft)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if(mbClahe)
            {
                mClahe->apply(imLeft,imLeft);
                mClahe->apply(imRight,imRight);
            }
            // std::cout<<"tImLeft: "<<std::endl;
            if(do_rectify)
            {
                cv::remap(imLeft,imLeft,M1l,M2l,cv::INTER_LINEAR);
                cv::remap(imRight,imRight,M1r,M2r,cv::INTER_LINEAR);
            }
            std::cout << "before TrackStereo" << std::endl;
            while(lidar_time_queue_for_vslam.empty())
            {
                std::cout << "waiting for lidar_time_queue_for_vslam" << std::endl;
                continue;
            }
            double last_lidar_time = lidar_time_queue_for_vslam.front();
            double second_lidar_time;
            std::cout << "last_lidar_time = " << std::to_string(last_lidar_time) << std::endl;
            if(lidar_time_queue_for_vslam.size() > 1)
            {
                second_lidar_time = lidar_time_queue_for_vslam[1];
                std::cout << "second_lidar_time = " << std::to_string(second_lidar_time) << std::endl;

                if(frame_time < last_lidar_time)
                {
                    // lidar_time_queue_for_vslam.pop_front();
                    std::cout << "frame_time is smaller than last_lidar_time" << std::endl;
                    Sophus::SE3f Tcw = mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
                    // if Tcw is no more identity, then we have a valid pose
                    if(!bTracker_ikdtree)
                    {
                        bTracker_ikdtree = true;
                        // get the pointer of the global variable ikdtree
                        auto ikdtree_ptr = &ikdtree;
                        mpSLAM->SetIKdtree(ikdtree_ptr);
                    }
                    if(bVIBA0 && !bVIBA2)
                    {
                        mpSLAM->CheckIKdtree_size();
                    }
                    Eigen::Matrix3d Rcw = Tcw.so3().matrix().cast<double>();
                    Eigen::Vector3d tcw = Tcw.translation().cast<double>();
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd_cw(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3d Rcw_orthogonal = svd_cw.matrixU() * svd_cw.matrixV().transpose();
                    // convert the Tcw to a SE3d type
                    Sophus::SE3d Tcw_(Rcw_orthogonal, tcw);
                    if(!bVIBA0)
                    {
                        bVIBA0 = mpSLAM->mbInertialBA0;
                        Twc = Tcw_.inverse();
                        Twi = Twc*Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && !bVIBA1)
                    {
                        bVIBA1 = mpSLAM->mbInertialBA1;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && bVIBA1 && !bVIBA2)
                    {
                            
                        bVIBA2 = mpSLAM->mbInertialBA2;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // } 
                    }
                    else if(bVIBA2)
                    {
                        std::cout << "bVIBA2: " << bVIBA2 << std::endl;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        std::string frame_time_str1 = std::to_string(frame_time);
                        std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                }
                else if(last_lidar_time <= frame_time && frame_time <= second_lidar_time)
                {
                    lidar_time_queue_for_vslam.pop_front();
                    std::cout << "frame_time is between last_lidar_time and second_lidar_time" << std::endl;
                    while(!blidar_processed)
                    {
                        std::cout << "waiting for the lidar to be processed" << std::endl;
                        continue;
                    }
                    Sophus::SE3f Tcw = mpSLAM->TrackStereo_PCL(imLeft,imRight,tImLeft,vImuMeas,"",feats_down_body);
                    // if Tcw is no more identity, then we have a valid pose
                    if(!bTracker_ikdtree)
                    {
                        bTracker_ikdtree = true;
                        // get the pointer of the global variable ikdtree
                        auto ikdtree_ptr = &ikdtree;
                        mpSLAM->SetIKdtree(ikdtree_ptr);
                    }
                    if(bVIBA0 && !bVIBA2)
                    {
                        mpSLAM->CheckIKdtree_size();
                    }
                    Eigen::Matrix3d Rcw = Tcw.so3().matrix().cast<double>();
                    Eigen::Vector3d tcw = Tcw.translation().cast<double>();
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd_cw(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3d Rcw_orthogonal = svd_cw.matrixU() * svd_cw.matrixV().transpose();
                    // convert the Tcw to a SE3d type
                    Sophus::SE3d Tcw_(Rcw_orthogonal, tcw);
                    if(!bVIBA0)
                    {
                        bVIBA0 = mpSLAM->mbInertialBA0;
                        Twc = Tcw_.inverse();
                        Twi = Twc*Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && !bVIBA1)
                    {
                        bVIBA1 = mpSLAM->mbInertialBA1;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && bVIBA1 && !bVIBA2)
                    {
                            
                        bVIBA2 = mpSLAM->mbInertialBA2;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // } 
                    }
                    else if(bVIBA2)
                    {
                        std::cout << "bVIBA2: " << bVIBA2 << std::endl;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        std::string frame_time_str1 = std::to_string(frame_time);
                        std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                }
                else if(frame_time > second_lidar_time)
                {
                    std::cout << "frame_time is greater than second_lidar_time" << std::endl;
                    // Sophus::SE3f Tcw = mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
                    lidar_time_queue_for_vslam.pop_front();
                    lidar_time_queue_for_vslam.pop_front();
                    while(!blidar_processed)
                    {
                        std::cout << "waiting for the lidar to be processed" << std::endl;
                        continue;
                    }
                    Sophus::SE3f Tcw = mpSLAM->TrackStereo_PCL(imLeft,imRight,tImLeft,vImuMeas,"",feats_down_body);
                    // if Tcw is no more identity, then we have a valid pose
                    if(!bTracker_ikdtree)
                    {
                        bTracker_ikdtree = true;
                        // get the pointer of the global variable ikdtree
                        auto ikdtree_ptr = &ikdtree;
                        mpSLAM->SetIKdtree(ikdtree_ptr);
                    }
                    if(bVIBA0 && !bVIBA2)
                    {
                        mpSLAM->CheckIKdtree_size();
                    }
                    Eigen::Matrix3d Rcw = Tcw.so3().matrix().cast<double>();
                    Eigen::Vector3d tcw = Tcw.translation().cast<double>();
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd_cw(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3d Rcw_orthogonal = svd_cw.matrixU() * svd_cw.matrixV().transpose();
                    // convert the Tcw to a SE3d type
                    Sophus::SE3d Tcw_(Rcw_orthogonal, tcw);
                    if(!bVIBA0)
                    {
                        bVIBA0 = mpSLAM->mbInertialBA0;
                        Twc = Tcw_.inverse();
                        Twi = Twc*Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && !bVIBA1)
                    {
                        bVIBA1 = mpSLAM->mbInertialBA1;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && bVIBA1 && !bVIBA2)
                    {
                            
                        bVIBA2 = mpSLAM->mbInertialBA2;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // } 
                    }
                    else if(bVIBA2)
                    {
                        std::cout << "bVIBA2: " << bVIBA2 << std::endl;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        std::string frame_time_str1 = std::to_string(frame_time);
                        std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                }        
                continue; 
            }
            else
            {
                std::cout << "lidar_time_queue_for_vslam does not have the second lidar time" << std::endl;
                if(frame_time <= last_lidar_time)
                {
                    // lidar_time_queue_for_vslam.pop_front();
                    std::cout << "frame_time is smaller than last_lidar_time" << std::endl;
                    Sophus::SE3f Tcw = mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
                    // if Tcw is no more identity, then we have a valid pose
                    if(!bTracker_ikdtree)
                    {
                        bTracker_ikdtree = true;
                        // get the pointer of the global variable ikdtree
                        auto ikdtree_ptr = &ikdtree;
                        mpSLAM->SetIKdtree(ikdtree_ptr);
                    }
                    if(bVIBA0 && !bVIBA2)
                    {
                        mpSLAM->CheckIKdtree_size();
                    }
                    Eigen::Matrix3d Rcw = Tcw.so3().matrix().cast<double>();
                    Eigen::Vector3d tcw = Tcw.translation().cast<double>();
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd_cw(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3d Rcw_orthogonal = svd_cw.matrixU() * svd_cw.matrixV().transpose();
                    // convert the Tcw to a SE3d type
                    Sophus::SE3d Tcw_(Rcw_orthogonal, tcw);
                    if(!bVIBA0)
                    {
                        bVIBA0 = mpSLAM->mbInertialBA0;
                        Twc = Tcw_.inverse();
                        Twi = Twc*Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && !bVIBA1)
                    {
                        bVIBA1 = mpSLAM->mbInertialBA1;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    else if(bVIBA0 && bVIBA1 && !bVIBA2)
                    {
                            
                        bVIBA2 = mpSLAM->mbInertialBA2;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        // std::string frame_time_str1 = std::to_string(frame_time);
                        // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // } 
                    }
                    else if(bVIBA2)
                    {
                        std::cout << "bVIBA2: " << bVIBA2 << std::endl;
                        Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                        Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                        Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                        // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                        Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                        Tyc0 = Tyc0_;
                        Tc0y = Tyc0.inverse();
                        Tycn = Tcw_.inverse();
                        Tc0cn = Tc0y * Tycn;
                        Twi = Tc0cn * Tleftb;
                        Tiwi = Tbleft*Twi;
                        DataEntry Measures;
                        Measures.frame_or_not = true;
                        Measures.Tiwi = Tiwi;
                        std::string frame_time_str1 = std::to_string(frame_time);
                        std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                        dataStore.addData(frame_time, Measures);
                        // {
                        //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                        //     // insertSorted(time_queue, frame_time);
                        //     frame_time_queue.push_back(frame_time);
                        // }
                    }
                    continue;
                }
                else if(frame_time > last_lidar_time)
                {
                    // calculate the time difference
                    double time_diff = frame_time - last_lidar_time;
                    double delta_t = 1.0/15.0;
                    if(time_diff <= delta_t)
                    {
                        std::cout << "time_diff is less than delta_t" << std::endl;
                        lidar_time_queue_for_vslam.pop_front();
                        while(!blidar_processed)
                        {
                            std::cout << "waiting for the lidar to be processed" << std::endl;
                            continue;
                        }
                        Sophus::SE3f Tcw = mpSLAM->TrackStereo_PCL(imLeft,imRight,tImLeft,vImuMeas,"",feats_down_body);
                        // if Tcw is no more identity, then we have a valid pose
                        if(!bTracker_ikdtree)
                        {
                            bTracker_ikdtree = true;
                            // get the pointer of the global variable ikdtree
                            auto ikdtree_ptr = &ikdtree;
                            mpSLAM->SetIKdtree(ikdtree_ptr);
                        }
                        if(bVIBA0 && !bVIBA2)
                        {
                            mpSLAM->CheckIKdtree_size();
                        }
                        Eigen::Matrix3d Rcw = Tcw.so3().matrix().cast<double>();
                        Eigen::Vector3d tcw = Tcw.translation().cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_cw(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Rcw_orthogonal = svd_cw.matrixU() * svd_cw.matrixV().transpose();
                        // convert the Tcw to a SE3d type
                        Sophus::SE3d Tcw_(Rcw_orthogonal, tcw);
                        if(!bVIBA0)
                        {
                            bVIBA0 = mpSLAM->mbInertialBA0;
                            Twc = Tcw_.inverse();
                            Twi = Twc*Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // }
                        }
                        else if(bVIBA0 && !bVIBA1)
                        {
                            bVIBA1 = mpSLAM->mbInertialBA1;
                            Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                            Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                            Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                            Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                            Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                            // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                            Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                            Tyc0 = Tyc0_;
                            Tc0y = Tyc0.inverse();
                            Tycn = Tcw_.inverse();
                            Tc0cn = Tc0y * Tycn;
                            Twi = Tc0cn * Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            // std::string frame_time_str1 = std::to_string(frame_time);
                            // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // }
                        }
                        else if(bVIBA0 && bVIBA1 && !bVIBA2)
                        {
                                
                            bVIBA2 = mpSLAM->mbInertialBA2;
                            Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                            Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                            Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                            Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                            Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                            // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                            Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                            Tyc0 = Tyc0_;
                            Tc0y = Tyc0.inverse();
                            Tycn = Tcw_.inverse();
                            Tc0cn = Tc0y * Tycn;
                            Twi = Tc0cn * Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            // std::string frame_time_str1 = std::to_string(frame_time);
                            // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // } 
                        }
                        else if(bVIBA2)
                        {
                            std::cout << "bVIBA2: " << bVIBA2 << std::endl;
                            Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                            Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                            Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                            Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                            Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                            // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                            Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                            Tyc0 = Tyc0_;
                            Tc0y = Tyc0.inverse();
                            Tycn = Tcw_.inverse();
                            Tc0cn = Tc0y * Tycn;
                            Twi = Tc0cn * Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            std::string frame_time_str1 = std::to_string(frame_time);
                            std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // }
                        }
                    }
                    else if(time_diff > delta_t && time_diff < 2.0* delta_t)
                    {
                        while(!(lidar_time_queue_for_vslam.size() > 1))
                        {
                            std::cout << "waiting for the lidar time queue to have more than 1 element" << std::endl;
                            continue;
                        }
                        while(!blidar_processed)
                        {
                            std::cout << "waiting for the lidar to be processed" << std::endl;
                            continue;
                        }
                        lidar_time_queue_for_vslam.pop_front();
                        lidar_time_queue_for_vslam.pop_front();
                        Sophus::SE3f Tcw = mpSLAM->TrackStereo_PCL(imLeft,imRight,tImLeft,vImuMeas,"",feats_down_body);
                        // if Tcw is no more identity, then we have a valid pose
                        if(!bTracker_ikdtree)
                        {
                            bTracker_ikdtree = true;
                            // get the pointer of the global variable ikdtree
                            auto ikdtree_ptr = &ikdtree;
                            mpSLAM->SetIKdtree(ikdtree_ptr);
                        }
                        if(bVIBA0 && !bVIBA2)
                        {
                            mpSLAM->CheckIKdtree_size();
                        }
                        Eigen::Matrix3d Rcw = Tcw.so3().matrix().cast<double>();
                        Eigen::Vector3d tcw = Tcw.translation().cast<double>();
                        Eigen::JacobiSVD<Eigen::Matrix3d> svd_cw(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                        Eigen::Matrix3d Rcw_orthogonal = svd_cw.matrixU() * svd_cw.matrixV().transpose();
                        // convert the Tcw to a SE3d type
                        Sophus::SE3d Tcw_(Rcw_orthogonal, tcw);
                        if(!bVIBA0)
                        {
                            bVIBA0 = mpSLAM->mbInertialBA0;
                            Twc = Tcw_.inverse();
                            Twi = Twc*Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // }
                        }
                        else if(bVIBA0 && !bVIBA1)
                        {
                            bVIBA1 = mpSLAM->mbInertialBA1;
                            Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                            Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                            Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                            Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                            Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                            // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                            Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                            Tyc0 = Tyc0_;
                            Tc0y = Tyc0.inverse();
                            Tycn = Tcw_.inverse();
                            Tc0cn = Tc0y * Tycn;
                            Twi = Tc0cn * Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            // std::string frame_time_str1 = std::to_string(frame_time);
                            // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // }
                        }
                        else if(bVIBA0 && bVIBA1 && !bVIBA2)
                        {
                                
                            bVIBA2 = mpSLAM->mbInertialBA2;
                            Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                            Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                            Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                            Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                            Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                            // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                            Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                            Tyc0 = Tyc0_;
                            Tc0y = Tyc0.inverse();
                            Tycn = Tcw_.inverse();
                            Tc0cn = Tc0y * Tycn;
                            Twi = Tc0cn * Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            // std::string frame_time_str1 = std::to_string(frame_time);
                            // std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // } 
                        }
                        else if(bVIBA2)
                        {
                            std::cout << "bVIBA2: " << bVIBA2 << std::endl;
                            Eigen::Matrix3f Ryc0 = mpSLAM->mRwg;
                            Eigen::Matrix3d Ryc0_ = Ryc0.cast<double>();
                            Eigen::JacobiSVD<Eigen::Matrix3d> svd_wg(Ryc0_, Eigen::ComputeFullU | Eigen::ComputeFullV);
                            Eigen::Matrix3d Ryc0_orthogonal = svd_wg.matrixU() * svd_wg.matrixV().transpose();
                            Eigen::Vector3f tyc0 = Eigen::Vector3f::Zero();
                            // create a Sophus::SE3f Tyc0 out of Ryc0 and tyc0
                            Sophus::SE3d Tyc0_(Ryc0_orthogonal, tyc0.cast<double>());
                            Tyc0 = Tyc0_;
                            Tc0y = Tyc0.inverse();
                            Tycn = Tcw_.inverse();
                            Tc0cn = Tc0y * Tycn;
                            Twi = Tc0cn * Tleftb;
                            Tiwi = Tbleft*Twi;
                            DataEntry Measures;
                            Measures.frame_or_not = true;
                            Measures.Tiwi = Tiwi;
                            std::string frame_time_str1 = std::to_string(frame_time);
                            std::cout << "frame_time_str before adding = " << frame_time_str1 << std::endl;
                            dataStore.addData(frame_time, Measures);
                            // {
                            //     std::lock_guard<std::mutex> lock(frame_time_queue_mutex);
                            //     // insertSorted(time_queue, frame_time);
                            //     frame_time_queue.push_back(frame_time);
                            // }
                        }
                    }
                }

            }

        }

    }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}


