// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>

#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver2/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>


#include<iostream>
#include<algorithm>
// #include<fstream>
#include<chrono>
#include<vector>
#include<queue>
// #include<thread>
// #include<mutex>

// #include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Imu.h>

#include<opencv2/core/core.hpp>

#include"ORB_SLAM3/include/System.h"
#include"ORB_SLAM3/include/ImuTypes.h"

// using namespace std;
// pcl::VoxelGrid<pcl::PCLPointCloud2> sor2;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
bool flg_exit = false;
condition_variable sig_buffer;

std::string voc_path, settings_path, do_rectify;
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    std::queue<sensor_msgs::ImuConstPtr> imuBuf;
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

    std::queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft,mBufMutexRight;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};


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
  // while(1)
  // {
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
      // continue;
      std::cout << "big time difference" << std::endl;
      return;
    }
    if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec()){
      std::cout << "imuBuf is not sync with imgBuf" << std::endl;
      return;
    }
    this->mBufMutexLeft.lock();
    imLeft = GetImage(imgLeftBuf.front());
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
    mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
    std::chrono::milliseconds tSleep(1);
    std::this_thread::sleep_for(tSleep);
  }
  // }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    nh.param("ORB_SLAM3/VocabularyPath",voc_path,std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Vocabulary/ORBvoc.txt"));
    // retrieve the settings file path by ros parameter
    nh.param("ORB_SLAM3/SettingsPath",settings_path,std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Examples/Stereo-Inertial/3001.yaml"));
    nh.param("ORB_SLAM3/ImuStereo/DoRectify",do_rectify,std::string("false"));
    
    bool bEqual = false;

    // std::thread main_thread(main_process, pubOdomAftMapped, pubPath, pubLaserCloudFull, pubLaserCloudFull_body, frame_num, aver_time_consu, aver_time_icp, aver_time_match, aver_time_incre, aver_time_solve, aver_time_const_H_time);
    // ORBSLAM
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
    ros::Subscriber sub_imu_v = nh.subscribe("/livox/imu", 1000, &ImuGrabber::GrabImu, &imugb); 
    ros::Subscriber sub_img_left = nh.subscribe("/uas_cam2/image", 100, &ImageGrabber::GrabImageLeft,&igb);
    ros::Subscriber sub_img_right = nh.subscribe("/uas_cam4/image", 100, &ImageGrabber::GrabImageRight,&igb);
    ros::Rate rate(5000);
    signal(SIGINT, SigHandle);
    bool status = ros::ok();
    while(status)
    {
        if(flg_exit)
        {
        break;
        }
        igb.SyncWithImu();
        ros::spinOnce();
        status = ros::ok();
        rate.sleep();
    }
    return 0;
}
