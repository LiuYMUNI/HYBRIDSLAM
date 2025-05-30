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
#include<sensor_msgs/Imu.h>

#include<opencv2/core/core.hpp>

#include"ORB_SLAM3/include/System.h"
#include"ORB_SLAM3/include/ImuTypes.h"

using namespace std;
// create a queue to store the frame_time
std::deque<double> frame_time_queue;
std::deque<double> frame_time_queue_for_vslam;
std::deque<double> lidar_time_queue;
std::deque<double> lidar_time_queue_for_vslam;
// create corresponding mutex for the frame_time_queue
std::mutex frame_time_queue_mutex;
std::mutex frame_time_queue_for_vslam_mutex;
std::mutex lidar_time_queue_mutex;
std::mutex lidar_time_queue_for_vslam_mutex;
double frame_time_g_vslam;
std::ostream& operator<<(std::ostream& os, const boost::array<double, 3>& arr) {
    os << "[" << arr[0] << ", " << arr[1] << ", " << arr[2] << "]";
    return os;
}
struct DoubleHash {
    std::size_t operator()(const double& d) const {
        return std::hash<double>()(d);
    }
};

struct VSLAMData{
    bool frame_or_not;
    cv::Mat imLeft;
    cv::Mat imRight;
    double tImLeft;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;
};
class VSLAMDataStore {
private:
    std::unordered_map<double, VSLAMData, DoubleHash> dataMap; // Using unordered_map for faster access

public:
    void addData(double timestamp, const VSLAMData& entry) {
        dataMap[timestamp] = entry;
    }

    const VSLAMData& getData(double timestamp) const {
        return dataMap.at(timestamp);
    }

    bool hasData(double timestamp) const {
        return dataMap.find(timestamp) != dataMap.end();
    }
    void removeData(double timestamp) {
        dataMap.erase(timestamp);
    }
};

std::string voc_path, settings_path, do_rectify;

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
    void ProcessData();
    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft,mBufMutexRight;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};

VSLAMDataStore vslamDataStore;

int main(int argc, char **argv)
{
  frame_time_g_vslam = 0.0;
  ros::init(argc, argv, "Stereo_Inertial");
  ros::NodeHandle no;
  // retrieve the vocabulary file path by ros parameter

  no.param("ORB_SLAM3/VocabularyPath",voc_path,std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Vocabulary/ORBvoc.txt"));
  // retrieve the settings file path by ros parameter
  no.param("ORB_SLAM3/SettingsPath",settings_path,std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Examples/Stereo-Inertial/3001.yaml"));
  no.param("ORB_SLAM3/ImuStereo/DoRectify",do_rectify,std::string("false"));
  
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  bool bEqual = false;
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
  std::thread process_thread(&ImageGrabber::ProcessData,&igb);
  ros::spin();

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
      {
          std::lock_guard<std::mutex> lock(frame_time_queue_for_vslam_mutex);
          frame_time_queue_for_vslam.push_back(tImLeft);
      }
      VSLAMData vslamdata;
      vslamdata.frame_or_not = true;
      vslamdata.imLeft = imLeft;
      vslamdata.imRight = imRight;
      vslamdata.tImLeft = tImLeft;
      vslamdata.vImuMeas = vImuMeas;
      vslamDataStore.addData(tImLeft, vslamdata);

      // Sophus::SE3f Tcw = mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
      // Sophus::SE3f Twc = Tcw.inverse();
      // Eigen::Vector3f t = Tcw.translation();
      // Eigen::Vector3f tw = Twc.translation();
      // std::cout << "t: " << t[0] << " " << t[1] << " " << t[2] << std::endl;
      // std::cout << "tw: " << tw[0] << " " << tw[1] << " " << tw[2] << std::endl;
      // std::cout << "Tcw: " << Tcw.matrix() << std::endl;

      std::chrono::milliseconds tSleep(1);
      std::this_thread::sleep_for(tSleep);
    }
  }
}
void ImageGrabber::ProcessData()
{
  while(1)
  {
    if(!frame_time_queue_for_vslam.empty())
    {
      double frame_time = frame_time_queue_for_vslam.front();
      frame_time_queue_for_vslam.pop_front();
      frame_time_g_vslam = frame_time;
    }
    if(frame_time_g_vslam != 0.0)
    {
      while(!vslamDataStore.hasData(frame_time_g_vslam))
      {
        continue;
      }
    }
    else
    {
      continue;
    }
    VSLAMData vslamdata = vslamDataStore.getData(frame_time_g_vslam);
    Sophus::SE3f Tcw = mpSLAM->TrackStereo(vslamdata.imLeft,vslamdata.imRight,vslamdata.tImLeft,vslamdata.vImuMeas);
    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Vector3f t = Tcw.translation();
    Eigen::Vector3f tw = Twc.translation();
    std::cout << "t: " << t[0] << " " << t[1] << " " << t[2] << std::endl;
    std::cout << "tw: " << tw[0] << " " << tw[1] << " " << tw[2] << std::endl;
    std::cout << "Tcw: " << Tcw.matrix() << std::endl;
  }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}


