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
#include<message_filters/subscriber.h>
#include<message_filters/time_synchronizer.h>
#include<message_filters/sync_policies/approximate_time.h>

#include<opencv2/core/core.hpp>

#include"ORB_SLAM3/include/System.h"

using namespace std;

std::string voc_path, settings_path, do_rectify;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, const bool bRect): mpSLAM(pSLAM), do_rectify(bRect){}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncImages();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft, mBufMutexRight;

    ORB_SLAM3::System* mpSLAM;

    const bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Stereo");
    ros::NodeHandle nh;

    nh.param("ORB_SLAM3/VocabularyPath", voc_path, std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Vocabulary/ORBvoc.txt"));
    nh.param("ORB_SLAM3/SettingsPath", settings_path, std::string("/home/uas/catkin_sfslio/src/S-FAST_LIO/ORB_SLAM3/Examples/Stereo/usbcam_basalt_1901drone800_forcompare.yaml"));
    nh.param("ORB_SLAM3/ImuStereo/DoRectify", do_rectify, std::string("false"));

    ORB_SLAM3::System SLAM(voc_path, settings_path, ORB_SLAM3::System::STEREO, true);

    ImageGrabber igb(&SLAM, do_rectify == "true");

    if (igb.do_rectify)
    {
        cv::FileStorage fsSettings(settings_path, cv::FileStorage::READ);
        if (!fsSettings.isOpened())
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

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, igb.M1l, igb.M2l);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, igb.M1r, igb.M2r);
    }

    ros::Subscriber sub_img_left = nh.subscribe("/uas_cam2/image", 100, &ImageGrabber::GrabImageLeft, &igb);
    ros::Subscriber sub_img_right = nh.subscribe("/uas_cam4/image", 100, &ImageGrabber::GrabImageRight, &igb);

    std::thread sync_thread(&ImageGrabber::SyncImages, &igb);

    ros::spin();

    return 0;
}

void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
    std::lock_guard<std::mutex> lock(mBufMutexLeft);
    imgLeftBuf.push(img_msg);
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg)
{
    std::lock_guard<std::mutex> lock(mBufMutexRight);
    imgRightBuf.push(img_msg);
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    return cv_ptr->image.clone();
}

void ImageGrabber::SyncImages()
{
    const double maxTimeDiff = 0.01;
    while (1)
    {
        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;

        if (!imgLeftBuf.empty() && !imgRightBuf.empty())
        {
            tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            tImRight = imgRightBuf.front()->header.stamp.toSec();

            std::lock_guard<std::mutex> lockRight(mBufMutexRight);
            while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf.size() > 1)
            {
                imgRightBuf.pop();
                tImRight = imgRightBuf.front()->header.stamp.toSec();
            }

            std::lock_guard<std::mutex> lockLeft(mBufMutexLeft);
            while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf.size() > 1)
            {
                imgLeftBuf.pop();
                tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            }

            if ((tImLeft - tImRight) > maxTimeDiff || (tImRight - tImLeft) > maxTimeDiff)
            {
                continue;
            }

            imLeft = GetImage(imgLeftBuf.front());
            imgLeftBuf.pop();

            imRight = GetImage(imgRightBuf.front());
            imgRightBuf.pop();

            if (do_rectify)
            {
                cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
                cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
            }

            Sophus::SE3f Tcw = mpSLAM->TrackStereo(imLeft, imRight, tImLeft);
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f t = Twc.translation();
            std::cout << "t: " << t[0] << " " << t[1] << " " << t[2] << std::endl;

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}
