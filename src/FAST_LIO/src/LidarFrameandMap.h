#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sophus/se3.hpp>
#include <vector>
#include <stdexcept>
#include <deque>
#include <memory>
#include <numeric>
#include <algorithm>
#include <ikd-Tree/ikd_Tree.h>
#include <IKFoM_toolkit/esekfom/esekfom.hpp>

#include <Eigen/Core>



class PointCloudFrame {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZINormal>::Ptr;
    using Pose = Sophus::SE3d;
    using State_point = state_ikfom;
    using Point_vector = std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>>;

    // Constructor
    PointCloudFrame(PointCloud points, double lidar_timestamp, double img_timestamp, int degenerateIndex, const Pose& pose, const State_point& state_point, const Point_vector& PointToAdd, const Point_vector& PointNoNeedDownsample, const Point_vector& PointToAdd_body, const Point_vector& PointNoNeedDownsample_body) 
        : points_(points), lidar_timestamp_(lidar_timestamp), img_timestamp_(img_timestamp), degenerateIndex_(degenerateIndex), pose_(pose), state_point_(state_point), PointToAdd_(PointToAdd), PointNoNeedDownsample_(PointNoNeedDownsample), PointToAdd_body_(PointToAdd_body), PointNoNeedDownsample_body_(PointNoNeedDownsample_body), mapId_(-1) {}

    // Getters
    // PointCloud getPoints() const { return points_; }
    double getLidarTimestamp() const { return lidar_timestamp_; }
    double getImgTimestamp() const { return img_timestamp_; }
    int getDegenerateIndex() const { return degenerateIndex_; }
    const Pose& getPose() const { return pose_; }
    const State_point& getState_point() const { return state_point_; }
    int getMapId() const { return mapId_; }
    
    const PointCloud& getPoints() const {
        return points_;
    }
    // Setters
    void setPose(const Pose& newPose) { pose_ = newPose; }
    void setState_point(const State_point& newState_point) { state_point_ = newState_point; }
    void setMapId(int mapId) { mapId_ = mapId; }
    // points for building temporal ikdtree
    std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointToAdd_;
    std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointNoNeedDownsample_;
    // points for later ikdtree rebuilding
    std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointToAdd_body_;
    std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointNoNeedDownsample_body_;

private:
    PointCloud points_;   // Points in the point cloud
    double lidar_timestamp_;    // Timestamp of the point cloud frame
    double img_timestamp_;
    int degenerateIndex_; // Degenerate index
    Pose pose_;           // Pose of the point cloud frame
    int mapId_;           // ID of the submap
    state_ikfom state_point_;

};

class Submap {
public:
    // Constructor
    Submap(int submapId, bool isFeatureless = false)
        : submapId_(submapId), isFeatureless_(isFeatureless) {}

    // Getters
    int getSubmapId() const { return submapId_; }
    const std::vector<PointCloudFrame>& getPointCloudFrames() const { return pointCloudFrames_; }
    bool isFeatureless() const { return isFeatureless_; }
    // Twc of the current time, it is pose of the camera in camera world coordinate system, after applying SE3 or Sim3 transformation, 
    // the finally transformed trajectory should be backwardly transformed to Tiwi, not directly apply SE3 matrix to the Tiwi
    std::vector<double> frameTimes_IESEKF;
    std::vector<Sophus::SE3f> poses_IESEKF;
    std::pair<std::vector<double>, std::vector<Sophus::SE3f>> IESEKFtrajectory;
    
    // Setters
    void setFeatureless(bool featureless) { isFeatureless_ = featureless; }

    // Add a point cloud frame to the submap
    void addPointCloudFrame(const PointCloudFrame& frame) {
        pointCloudFrames_.emplace_back(frame);
    }

    // Remove the last point cloud frame from the submap and return it
    PointCloudFrame removeLastPointCloudFrame() {
        if (pointCloudFrames_.empty()) {
            throw std::runtime_error("No point cloud frames to remove");
        }
        PointCloudFrame lastFrame = pointCloudFrames_.back();
        pointCloudFrames_.pop_back();
        return lastFrame;
    }

    // Remove the last n point cloud frames from the submap and return them
    std::vector<PointCloudFrame> removeLastNPointCloudFrames(int n) {
        if (pointCloudFrames_.size() < n) {
            throw std::runtime_error("Not enough point cloud frames to remove");
        }
        std::vector<PointCloudFrame> frames(pointCloudFrames_.end() - n, pointCloudFrames_.end());
        pointCloudFrames_.erase(pointCloudFrames_.end() - n, pointCloudFrames_.end());
        return frames;
    }

    std::vector<PointCloudFrame> returnLastNPointCloudFrames(int n) {
        if (pointCloudFrames_.size() < n) {
            throw std::runtime_error("Not enough point cloud frames to remove");
        }
        std::vector<PointCloudFrame> frames(pointCloudFrames_.end() - n, pointCloudFrames_.end());
        return frames;
    }
    // KD_TREE<pcl::PointXYZINormal>::Ptr ikdtreeptr_; // IKD-tree pointer

    std::vector<double> returnLastNFrametimes(int n){
        if (frameTimes_IESEKF.size() < n) {
            throw std::runtime_error("Not enough frame times to return");
        }
        std::vector<double> frameTimes(frameTimes_IESEKF.end() - n, frameTimes_IESEKF.end());
        return frameTimes;
    }

    std::vector<Sophus::SE3f> returnLastNPoses(int n){
        if (poses_IESEKF.size() < n) {
            throw std::runtime_error("Not enough poses to return");
        }
        std::vector<Sophus::SE3f> poses(poses_IESEKF.end() - n, poses_IESEKF.end());
        return poses;
    }

private:
    int submapId_;  // ID of the submap
    std::vector<PointCloudFrame> pointCloudFrames_;  // Collection of point cloud frames
    bool isFeatureless_;  // Indicator of whether the submap is featureless (degenerate)

};

class KDTreeManager {
public:
    using KDTreePtr = KD_TREE<pcl::PointXYZINormal>::Ptr;

    // Add a KD-Tree associated with a submap ID
    void addKDTree(int submapId, KDTreePtr kdTree) {
        kdTreeMap_[submapId] = kdTree;
    }

    // Retrieve a KD-Tree by submap ID
    KDTreePtr getKDTree(int submapId) const {
        auto it = kdTreeMap_.find(submapId);
        if (it != kdTreeMap_.end()) {
            return it->second;
        }
        throw std::runtime_error("KD-Tree not found for submap ID " + std::to_string(submapId));
    }

    // Check if a KD-Tree exists for a submap ID
    bool hasKDTree(int submapId) const {
        return kdTreeMap_.find(submapId) != kdTreeMap_.end();
    }

private:
    std::unordered_map<int, KDTreePtr> kdTreeMap_;
};


class LidarMap {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZINormal>::Ptr;
    using Pose = Sophus::SE3d;
    using State_point = state_ikfom;
    using Point_vector = std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>>;

    // Constructor
    LidarMap() : currentSubmap_(nullptr), isFeatureRich_(true), degenerateIndexThreshold_(400), current_ikdtreeptr_(nullptr) {}

    std::vector<std::vector<PointCloudFrame>> returnAllPointCloudFrames() {
        std::vector<std::vector<PointCloudFrame>> allPointCloudFrames;
        for (const auto& submap : submaps_) {
            allPointCloudFrames.push_back(submap->getPointCloudFrames());
        }
        return allPointCloudFrames;
    }

    std::vector<PointCloudFrame> returnSubMapPointCloudFrames(int submapId) {
        for (const auto& submap : submaps_) {
            if (submap->getSubmapId() == submapId) {
                return submap->getPointCloudFrames();
            }
        }
        throw std::runtime_error("Submap ID not found");
    }

    void buildsubMapTrajectory(const double& currentframe_time, const Sophus::SE3f& currentframe_pose)
    {
        if(currentSubmap_)
        {
            currentSubmap_->frameTimes_IESEKF.push_back(currentframe_time);
            currentSubmap_->poses_IESEKF.push_back(currentframe_pose);
        }
        else
        {
            std::cout << "current submap is null" << std::endl;
        }
    }

    // need a method to return all submap trajectories
    std::vector<std::pair<std::vector<double>, std::vector<Sophus::SE3f>>> returnSubmapTrajectories()
    {
        std::vector<std::pair<std::vector<double>, std::vector<Sophus::SE3f>>> submapTrajectories;
        for (const auto& submap : submaps_) {
            submap->IESEKFtrajectory = std::make_pair(submap->frameTimes_IESEKF, submap->poses_IESEKF);
            submapTrajectories.push_back(submap->IESEKFtrajectory);
        }
        return submapTrajectories;
    }

    // need a method to return whether the submaps are featureless
    std::vector<bool> returnSubmapFeatureless()
    {
        std::vector<bool> submapFeatureless;
        for (const auto& submap : submaps_) {
            submapFeatureless.push_back(submap->isFeatureless());
        }
        return submapFeatureless;
    }

    // Add a point cloud frame to the current submap
    void addPointCloudFrame(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, double lidar_timestamp, double img_timestamp, int degenerateIndex, const Sophus::SE3d& pose, const state_ikfom& state_point, const Point_vector& PointToAdd, const Point_vector& PointNoNeedDownsample, const Point_vector& PointToAdd_body, const Point_vector& PointNoNeedDownsample_body) {
        PointCloudFrame frame(cloud, lidar_timestamp, img_timestamp, degenerateIndex, pose, state_point, PointToAdd, PointNoNeedDownsample, PointToAdd_body, PointNoNeedDownsample_body);
        bool shouldCreateNewSubmap = false;
        // If there's no current submap, create a feature-rich one by default
        if (!currentSubmap_) {

            createNewSubmap(true);
            std::cout << "first submap created" << std::endl;
        } else {
            // Update the queue with the latest degenerate index
            degenerateIndices_.push_back(degenerateIndex);
            if (degenerateIndices_.size() > 30) {
                degenerateIndices_.pop_front();
            }

            // Determine if a new submap should be created
            
            if (isFeatureRich_) {
                // Check if the last 20 frames are all degenerate
                shouldCreateNewSubmap = std::all_of(degenerateIndices_.begin(), degenerateIndices_.end(),
                                                    [this](int idx) { return idx < degenerateIndexThreshold_; });
                // check if there are more than 40 frames in the current submap
                bool enoughFrames = currentSubmap_->getPointCloudFrames().size() > 100;

                if (shouldCreateNewSubmap && enoughFrames) {
                    std::cout << "saveCurrentSubmapAndTransferFrames0" << std::endl;
                    saveCurrentSubmapAndTransferFrames(29, false);
                }
            } else {
                // Check if the last 20 frames are all feature-rich
                shouldCreateNewSubmap = std::all_of(degenerateIndices_.begin(), degenerateIndices_.end(),
                                                    [this](int idx) { return idx >= degenerateIndexThreshold_; });
                bool enoughFrames = currentSubmap_->getPointCloudFrames().size() > 100;

                if (shouldCreateNewSubmap && enoughFrames) {
                    std::cout << "saveCurrentSubmapAndTransferFrames1" << std::endl;
                    saveCurrentSubmapAndTransferFrames(29, true);
                }
            }
        }

        // Add the frame to the current submap
        currentSubmap_->addPointCloudFrame(frame);
        std::cout << "new frame added to submap" << std::endl;
        if(shouldCreateNewSubmap)
        {
            int size;
            std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointToAddnew;
            PointToAddnew = frame.PointToAdd_;
            size = current_ikdtreeptr_->Add_Points(PointToAddnew, true);
            std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointNoNeedDownsamplenew;
            PointNoNeedDownsamplenew = frame.PointNoNeedDownsample_;
            current_ikdtreeptr_->Add_Points(PointNoNeedDownsamplenew, false);
            std::cout << "last previous frame points added to ikdtree" << std::endl;
        }
    }

    // Get the current active submap
    std::shared_ptr<Submap> getCurrentSubmap() const {
        return currentSubmap_;
    }

    // Check if the current active submap is degenerate
    bool isCurrentSubmapDegenerate() const {
        if (currentSubmap_) {
            return currentSubmap_->isFeatureless();
        }
        return false;
    }

    int getMapSize() const {
        return submaps_.size();
    }

    void pointBodyToWorld(pcl::PointXYZINormal const * const pi, pcl::PointXYZINormal * const po, const state_ikfom& state_point)
    {
        using V3D = Eigen::Vector3d;
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }
    bool changeCurrentSubmap(double dMatchedTime) {
        bool gotMatch = false;
        std::cout << "changeCurrentSubmap" << std::endl;
        for (const auto& submap : submaps_) {
            // int size1 = submap->ikdtreeptr_->size();
            // std::cout << "submap ikdtree size: " << size1 << std::endl;
            std::cout << "loop through submaps" << std::endl;
            
            // for (const auto& frame : submap->getPointCloudFrames()) {
            //     if (fabs(frame.getImgTimestamp() - dMatchedTime) < 0.1) {
            //         gotMatch = true;
            //         // quit the current for loop
            //         break;

            //     }
            // }
            // convert dMatchedTime to a string
            std::string sMatchedTime = std::to_string(dMatchedTime);
            auto frames = submap->getPointCloudFrames();
            // get the timestamp of the first frame
            double dFirstFrameTime = frames[0].getImgTimestamp();
            std::string sFirstFrameTime = std::to_string(dFirstFrameTime);
            // get the timestamp of the last frame
            double dLastFrameTime = frames[frames.size()-1].getImgTimestamp();
            std::string sLastFrameTime = std::to_string(dLastFrameTime);
            std::cout << "Matched Time: " << sMatchedTime << std::endl;
            std::cout << "First Frame Time: " << sFirstFrameTime << std::endl;
            std::cout << "Last Frame Time: " << sLastFrameTime << std::endl;
            if (dMatchedTime >= dFirstFrameTime && dMatchedTime <= dLastFrameTime) {
                gotMatch = true;
            }
            if(gotMatch)
            {
                // int size2 = submap->ikdtreeptr_->size();
                // std::cout << "submap ikdtree size got: " << size2 << std::endl;
                // if (currentSubmap_) {
                //     std::cout << "save current submap before switching submap" << std::endl;
                //     saveCurrentSubmap();
                //     std::cout << "current submap saved" << std::endl;
                // }
                // currentSubmap_->ikdtreeptr_ = current_ikdtreeptr_;
                if(kdTreeManager_.hasKDTree(currentSubmap_->getSubmapId()))
                {
                    std::cout << "ikdtree already exist in kdtree manager" << std::endl;
                }
                else
                {
                    kdTreeManager_.addKDTree(currentSubmap_->getSubmapId(), current_ikdtreeptr_);
                }
                
                int size2 = kdTreeManager_.getKDTree(submap->getSubmapId())->size();
                std::cout << "submap ikdtree size got: " << size2 << std::endl;
                // currentSubmap_ = submap;
                std::cout << "ikdtree saved to submap1" << std::endl;
                // submaps_.push_back(currentSubmap_);
                
                current_ikdtreeptr_ = kdTreeManager_.getKDTree(submap->getSubmapId());
                saveCurrentSubmap();
                currentSubmap_ = submap;
                // current_ikdtreeptr_ = submap->ikdtreeptr_;
                std::cout << "current submap and ikdtree changed" << std::endl;
                break;
            }
        }
        return gotMatch;
    }

bool Submapchangeable(double dMatchedTime) {
        bool gotMatch = false;
        std::cout << "changeCurrentSubmap" << std::endl;
        for (const auto& submap : submaps_) {
            // int size1 = submap->ikdtreeptr_->size();
            // std::cout << "submap ikdtree size: " << size1 << std::endl;
            std::cout << "loop through submaps" << std::endl;
            
            // for (const auto& frame : submap->getPointCloudFrames()) {
            //     if (fabs(frame.getImgTimestamp() - dMatchedTime) < 0.1) {
            //         gotMatch = true;
            //         // quit the current for loop
            //         break;

            //     }
            // }
            // convert dMatchedTime to a string
            std::string sMatchedTime = std::to_string(dMatchedTime);
            auto frames = submap->getPointCloudFrames();
            // get the timestamp of the first frame
            double dFirstFrameTime = frames[0].getImgTimestamp();
            std::string sFirstFrameTime = std::to_string(dFirstFrameTime);
            // get the timestamp of the last frame
            double dLastFrameTime = frames[frames.size()-1].getImgTimestamp();
            std::string sLastFrameTime = std::to_string(dLastFrameTime);
            std::cout << "Matched Time: " << sMatchedTime << std::endl;
            std::cout << "First Frame Time: " << sFirstFrameTime << std::endl;
            std::cout << "Last Frame Time: " << sLastFrameTime << std::endl;
            if (dMatchedTime >= dFirstFrameTime && dMatchedTime <= dLastFrameTime) {
                gotMatch = true;
            }
        }
        return gotMatch;
    }

    // KD_TREE<pcl::PointXYZINormal> current_ikdtree_;
    KD_TREE<pcl::PointXYZINormal>::Ptr current_ikdtreeptr_;

private:
    void saveCurrentSubmap() {
        std::cout << "saveCurrentSubmap0" << std::endl;
        if (currentSubmap_) {
            std::cout << "saveCurrentSubmap1" << std::endl;
            // currentSubmap_->ikdtreeptr_ = current_ikdtreeptr_;
            submaps_.push_back(currentSubmap_);
        }
        else
        {
            std::cout << "saveCurrentSubmap2" << std::endl;
        }
        std::cout << "saveCurrentSubmap3" << std::endl;
    }

    void saveCurrentSubmapAndTransferFrames(int n, bool isFeatureRich) {
        std::cout << "saveCurrentSubmapAndTransferFramesx" << std::endl;
        if (currentSubmap_) {
            std::cout << "saveCurrentSubmapAndTransferFramesxx" << std::endl;
            std::vector<PointCloudFrame> frames = currentSubmap_->returnLastNPointCloudFrames(n);
            std::vector<double> frameTimes = currentSubmap_->returnLastNFrametimes(n);
            std::vector<Sophus::SE3f> poses = currentSubmap_->returnLastNPoses(n);
            // std::cout << "saveCurrentSubmapAndTransferFramesxxx" << std::endl;
            // currentSubmap_->ikdtreeptr_ = current_ikdtreeptr_;
            if(kdTreeManager_.hasKDTree(currentSubmap_->getSubmapId()))
            {
                std::cout << "ikdtree already exist in kdtree manager" << std::endl;
            }
            else
            {
                kdTreeManager_.addKDTree(currentSubmap_->getSubmapId(), current_ikdtreeptr_);
            }
            // kdTreeManager_.addKDTree(currentSubmap_->getSubmapId(), current_ikdtreeptr_);

            std::cout << "ikdtree saved to submap, waiting for new map to be created" << std::endl;
            int size;
            // size =  currentSubmap_->ikdtreeptr_->size();
            // std::cout << "submap ikdtree size: " << size << std::endl;
            submaps_.push_back(currentSubmap_);
            // std::cout << "saveCurrentSubmapAndTransferFramesxxxxxx" << std::endl;
            createNewSubmap(isFeatureRich);
            // std::cout << "Build ikdtree in globalmap" << std::endl;
            // create a new ikdtree
            // current_ikdtree_ = 
            // KD_TREE<pcl::PointXYZINormal> new_ikdtree;
            KD_TREE<pcl::PointXYZINormal>::Ptr kdTreeptr(new KD_TREE<pcl::PointXYZINormal>());
            
            current_ikdtreeptr_ = kdTreeptr;
            std::cout << "new ikdtree for new submap created" << std::endl;
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_down_world(new pcl::PointCloud<pcl::PointXYZINormal>());
            std::cout << "Build ikdtree in globalmap0" << std::endl;
            for (const auto& frame : frames) {
                currentSubmap_->addPointCloudFrame(frame);
                int index = &frame - &frames[0];
                currentSubmap_->frameTimes_IESEKF.push_back(frameTimes[index]);
                // get the index of frame in vector frames
                
                currentSubmap_->poses_IESEKF.push_back(poses[index]);
                if(current_ikdtreeptr_->Root_Node == nullptr)
                {
                    // std::cout << "Build ikdtree in globalmap" << std::endl;
                    current_ikdtreeptr_->set_downsample_param(0.3);
                    // std::cout << "Build ikdtree in globalmap0" << std::endl;
                    int pcl_size = frame.getPoints()->size();
                    // std::cout << "Build ikdtree in globalmap1" << std::endl;
                    feats_down_world->points.resize(pcl_size);
                    for (int i = 0; i < pcl_size; i++)
                    {
                        pointBodyToWorld(&frame.getPoints()->points[i], &feats_down_world->points[i], frame.getState_point());
                    }
                    // std::cout << "Build ikdtree in globalmap2" << std::endl;
                    current_ikdtreeptr_->Build(feats_down_world->points);
                    std::cout << "new current ikdtree built" << std::endl;
                }
                else
                {
                    int size;
                    std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointToAdd;
                    PointToAdd = frame.PointToAdd_;
                    size = current_ikdtreeptr_->Add_Points(PointToAdd, true);
                    std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> PointNoNeedDownsample;
                    PointNoNeedDownsample = frame.PointNoNeedDownsample_;
                    current_ikdtreeptr_->Add_Points(PointNoNeedDownsample, false);
                    std::cout << "previous frame points added to ikdtree" << std::endl;
                }

            }
            // for (const auto& frame : frames) {
            //     std::cout << "saveCurrentSubmapAndTransferFrameso" << std::endl;
            //     currentSubmap_->addPointCloudFrame(frame);
            // }
        } else {
            
            createNewSubmap(isFeatureRich);
            std::cout << "first submap created" << std::endl;
        }
    }
    // Create a new submap
    void createNewSubmap(bool isFeatureRich) {
        static int submapIdCounter = 0;
        currentSubmap_ = std::make_shared<Submap>(submapIdCounter++, !isFeatureRich);
        isFeatureRich_ = isFeatureRich;
    }

    std::vector<std::shared_ptr<Submap>> submaps_; // Collection of saved submaps
    std::shared_ptr<Submap> currentSubmap_; // Currently active submap
    bool isFeatureRich_; // Indicator of whether the current submap is feature-rich
    const int degenerateIndexThreshold_; // Threshold for determining degeneracy
    std::deque<int> degenerateIndices_; // Queue to store the latest 20 degenerate indices
    KDTreeManager kdTreeManager_;

};
