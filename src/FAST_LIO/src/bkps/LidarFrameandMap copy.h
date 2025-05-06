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
#include "IMU_Processing.hpp"
#include "preprocess.h"
#include <Eigen/Core>

// void pointBodyToWorld(pcl::PointXYZINormal const * const pi, pcl::PointXYZINormal * const po)
// {
//     Eigen::Matrix(3,1) p_body(pi->x, pi->y, pi->z);
//     Eigen::Matrix(3,1) p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

//     po->x = p_global(0);
//     po->y = p_global(1);
//     po->z = p_global(2);
//     po->intensity = pi->intensity;
// }

class PointCloudFrame {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZINormal>::Ptr;
    using State_point = state_ikfom;

    // Constructor
    PointCloudFrame(PointCloud points, double timestamp, int degenerateIndex, const State_point& state_point)
        : points_(points), timestamp_(timestamp), degenerateIndex_(degenerateIndex), state_point_(state_point), mapId_(-1) {}

    // Getters
    PointCloud getPoints() const { return points_; }
    double getTimestamp() const { return timestamp_; }
    int getDegenerateIndex() const { return degenerateIndex_; }
    const State_point& getState_point() const { return state_point_; }
    int getMapId() const { return mapId_; }

    // Setters
    void setState_point(const State_point& newState_point) { state_point_ = newState_point; }
    void setMapId(int mapId) { mapId_ = mapId; }

private:
    PointCloud points_;   // Points in the point cloud
    double timestamp_;    // Timestamp of the point cloud frame
    int degenerateIndex_; // Degenerate index
    State_point state_point_;           // State_point of the point cloud frame
    int mapId_;           // ID of the submap
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

    void setIkdTree(KD_TREE<pcl::PointXYZINormal> ikdtree_) {
        ikdtree = ikdtree_;
    }

private:
    int submapId_;  // ID of the submap
    std::vector<PointCloudFrame> pointCloudFrames_;  // Collection of point cloud frames
    bool isFeatureless_;  // Indicator of whether the submap is featureless (degenerate)
    KD_TREE<pcl::PointXYZINormal> ikdtree;
};

class LidarMap {
public:
    // Constructor
    LidarMap() : currentSubmap_(nullptr), isFeatureRich_(true), degenerateIndexThreshold_(1500) {}

    // Add a point cloud frame to the current submap
    void addPointCloudFrame(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, double timestamp, int degenerateIndex, const state_ikfom& state_point) {
        PointCloudFrame frame(cloud, timestamp, degenerateIndex, state_point);

        // If there's no current submap, create a feature-rich one by default
        if (!currentSubmap_) {
            createNewSubmap(true);
        } else {
            // Update the queue with the latest degenerate index
            degenerateIndices_.push_back(degenerateIndex);
            if (degenerateIndices_.size() > 20) {
                degenerateIndices_.pop_front();
            }

            // Determine if a new submap should be created
            bool shouldCreateNewSubmap = false;
            if (isFeatureRich_) {
                // Check if the last 20 frames are all degenerate
                shouldCreateNewSubmap = std::all_of(degenerateIndices_.begin(), degenerateIndices_.end(),
                                                    [this](int idx) { return idx < degenerateIndexThreshold_; });

                if (shouldCreateNewSubmap) {
                    saveCurrentSubmapAndTransferFrames(19, false);
                }
            } else {
                // Check if the last 20 frames are all feature-rich
                shouldCreateNewSubmap = std::all_of(degenerateIndices_.begin(), degenerateIndices_.end(),
                                                    [this](int idx) { return idx >= degenerateIndexThreshold_; });

                if (shouldCreateNewSubmap) {
                    saveCurrentSubmapAndTransferFrames(19, true);
                }
            }
        }

        // Add the frame to the current submap
        currentSubmap_->addPointCloudFrame(frame);
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

private:
    // Save the current submap and set it inactive, transferring the last n frames to the new submap
    void saveCurrentSubmapAndTransferFrames(int n, bool isFeatureRich) {
        if (currentSubmap_) {
            std::vector<PointCloudFrame> frames = currentSubmap_->removeLastNPointCloudFrames(n);
            currentSubmap_->setIkdTree(current_ikdtree);
            submaps_.push_back(currentSubmap_);
            createNewSubmap(isFeatureRich);
            KD_TREE<pcl::PointXYZINormal> new_ikdtree;
            current_ikdtree = new_ikdtree;
            for (const auto& frame : frames) {
                currentSubmap_->addPointCloudFrame(frame);
                if(current_ikdtree.Root_Node == nullptr)
                {
                    current_ikdtree.set_downsample_param(0.3);

                }
            }
        } else {
            createNewSubmap(isFeatureRich);
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
    KD_TREE<pcl::PointXYZINormal> current_ikdtree;
};