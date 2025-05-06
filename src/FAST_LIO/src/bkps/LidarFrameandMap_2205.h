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

class PointCloudFrame {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZINormal>::Ptr;
    using Pose = Sophus::SE3d;

    // Constructor
    PointCloudFrame(PointCloud points, double timestamp, int degenerateIndex, const Pose& pose)
        : points_(points), timestamp_(timestamp), degenerateIndex_(degenerateIndex), pose_(pose), mapId_(-1) {}

    // Getters
    PointCloud getPoints() const { return points_; }
    double getTimestamp() const { return timestamp_; }
    int getDegenerateIndex() const { return degenerateIndex_; }
    const Pose& getPose() const { return pose_; }
    int getMapId() const { return mapId_; }

    // Setters
    void setPose(const Pose& newPose) { pose_ = newPose; }
    void setMapId(int mapId) { mapId_ = mapId; }

private:
    PointCloud points_;   // Points in the point cloud
    double timestamp_;    // Timestamp of the point cloud frame
    int degenerateIndex_; // Degenerate index
    Pose pose_;           // Pose of the point cloud frame
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

private:
    int submapId_;  // ID of the submap
    std::vector<PointCloudFrame> pointCloudFrames_;  // Collection of point cloud frames
    bool isFeatureless_;  // Indicator of whether the submap is featureless (degenerate)
};

class LidarMap {
public:
    // Constructor
    LidarMap() : currentSubmap_(nullptr), isFeatureRich_(true), degenerateIndexThreshold_(1500) {}

    // Add a point cloud frame to the current submap
    void addPointCloudFrame(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, double timestamp, int degenerateIndex, const Sophus::SE3d& pose) {
        PointCloudFrame frame(cloud, timestamp, degenerateIndex, pose);

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
            submaps_.push_back(currentSubmap_);
            createNewSubmap(isFeatureRich);
            for (const auto& frame : frames) {
                currentSubmap_->addPointCloudFrame(frame);
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
};