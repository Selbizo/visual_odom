#include "loopclosure_simple.h"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

LoopClosureSimple::LoopClosureSimple() 
    : enabled_(true),
      min_keyframes_for_loop_(5),
      max_pose_distance_(50.0),
      min_pose_distance_(10.0),
      running_(false)
{
    // Initialize with default parameters
}

LoopClosureSimple::~LoopClosureSimple()
{
    if (running_.load()) {
        running_.store(false);
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }
}

void LoopClosureSimple::AddFrame(const cv::Mat& image, const cv::Mat& pose)
{
    // Store keyframe information
    keyframe_images_.push_back(image.clone());
    keyframe_poses_.push_back(pose.clone());
    keyframe_ids_.push_back(keyframe_ids_.empty() ? 0 : keyframe_ids_.back() + 1);
    
    // Limit number of stored keyframes to avoid memory issues
    if (keyframe_images_.size() > 20) {
        keyframe_images_.erase(keyframe_images_.begin());
        keyframe_poses_.erase(keyframe_poses_.begin());
        keyframe_ids_.erase(keyframe_ids_.begin());
    }
}

bool LoopClosureSimple::CheckForLoop(int current_frame_id, const cv::Mat& current_pose)
{
    if (!enabled_ || keyframe_images_.size() < min_keyframes_for_loop_) {
        return false;
    }

    // Simple distance-based loop detection
    double min_distance = max_pose_distance_;
    int closest_keyframe_idx = -1;
    
    for (size_t i = 0; i < keyframe_poses_.size(); i++) {
        // Calculate Euclidean distance between poses
        cv::Mat diff = current_pose.col(3) - keyframe_poses_[i].col(3);
        double distance = cv::norm(diff);
        
        if (distance < min_distance && distance > min_pose_distance_) {
            min_distance = distance;
            closest_keyframe_idx = i;
        }
    }
    
    // If we found a potential loop
    if (closest_keyframe_idx != -1 && min_distance < max_pose_distance_ / 2.0) {
        std::cout << "[LoopClosure] Potential loop detected at frame " 
                   << current_frame_id << ", distance: " << min_distance << std::endl;
        return true;
    }
    
    return false;
}

void LoopClosureSimple::ProcessLoopClosure()
{
    // This would be for more complex processing in background
    while (running_.load()) {
        // Background loop closure processing
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}