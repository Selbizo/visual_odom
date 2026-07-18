#ifndef LOOPCLOSURE_SIMPLE_H
#define LOOPCLOSURE_SIMPLE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

// Simplified loop closure for visual odometry system
class LoopClosureSimple
{
public:
    LoopClosureSimple();
    ~LoopClosureSimple();
    
    // Add a frame to be considered for loop closure
    void AddFrame(const cv::Mat& image, const cv::Mat& pose);
    
    // Check for potential loops (simplified version)
    bool CheckForLoop(int current_frame_id, const cv::Mat& current_pose);
    
    // Enable/disable loop closure
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    
private:
    // Store keyframes for loop detection
    std::vector<cv::Mat> keyframe_images_;
    std::vector<cv::Mat> keyframe_poses_;
    std::vector<int> keyframe_ids_;
    
    // Loop closure parameters
    bool enabled_;
    int min_keyframes_for_loop_;
    double max_pose_distance_;
    double min_pose_distance_;
    
    // Thread for background processing
    std::atomic<bool> running_;
    std::thread processing_thread_;
    
    // Internal methods
    void ProcessLoopClosure();
};

#endif // LOOPCLOSURE_SIMPLE_H