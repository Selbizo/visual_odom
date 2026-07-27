#ifndef LOOP_CLOSURE_H
#define LOOP_CLOSURE_H

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <vector>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>

#include "Frame.h"

struct KeyFrame {
    int id;
    cv::Mat descriptor;       // Deep learning features (1000-dim)
    cv::Mat orb_descriptor;   // ORB keypoint descriptors (256-dim per descriptor)
    std::vector<int> desc_feat_indx;
    cv::Mat rotation;
    cv::Mat translation;
    std::vector<cv::Point2f> keypoints;
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> keypoints_matched;
    bool is_keyframe;
    cv::Mat full_pose; // 4x4 transformation matrix
    cv::Mat last_keyframe_pose; // 4x4 pose of previous keyframe (for distance calculation)
    
    KeyFrame() : id(0), is_keyframe(false) {}
};

class LoopClosure {
public:
    LoopClosure();
    ~LoopClosure();
    
    void setCameraParameters(const cv::Mat& projMatl, const cv::Mat& projMatr);
    void setParameters(int min_keypoints = 30, 
                       float weak_threshold = 0.7f, 
                       float strong_threshold = 0.85f,
                       int max_weak_candidates = 5,
                       int min_match_count = 20,
                       float keyframe_distance_meters = 50.0f);
    
    // world_pose: 4x4 CV_64F ACCUMULATED pose (frame_pose) of this frame in world/start
    // coordinates. This is REQUIRED for loop-closure correction math to be correct -
    // passing the frame-to-frame incremental rotation/translation here (as before) is wrong.
    bool addFrame(int frame_id, const cv::Mat& image_left, const cv::Mat& image_right,
                  const std::vector<cv::Point2f>& keypoints_left,
                  const std::vector<cv::Point2f>& keypoints_right,
                  const cv::Mat& rotation, const cv::Mat& translation,
                  const cv::Mat& points3D, const cv::Mat& world_pose,
                  bool force_keyframe = false);
    
    bool detectLoop();
    
    cv::Mat getLoopCorrection();
    cv::Mat getLoopRotation();
    cv::Mat getLoopTranslation();
    
    int getLastKeyframeId() const { return last_keyframe_id_; }
    int getCurrentKeyframeId() const { return current_keyframe_id_; }
    bool isLoopDetected() const { return loop_detected_; }
    int getCandidateKeyframeId() const { return candidate_keyframe_id_; }
    
    bool needsCorrection() const { return needs_correction_; }
    
    float getCurrentSimilarity() const { return current_similarity_; }
    int getMatchCount() const { return static_cast<int>(current_match_count_); }
    cv::Mat getCurrentDescriptor() const { return current_desc_.clone(); }
    cv::Mat getCandidateDescriptor() const { return candidate_desc_.clone(); }
    cv::Mat getKeyframeDescriptor(int index) const { 
        if (index < 0 || index >= static_cast<int>(keyframes_.size())) return cv::Mat();
        return keyframes_[index].descriptor.clone();
    }
    float computeFrameSimilarity(int frame_id, const cv::Mat& desc) { 
        for (const auto& kf : keyframes_) {
            if (kf.id == frame_id && !kf.descriptor.empty()) {
                return computeSimilarity(desc, kf.descriptor);
            }
        }
        return 0.0f;
    }
    
    void setMaxPoseDistance(float value) { max_pose_distance_between_loop_keyframes_ = value; }
    void setMaxPoseDifference(float value) { max_pose_differnece_between_old_new_ = value; }
    void setMinLoopGap(int value) { min_loop_gap_ = value; }
    void setDebugMode(bool debug) { debug_mode_ = debug; }
    void setMaxLoopDistance(float value) { max_loop_distance_meters_ = value; }
    void setMinKeyframeDistance(float value) { min_keyframe_distance_meters_ = value; }
    
    void applyCorrectionToKeyframes();
    std::vector<KeyFrame> getKeyframes();
    int getKeyframeCount();
    int getRawKeyframeCount();
    
private:
    bool extractDeepFeatures(const cv::Mat& image, cv::Mat& feature_vec);
    bool extractKeypointDescriptors(const cv::Mat& image,
                                   const std::vector<cv::Point2f>& keypoints,
                                   cv::Mat& descriptors,
                                   std::vector<int>& feature_indices);
    bool matchDescriptors(const cv::Mat& desc1, const cv::Mat& desc2,
                         std::vector<cv::DMatch>& matches);
    bool poseCorrectionPnP(const std::vector<cv::Point3f>& points3D,
                          const std::vector<cv::Point2f>& points2D,
                          cv::Mat& rotation, cv::Mat& translation);
    
    float computeSimilarity(const cv::Mat& vec1, const cv::Mat& vec2);
    
    bool isKeyframe(const cv::Mat& points3D, int min_points = 50, const cv::Mat& current_pose = cv::Mat(), const cv::Mat& last_kf_pose = cv::Mat(), float keyframe_distance_meters = 50.0f);
    
    std::string model_path_;
    
    cv::Mat projMatl_;
    cv::Mat projMatr_;
    
    int min_keypoints_;
    float weak_threshold_;
    float strong_threshold_;
    int max_weak_candidates_;
    int min_match_count_;
    
    // Loop closure validation parameters
    float max_pose_distance_between_loop_keyframes_ = 50.0;
    float max_pose_differnece_between_old_new_ = 10.0;
    int min_loop_gap_ = 20;
    float max_loop_distance_meters_ = 50.0f;
    float min_keyframe_distance_meters_ = 50.0f;
    
    int last_keyframe_id_;
    int current_keyframe_id_;
    bool loop_detected_;
    int candidate_keyframe_id_;
    int candidate_index_;   // index of matched candidate inside keyframes_ (needed to
                              // interpolate the correction only over the affected span)
    bool needs_correction_;
    
    float keyframe_distance_meters_ = 50.0f;
    
    cv::Mat loop_rotation_;
    cv::Mat loop_translation_;
    cv::Mat loop_correction_;
    
    float current_similarity_ = 0.0f;
    int current_match_count_ = 0;
    cv::Mat current_desc_;
    cv::Mat candidate_desc_;
    
    bool debug_mode_ = false;
    
    std::vector<KeyFrame> keyframes_;
    
    cv::Ptr<cv::dnn::Net> network_;
    cv::Ptr<cv::ORB> orb_descriptor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    
    std::mutex keyframe_mutex_;
};

#endif
