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
    cv::Mat image;
    cv::Mat depth;
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
                       int min_match_count = 20);
    
    bool addFrame(int frame_id, const cv::Mat& image_left, const cv::Mat& image_right,
                  const std::vector<cv::Point2f>& keypoints_left,
                  const std::vector<cv::Point2f>& keypoints_right,
                  const cv::Mat& rotation, const cv::Mat& translation,
                  const cv::Mat& points3D, bool force_keyframe = false);
    
    bool detectLoop();
    
    cv::Mat getLoopCorrection();
    cv::Mat getLoopRotation();
    cv::Mat getLoopTranslation();
    
    int getLastKeyframeId() const { return last_keyframe_id_; }
    int getCurrentKeyframeId() const { return current_keyframe_id_; }
    bool isLoopDetected() const { return loop_detected_; }
    int getCandidateKeyframeId() const { return candidate_keyframe_id_; }
    
    bool needsCorrection() const { return needs_correction_; }
    
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
    
    bool isKeyframe(const cv::Mat& points3D, int min_points = 50);
    
    std::string model_path_;
    
    cv::Mat projMatl_;
    cv::Mat projMatr_;
    
    int min_keypoints_;
    float weak_threshold_;
    float strong_threshold_;
    int max_weak_candidates_;
    int min_match_count_;
    
    int last_keyframe_id_;
    int current_keyframe_id_;
    bool loop_detected_;
    int candidate_keyframe_id_;
    bool needs_correction_;
    
    cv::Mat loop_rotation_;
    cv::Mat loop_translation_;
    cv::Mat loop_correction_;
    
    std::vector<KeyFrame> keyframes_;
    
    cv::Ptr<cv::dnn::Net> network_;
    cv::Ptr<cv::ORB> orb_descriptor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    
    std::mutex keyframe_mutex_;
};

#endif
