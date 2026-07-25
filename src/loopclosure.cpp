#include "loopclosure.h"

#include <iostream>
#include <algorithm>

LoopClosure::LoopClosure() 
    : min_keypoints_(30),
      weak_threshold_(0.7f),
      strong_threshold_(0.85f),
      max_weak_candidates_(5),
      min_match_count_(20),
      last_keyframe_id_(0),
      current_keyframe_id_(0),
      loop_detected_(false),
      candidate_keyframe_id_(-1),
      needs_correction_(false) {
    
    std::string model_path = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/dnn_weights/mobilenet_v2_simplified.onnx";
    
    network_ = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(model_path));
    
    if (network_.empty()) {
        std::cerr << "[LoopClosure] Failed to load MobileNetV2 ONNX model" << std::endl;
    } else {
        std::cerr << "[LoopClosure] MobileNetV2 model loaded successfully" << std::endl;
    }
    
    orb_descriptor_ = cv::ORB::create(400);
    matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

LoopClosure::~LoopClosure() {
}

void LoopClosure::setCameraParameters(const cv::Mat& projMatl, const cv::Mat& projMatr) {
    projMatl_ = projMatl.clone();
    projMatr_ = projMatr.clone();
}

void LoopClosure::setParameters(int min_keypoints, float weak_threshold,
                                float strong_threshold, int max_weak_candidates,
                                int min_match_count) {
    min_keypoints_ = min_keypoints;
    weak_threshold_ = weak_threshold;
    strong_threshold_ = strong_threshold;
    max_weak_candidates_ = max_weak_candidates;
    min_match_count_ = min_match_count;
}

bool LoopClosure::isKeyframe(const cv::Mat& points3D, int min_points) {
    return !points3D.empty() && points3D.rows >= min_points;
}

bool LoopClosure::extractDeepFeatures(const cv::Mat& image, cv::Mat& feature_vec) {
    if (network_.empty()) {
        return false;
    }
    
    cv::Mat dst;
    if (image.channels() == 1) {
        cv::cvtColor(image, dst, cv::COLOR_GRAY2RGB);
    } else {
        image.copyTo(dst);
    }
    
    cv::Mat blurred;
    cv::GaussianBlur(dst, blurred, cv::Size(7, 7), 0);
    
    cv::Mat blob;
    cv::dnn::blobFromImage(blurred, blob, 1.0/255.0, cv::Size(224, 224),
                          cv::Scalar(0.485, 0.456, 0.406), true, false);
    
    network_->setInput(blob);
    
    std::string feature_layer_name = "output";
    cv::Mat output = network_->forward(feature_layer_name);
    
    if (output.empty()) {
        return false;
    }
    
    output.copyTo(feature_vec);
    
    float norm = cv::norm(feature_vec);
    if (norm > 1e-6) {
        feature_vec /= norm;
    }
    
    return true;
}

bool LoopClosure::extractKeypointDescriptors(const cv::Mat& image,
                                            const std::vector<cv::Point2f>& keypoints,
                                            cv::Mat& descriptors,
                                            std::vector<int>& feature_indices) {
    if (orb_descriptor_.empty()) {
        return false;
    }
    
    std::vector<cv::KeyPoint> keypoints_cv;
    for (const auto& kp : keypoints) {
        keypoints_cv.push_back(cv::KeyPoint(kp.x, kp.y, 7.0f));
    }
    
    if (keypoints_cv.empty()) {
        return false;
    }
    
    orb_descriptor_->compute(image, keypoints_cv, descriptors);
    
    if (descriptors.empty()) {
        return false;
    }
    
    feature_indices.clear();
    for (size_t i = 0; i < keypoints_cv.size(); i++) {
        for (size_t j = 0; j < keypoints.size(); j++) {
            if (std::abs(keypoints_cv[i].pt.x - keypoints[j].x) < 1.0f &&
                std::abs(keypoints_cv[i].pt.y - keypoints[j].y) < 1.0f) {
                feature_indices.push_back(static_cast<int>(j));
                break;
            }
        }
    }
    
    return true;
}

bool LoopClosure::matchDescriptors(const cv::Mat& desc1, const cv::Mat& desc2,
                                   std::vector<cv::DMatch>& matches) {
    if (matcher_.empty() || desc1.empty() || desc2.empty()) {
        return false;
    }
    
    matcher_->match(desc1, desc2, matches);
    
    if (matches.empty()) {
        return false;
    }
    
    auto min_it = std::min_element(matches.begin(), matches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });
    
    double distance_threshold = std::max(2.0 * min_it->distance, 30.0);
    
    std::vector<cv::DMatch> filtered_matches;
    for (const auto& match : matches) {
        if (match.distance <= distance_threshold) {
            filtered_matches.push_back(match);
        }
    }
    
    matches = filtered_matches;
    
    return !matches.empty();
}

float LoopClosure::computeSimilarity(const cv::Mat& vec1, const cv::Mat& vec2) {
    if (vec1.empty() || vec2.empty() || vec1.rows != vec2.rows) {
        return 0.0f;
    }
    
    // Ensure both matrices are 1D vectors (single column or row)
    cv::Mat v1 = vec1.reshape(1, vec1.total());
    cv::Mat v2 = vec2.reshape(1, vec2.total());
    
    if (v1.total() != v2.total()) {
        return 0.0f;
    }
    
    int len = static_cast<int>(v1.total());
    float sum = 0.0f;
    
    if (v1.type() == CV_32F) {
        const float* p1 = v1.ptr<float>();
        const float* p2 = v2.ptr<float>();
        for (int i = 0; i < len; i++) {
            sum += p1[i] * p2[i];
        }
    } else if (v1.type() == CV_64F) {
        const double* p1 = v1.ptr<double>();
        const double* p2 = v2.ptr<double>();
        for (int i = 0; i < len; i++) {
            sum += static_cast<float>(p1[i] * p2[i]);
        }
    }
    
    return sum;
}

bool LoopClosure::poseCorrectionPnP(const std::vector<cv::Point3f>& points3D,
                                    const std::vector<cv::Point2f>& points2D,
                                    cv::Mat& rotation, cv::Mat& translation) {
    if (points3D.size() < 6 || points2D.size() < 6) {
        return false;
    }
    
    cv::Mat dist_coeff = cv::Mat::zeros(4, 1, CV_64F);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
                 projMatl_.at<double>(0, 0), 0, projMatl_.at<double>(0, 2),
                 0, projMatl_.at<double>(1, 1), projMatl_.at<double>(1, 2),
                 0, 0, 1);
    
    std::vector<int> inliers;
    cv::solvePnPRansac(points3D, points2D, K, dist_coeff, rvec, tvec,
                      false, 100, 5.991, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    
    if (inliers.size() < min_match_count_) {
        return false;
    }
    
    cv::Rodrigues(rvec, rotation);
    tvec.copyTo(translation);
    
    return true;
}

bool LoopClosure::addFrame(int frame_id, const cv::Mat& image_left, const cv::Mat& image_right,
                          const std::vector<cv::Point2f>& keypoints_left,
                          const std::vector<cv::Point2f>& keypoints_right,
                          const cv::Mat& rotation, const cv::Mat& translation,
                          const cv::Mat& points3D) {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    
    bool is_kf = isKeyframe(points3D, min_keypoints_);
    
    KeyFrame kf;
    kf.id = frame_id;
    image_left.copyTo(kf.image);
    kf.rotation = rotation.clone();
    kf.translation = translation.clone();
    kf.keypoints = keypoints_left;
    kf.points3D = points3D;
    kf.is_keyframe = is_kf;
    
    if (is_kf) {
        if (!extractDeepFeatures(image_left, kf.descriptor)) {
            std::cerr << "[LoopClosure] Failed to extract deep features for frame " << frame_id << std::endl;
            kf.descriptor = cv::Mat::zeros(1, 1280, CV_32F);
        }
        
        if (!extractKeypointDescriptors(image_left, keypoints_left, kf.descriptor, kf.desc_feat_indx)) {
            std::cerr << "[LoopClosure] Failed to extract ORB descriptors for frame " << frame_id << std::endl;
            kf.descriptor = cv::Mat::zeros(1, 1280, CV_32F);
        }
        
        keyframes_.push_back(kf);
        last_keyframe_id_ = frame_id;
    } else {
        keyframes_.push_back(kf);
    }
    
    current_keyframe_id_ = frame_id;
    
    return true;
}

bool LoopClosure::detectLoop() {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    
    loop_detected_ = false;
    candidate_keyframe_id_ = -1;
    needs_correction_ = false;
    
    if (keyframes_.empty()) {
        return false;
    }
    
    KeyFrame& current_kf = keyframes_.back();
    
    if (!current_kf.is_keyframe) {
        return false;
    }
    
    cv::Mat current_desc = current_kf.descriptor;
    if (current_desc.empty()) {
        return false;
    }
    
    float max_similarity = 0.0f;
    int max_sim_id = -1;
    int num_weak_candidates = 0;
    
    for (size_t i = 0; i < keyframes_.size() - 1; i++) {
        const KeyFrame& kf = keyframes_[i];
        
        if (!kf.is_keyframe) {
            continue;
        }
        
        if (current_kf.id - kf.id < 20) {
            continue;
        }
        
        float similarity = computeSimilarity(current_desc, kf.descriptor);
        
        if (similarity > max_similarity) {
            max_similarity = similarity;
            max_sim_id = kf.id;
        }
        
        if (similarity > weak_threshold_) {
            num_weak_candidates++;
        }
    }
    
    if (max_similarity < strong_threshold_ || num_weak_candidates > max_weak_candidates_) {
        return false;
    }
    
    KeyFrame& candidate_kf = keyframes_[max_sim_id];
    
    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(candidate_kf.descriptor, current_kf.descriptor, matches)) {
        return false;
    }
    
    if (static_cast<int>(matches.size()) < min_match_count_) {
        return false;
    }
    
    std::vector<cv::Point3f> points3D_cand;
    std::vector<cv::Point2f> points2D_curr;
    
    for (const auto& match : matches) {
        if (match.queryIdx >= static_cast<int>(candidate_kf.desc_feat_indx.size()) ||
            match.trainIdx >= static_cast<int>(current_kf.desc_feat_indx.size())) {
            continue;
        }
        
        int cand_idx = candidate_kf.desc_feat_indx[match.queryIdx];
        int curr_idx = current_kf.desc_feat_indx[match.trainIdx];
        
        if (cand_idx >= 0 && cand_idx < static_cast<int>(candidate_kf.points3D.size()) &&
            curr_idx >= 0 && curr_idx < static_cast<int>(current_kf.keypoints.size())) {
            points3D_cand.push_back(candidate_kf.points3D[cand_idx]);
            points2D_curr.push_back(current_kf.keypoints[curr_idx]);
        }
    }
    
    if (static_cast<int>(points3D_cand.size()) < min_match_count_) {
        return false;
    }
    
    cv::Mat R_correction, t_correction;
    if (!poseCorrectionPnP(points3D_cand, points2D_curr, R_correction, t_correction)) {
        return false;
    }
    
    loop_rotation_ = R_correction.clone();
    loop_translation_ = t_correction.clone();
    
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R_correction.copyTo(T(cv::Rect(0, 0, 3, 3)));
    t_correction.copyTo(T(cv::Rect(3, 0, 1, 3)));
    T.copyTo(loop_correction_);
    
    candidate_keyframe_id_ = candidate_kf.id;
    loop_detected_ = true;
    needs_correction_ = true;
    
    std::cerr << "[LoopClosure] Loop detected! Frame " << current_kf.id 
              << " matches with frame " << candidate_kf.id 
              << " (similarity: " << max_similarity << ", matches: " << matches.size() << ")" << std::endl;
    
    return true;
}

cv::Mat LoopClosure::getLoopCorrection() {
    return loop_correction_.clone();
}

cv::Mat LoopClosure::getLoopRotation() {
    return loop_rotation_.clone();
}

cv::Mat LoopClosure::getLoopTranslation() {
    return loop_translation_.clone();
}
