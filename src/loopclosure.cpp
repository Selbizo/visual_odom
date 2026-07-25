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
      candidate_index_(-1),
      needs_correction_(false) {
    
    std::string model_path = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/dnn_weights/mobilenet_v2_simplified.onnx";
    
    network_ = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(model_path));
    
    if (network_.empty()) {
        std::cerr << "[LoopClosure] Failed to load MobileNetV2 ONNX model" << std::endl;
    } else {
        std::cerr << "[LoopClosure] MobileNetV2 model loaded successfully" << std::endl;
        std::vector<cv::String> layerNames = network_->getLayerNames();
        std::cerr << "[LoopClosure] Network layers:" << std::endl;
        for (const auto& name : layerNames) {
            std::cerr << "  - " << name << std::endl;
        }
    }
    
    orb_descriptor_ = cv::ORB::create(200);
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
        std::cerr << "[LoopClosure] Network is empty!" << std::endl;
        return false;
    }
    
    cv::Mat dst;
    if (image.empty()) {
        std::cerr << "[LoopClosure] Input image is empty!" << std::endl;
        return false;
    }
    
    if (image.channels() == 1) {
        cv::cvtColor(image, dst, cv::COLOR_GRAY2RGB);
    } else {
        image.copyTo(dst);
    }
    
    if (dst.empty()) {
        std::cerr << "[LoopClosure] Converted image is empty!" << std::endl;
        return false;
    }
    
    cv::Mat blurred;
    cv::GaussianBlur(dst, blurred, cv::Size(7, 7), 0);
    
    cv::Mat blob;
    cv::dnn::blobFromImage(blurred, blob, 1.0/255.0, cv::Size(224, 224),
                           cv::Scalar(0.485, 0.456, 0.406), true, false);
    
    if (blob.empty()) {
        std::cerr << "[LoopClosure] Blob is empty!" << std::endl;
        return false;
    }
    
    network_->setInput(blob);
    
    std::string feature_layer_name = "output";
    cv::Mat output = network_->forward(feature_layer_name);
    
    if (output.empty()) {
        std::cerr << "[LoopClosure] Network output is empty! Layer name: " << feature_layer_name << std::endl;
        return false;
    }
    
    // std::cout << "[LoopClosure] Network output shape: " << output.size << std::endl;
    
    output.copyTo(feature_vec);
    
    float norm = cv::norm(feature_vec);
    if (norm > 1e-6) {
        feature_vec /= norm;
        // std::cout << "[LoopClosure] Feature extracted, norm: " << norm << std::endl;
    } else {
        std::cerr << "[LoopClosure] Feature norm is too small: " << norm << std::endl;
        return false;
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
    // std::cerr << "[LoopClosure] computeSimilarity called: vec1=" << vec1.size() << " vec2=" << vec2.size() << std::endl;
    
    if (vec1.empty() || vec2.empty()) {
        std::cerr << "[LoopClosure] computeSimilarity: empty input" << std::endl;
        return 0.0f;
    }
    
    if (vec1.type() != vec2.type()) {
        std::cerr << "[LoopClosure] computeSimilarity: type mismatch" << std::endl;
        return 0.0f;
    }
    
    cv::Mat v1 = vec1.reshape(1, 1);
    cv::Mat v2 = vec2.reshape(1, 1);
    
    // std::cerr << "[LoopClosure] computeSimilarity: v1=" << v1.size() << " v2=" << v2.size() << std::endl;
    
    if (v1.cols != v2.cols) {
        // std::cerr << "[LoopClosure] computeSimilarity: column mismatch " << v1.cols << " vs " << v2.cols << std::endl;
        return 0.0f;
    }
    
    int len = v1.cols;
    float sum = 0.0f;
    
    // std::cerr << "[LoopClosure] computeSimilarity: computing dot product of length " << len << std::endl;
    
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
    } else {
        std::cerr << "[LoopClosure] computeSimilarity: unsupported type" << std::endl;
        return 0.0f;
    }
    
    // std::cerr << "[LoopClosure] computeSimilarity: result=" << sum << std::endl;
    
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
    
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    K.at<double>(0, 0) = projMatl_.at<float>(0, 0);
    K.at<double>(1, 1) = projMatl_.at<float>(1, 1);
    K.at<double>(0, 2) = projMatl_.at<float>(0, 2);
    K.at<double>(1, 2) = projMatl_.at<float>(1, 2);
    K.at<double>(2, 2) = 1.0;
    
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
                            const cv::Mat& points3D, const cv::Mat& world_pose,
                            bool force_keyframe) {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    
    bool is_kf = force_keyframe || isKeyframe(points3D, min_keypoints_);
    
    KeyFrame kf;
    kf.id = frame_id;
    image_left.copyTo(kf.image);
    kf.rotation = rotation.clone();
    kf.translation = translation.clone();
    kf.keypoints = keypoints_left;
    kf.points3D = points3D;
    kf.is_keyframe = is_kf;
    
    // IMPORTANT: full_pose must be the ACCUMULATED world pose of this frame (frame_pose
    // in main.cpp), NOT the frame-to-frame incremental rotation/translation. The loop
    // correction math below relies on candidate_kf.full_pose / current_kf.full_pose being
    // expressed in the same, consistent world coordinate system.
    if (world_pose.rows == 4 && world_pose.cols == 4) {
        kf.full_pose = world_pose.clone();
        if (kf.full_pose.type() != CV_64F) {
            kf.full_pose.convertTo(kf.full_pose, CV_64F);
        }
    } else {
        std::cerr << "[LoopClosure] WARNING: world_pose is not 4x4, falling back to identity "
                   "for keyframe " << frame_id << " (loop correction will be unreliable)" << std::endl;
        kf.full_pose = cv::Mat::eye(4, 4, CV_64F);
    }
    
    if (is_kf) {
        cv::Mat orb_descriptors;
        // std::cout << "[LoopClosure] Adding keyframe " << frame_id << ", image size: " << image_left.size() 
        //           << ", channels: " << image_left.channels() << std::endl;
        
        if (extractDeepFeatures(image_left, kf.descriptor)) {
            // std::cout << "[LoopClosure] Deep features extracted, size: " << kf.descriptor.size() 
            //           << ", norm: " << cv::norm(kf.descriptor) << std::endl;
            
            if (extractKeypointDescriptors(image_left, keypoints_left, orb_descriptors, kf.desc_feat_indx)) {
                orb_descriptors.copyTo(kf.orb_descriptor);
                keyframes_.push_back(kf);
                last_keyframe_id_ = frame_id;
                // std::cout << "[LoopClosure] Keyframe " << frame_id << " added successfully" << std::endl;
            } else {
                std::cerr << "[LoopClosure] Failed to extract ORB descriptors for keyframe " << frame_id << std::endl;
            }
        } else {
            std::cerr << "[LoopClosure] Failed to extract deep features for keyframe " << frame_id << std::endl;
        }
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
    size_t max_sim_index = 0;
    int num_weak_candidates = 0;
    
    std::cout << "[LoopClosure] Checking " << keyframes_.size() << " keyframes for loop closure..." << std::endl;
    
    for (size_t i = 0; i < keyframes_.size() - 1; i++) {
        const KeyFrame& kf = keyframes_[i];
        
        if (!kf.is_keyframe) {
            continue;
        }
        
        if (current_kf.id - kf.id < 20) {
            continue;
        }
        
        float similarity = computeSimilarity(current_desc, kf.descriptor);
        // std::cout << "[LoopClosure] Comparing frame " << current_kf.id << " with " << kf.id << ": similarity=" << similarity << std::endl;
        
        // if (similarity > 0.7f) {
        //     std::cout << "[LoopClosure] Potential match: current=" << current_kf.id 
        //               << ", candidate=" << kf.id << ", similarity=" << similarity << std::endl;
        // }
        
        if (similarity > max_similarity) {
            max_similarity = similarity;
            max_sim_index = i;
        }
        
        if (similarity > weak_threshold_) {
            num_weak_candidates++;
        }
    }
    
    std::cout << "[LoopClosure] Max similarity: " << max_similarity << " (id=" << keyframes_[max_sim_index].id << ")" << std::endl;
    std::cout << "[LoopClosure] Weak candidates: " << num_weak_candidates << std::endl;
    
    if (max_similarity < strong_threshold_ || keyframes_.empty()) {
        return false;
    }
    
    KeyFrame& candidate_kf = keyframes_[max_sim_index];
    
    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(candidate_kf.orb_descriptor, current_kf.orb_descriptor, matches)) {
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
    
    // R_correction/t_correction from solvePnPRansac describe the CURRENT camera pose
    // expressed in the CANDIDATE keyframe's own LOCAL coordinate frame (because
    // candidate_kf.points3D were triangulated relative to the candidate camera at
    // capture time). This is NOT yet a world-frame correction - it must be composed
    // with the candidate's known world pose to find out where the current frame
    // *should* be in world coordinates, and compared against where odometry drift
    // currently thinks it is.
    cv::Mat T_pnp = cv::Mat::eye(4, 4, CV_64F);
    R_correction.copyTo(T_pnp(cv::Rect(0, 0, 3, 3)));
    t_correction.copyTo(T_pnp(cv::Rect(3, 0, 1, 3)));
    
    if (candidate_kf.full_pose.empty() || current_kf.full_pose.empty()) {
        std::cerr << "[LoopClosure] Missing world pose on keyframe(s), cannot correct." << std::endl;
        return false;
    }
    
    // Where the current frame SHOULD be in world coordinates, according to the loop match:
    cv::Mat T_current_estimated_world = candidate_kf.full_pose * T_pnp.inv();
    
    // Where odometry (with accumulated drift) currently thinks the current frame is:
    cv::Mat T_current_naive_world = current_kf.full_pose;
    
    double det = cv::determinant(T_current_naive_world(cv::Rect(0, 0, 3, 3)));
    if (std::abs(det) < 1e-9) {
        std::cerr << "[LoopClosure] Degenerate current world pose, skipping correction." << std::endl;
        return false;
    }
    
    // Correction delta (world frame) that removes the accumulated drift between
    // candidate and current:
    cv::Mat T_delta = T_current_estimated_world * T_current_naive_world.inv();
    
    loop_rotation_ = T_delta(cv::Rect(0, 0, 3, 3)).clone();
    loop_translation_ = T_delta(cv::Rect(3, 0, 1, 3)).clone();
    loop_correction_ = T_delta.clone();
    
    candidate_keyframe_id_ = candidate_kf.id;
    candidate_index_ = static_cast<int>(max_sim_index);
    loop_detected_ = true;
    needs_correction_ = true;
    
    return true;
}

void LoopClosure::applyCorrectionToKeyframes() {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    
    if (!needs_correction_ || keyframes_.empty() || candidate_index_ < 0) {
        return;
    }
    
    int current_index = static_cast<int>(keyframes_.size()) - 1;
    int span = current_index - candidate_index_;
    
    if (span <= 0) {
        return;
    }
    
    // Keyframes at/before the candidate are treated as the trusted anchor and are left
    // untouched. Keyframes strictly between candidate and current get a correction that
    // is linearly interpolated from 0 (at the candidate) up to the full delta (at the
    // current frame), instead of slamming the *entire* history (including the trusted
    // anchor) with the same rigid transform. This is a simplified stand-in for real
    // pose-graph optimization (no g2o in this project), but avoids re-warping keyframes
    // that were already correct.
    cv::Mat R_delta = loop_correction_(cv::Rect(0, 0, 3, 3));
    cv::Mat t_delta = loop_correction_(cv::Rect(3, 0, 1, 3));
    cv::Mat rvec_delta;
    cv::Rodrigues(R_delta, rvec_delta);
    
    int corrected_count = 0;
    for (int i = candidate_index_ + 1; i < current_index; i++) {
        double alpha = static_cast<double>(i - candidate_index_) / static_cast<double>(span);
        
        cv::Mat rvec_partial = rvec_delta * alpha;
        cv::Mat R_partial;
        cv::Rodrigues(rvec_partial, R_partial);
        cv::Mat t_partial = t_delta * alpha;
        
        cv::Mat T_partial = cv::Mat::eye(4, 4, CV_64F);
        R_partial.copyTo(T_partial(cv::Rect(0, 0, 3, 3)));
        t_partial.copyTo(T_partial(cv::Rect(3, 0, 1, 3)));
        
        KeyFrame& kf = keyframes_[i];
        cv::Mat new_pose = T_partial * kf.full_pose;
        new_pose.copyTo(kf.full_pose);
        kf.rotation = kf.full_pose(cv::Rect(0, 0, 3, 3)).clone();
        kf.translation = kf.full_pose(cv::Rect(3, 0, 1, 3)).clone();
        corrected_count++;
    }
    
    // The current (last) keyframe gets the FULL delta - this must be kept consistent
    // with whatever main.cpp applies to its own live frame_pose for this frame.
    {
        KeyFrame& kf = keyframes_[current_index];
        cv::Mat new_pose = loop_correction_ * kf.full_pose;
        new_pose.copyTo(kf.full_pose);
        kf.rotation = kf.full_pose(cv::Rect(0, 0, 3, 3)).clone();
        kf.translation = kf.full_pose(cv::Rect(3, 0, 1, 3)).clone();
    }
    
    std::cout << "[LoopClosure] Interpolated correction applied to " << corrected_count
               << " keyframes between candidate " << keyframes_[candidate_index_].id
               << " and current " << keyframes_[current_index].id << std::endl;
}

std::vector<KeyFrame> LoopClosure::getKeyframes() {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    return keyframes_;
}

int LoopClosure::getKeyframeCount() {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    return static_cast<int>(keyframes_.size());
}

int LoopClosure::getRawKeyframeCount() {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    return static_cast<int>(keyframes_.size());
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
