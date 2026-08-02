#include "loopclosure.h"

#include <iostream>
#include <algorithm>
#include <cmath>

LoopClosure::LoopClosure() 
    : min_keypoints_(30),
      weak_threshold_(0.7f),
      strong_threshold_(0.85f),
      max_weak_candidates_(5),
      min_match_count_(20),
       keyframe_distance_meters_(50.0f),
       last_keyframe_id_(0),
     current_keyframe_id_(0),
     loop_detected_(false),
     candidate_keyframe_id_(-1),
     candidate_index_(-1),
     needs_correction_(false),
     debug_mode_(false),
     current_similarity_(0.0f),
     current_match_count_(0),
      max_loop_distance_meters_(50.0f),
      min_keyframe_distance_meters_(50.0f) {
    
    std::string model_path = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/dnn_weights/mobilenet_v2_simplified.onnx";
    
    network_ = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(model_path));
    
    if (network_.empty()) {
        std::cerr << "[LoopClosure] Failed to load MobileNetV2 ONNX model" << std::endl;
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
                                int min_match_count, float keyframe_distance_meters) {
    min_keypoints_ = min_keypoints;
    weak_threshold_ = weak_threshold;
    strong_threshold_ = strong_threshold;
    max_weak_candidates_ = max_weak_candidates;
    min_match_count_ = min_match_count;
    keyframe_distance_meters_ = keyframe_distance_meters;
}

bool LoopClosure::isKeyframe(const cv::Mat& points3D, int min_points, const cv::Mat& current_pose, const cv::Mat& last_kf_pose, float keyframe_distance_meters) {
    if (!points3D.empty() && points3D.rows >= min_points) {
        if (!current_pose.empty() && !last_kf_pose.empty() && 
            current_pose.rows == 4 && current_pose.cols == 4 &&
            last_kf_pose.rows == 4 && last_kf_pose.cols == 4) {
            
            cv::Mat t1 = current_pose(cv::Rect(3, 0, 1, 3));
            cv::Mat t2 = last_kf_pose(cv::Rect(3, 0, 1, 3));
            
            double distance = cv::norm(t1 - t2);
            return distance >= keyframe_distance_meters;
        }
        return true;
    }
    return false;
}

bool LoopClosure::extractDeepFeatures(const cv::Mat& image, cv::Mat& feature_vec) {
    if (network_.empty()) {
        return false;
    }
    
    cv::Mat dst;
    if (image.empty()) {
        return false;
    }
    
    if (image.channels() == 1) {
        cv::cvtColor(image, dst, cv::COLOR_GRAY2RGB);
    } else {
        image.copyTo(dst);
    }
    
    if (dst.empty()) {
        return false;
    }
    
    cv::Mat blurred;
    cv::GaussianBlur(dst, blurred, cv::Size(7, 7), 0);
    
    cv::Mat blob;
    cv::dnn::blobFromImage(blurred, blob, 1.0/255.0, cv::Size(224, 224),
                           cv::Scalar(0.485, 0.456, 0.406), true, false);
    
    if (blob.empty()) {
        return false;
    }
    
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
    } else {
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
    feature_indices.resize(keypoints_cv.size(), -1);
    
    for (size_t i = 0; i < keypoints_cv.size(); i++) {
        for (size_t j = 0; j < keypoints.size(); j++) {
            if (std::abs(keypoints_cv[i].pt.x - keypoints[j].x) < 1.0f &&
                std::abs(keypoints_cv[i].pt.y - keypoints[j].y) < 1.0f) {
                feature_indices[i] = static_cast<int>(j);
                break;
            }
        }
    }
    
    // Remove any -1 values if they exist
    size_t write_idx = 0;
    for (size_t i = 0; i < feature_indices.size(); i++) {
        if (feature_indices[i] >= 0) {
            feature_indices[write_idx++] = feature_indices[i];
        }
    }
    feature_indices.resize(write_idx);
    
    return feature_indices.size() > 0;
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
    if (vec1.empty() || vec2.empty()) {
        return 0.0f;
    }
    
    cv::Mat v1 = vec1.reshape(1, 1);
    cv::Mat v2 = vec2.reshape(1, 1);
    
    if (v1.cols != v2.cols) {
        return 0.0f;
    }
    
    int len = v1.cols;
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
    } else {
        return 0.0f;
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
    
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    K.at<double>(0, 0) = projMatl_.at<float>(0, 0);
    K.at<double>(1, 1) = projMatl_.at<float>(1, 1);
    K.at<double>(0, 2) = projMatl_.at<float>(0, 2);
    K.at<double>(1, 2) = projMatl_.at<float>(1, 2);
    K.at<double>(2, 2) = 1.0;
    
    std::vector<int> inliers;
    bool success = cv::solvePnPRansac(points3D, points2D, K, dist_coeff, rvec, tvec,
                                      false, 200, 3.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    
    if (!success) {
        return false;
    }
    
    if (static_cast<int>(inliers.size()) < 8) {
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
    
    KeyFrame last_kf;
    bool has_last_kf = false;
    if (!keyframes_.empty()) {
        for (int i = static_cast<int>(keyframes_.size()) - 1; i >= 0; i--) {
            if (keyframes_[i].is_keyframe) {
                last_kf = keyframes_[i];
                has_last_kf = true;
                break;
            }
        }
    }
    
    bool is_kf = force_keyframe || isKeyframe(points3D, min_keypoints_, world_pose, has_last_kf ? last_kf.full_pose : cv::Mat(), keyframe_distance_meters_);
    
    if (is_kf && !force_keyframe && min_keyframe_distance_meters_ > 0 && !world_pose.empty() && world_pose.rows == 4 && world_pose.cols == 4) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& kf : keyframes_) {
            if (kf.is_keyframe && !kf.full_pose.empty() && kf.full_pose.rows == 4 && kf.full_pose.cols == 4) {
                cv::Mat t1 = world_pose(cv::Rect(3, 0, 1, 3));
                cv::Mat t2 = kf.full_pose(cv::Rect(3, 0, 1, 3));
                double dist = cv::norm(t1 - t2);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }
        if (min_dist < min_keyframe_distance_meters_) {
            if (debug_mode_) {
                std::cout << "[LoopClosure] Frame " << frame_id << " skipped: min distance to existing keyframe is " 
                         << min_dist << "m < " << min_keyframe_distance_meters_ << "m" << std::endl;
            }
            is_kf = false;
        }
    }
    
    KeyFrame kf;
    kf.id = frame_id;
    kf.rotation = rotation.clone();
    kf.translation = translation.clone();
    kf.keypoints = keypoints_left;
    kf.points3D = points3D;
    kf.is_keyframe = is_kf;
    kf.last_keyframe_pose = has_last_kf ? last_kf.full_pose.clone() : cv::Mat();
    
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
        kf.full_pose = cv::Mat::eye(4, 4, CV_64F);
        if (debug_mode_) {
            std::cerr << "[LoopClosure] WARNING: world_pose is not 4x4, falling back to identity "
                      "for keyframe " << frame_id << std::endl;
        }
    }
    
    if (is_kf) {
        last_keyframe_id_ = frame_id;
        if (debug_mode_) {
            std::cout << "[LoopClosure] Keyframe " << frame_id << " added" << std::endl;
        }
    }
    
    // Извлекаем дескрипторы для каждого кадра для проверки loop closure
    cv::Mat orb_descriptors;
    if (kf.descriptor.empty() && extractDeepFeatures(image_left, kf.descriptor)) {
        extractKeypointDescriptors(image_left, keypoints_left, orb_descriptors, kf.desc_feat_indx);
        orb_descriptors.copyTo(kf.orb_descriptor);
    }
    
    keyframes_.push_back(kf);
    
    current_keyframe_id_ = frame_id;
    
    return true;
}

bool LoopClosure::detectLoop() {
    std::lock_guard<std::mutex> lock(keyframe_mutex_);
    
    if (debug_mode_ && !keyframes_.empty()) {
        std::cout << "[LoopClosure] detectLoop called for frame " 
                  << keyframes_.back().id << ", keyframe count: " << keyframes_.size() << std::endl;
    }
    
    loop_detected_ = false;
    candidate_keyframe_id_ = -1;
    needs_correction_ = false;
    current_similarity_ = 0.0f;
    current_match_count_ = 0;
    current_desc_.release();
    candidate_desc_.release();
    
    if (keyframes_.empty()) {
        return false;
    }
    
    KeyFrame& current_kf = keyframes_.back();
    
    cv::Mat current_desc = current_kf.descriptor;
    if (current_desc.empty()) {
        return false;
    }
    
    float max_similarity = 0.0f;
    size_t max_sim_index = 0;
    int num_weak_candidates = 0;
    
    int num_checked = 0;
    for (size_t i = 0; i < keyframes_.size() - 1; i++) {
        const KeyFrame& kf = keyframes_[i];
        
        if (!kf.is_keyframe) {
            continue;
        }
        
        int id_diff = current_kf.id - kf.id;
        
        if (id_diff < min_loop_gap_) {
            continue;
        }
        
        if (max_loop_distance_meters_ > 0) {
            if (!kf.full_pose.empty() && !current_kf.full_pose.empty() &&
                kf.full_pose.rows == 4 && kf.full_pose.cols == 4 &&
                current_kf.full_pose.rows == 4 && current_kf.full_pose.cols == 4) {
                
                cv::Mat t_kf = kf.full_pose(cv::Rect(3, 0, 1, 3));
                cv::Mat t_current = current_kf.full_pose(cv::Rect(3, 0, 1, 3));
                double dist = cv::norm(t_kf - t_current);
                
                if (debug_mode_) {
                    std::cout << "[LoopClosure] Checking frame " << kf.id 
                             << ", world distance: " << dist << "m" << std::endl;
                }
                
                if (dist > max_loop_distance_meters_) {
                    if (debug_mode_) {
                        std::cout << "[LoopClosure] Skip (too far: " << dist << "m > " 
                                 << max_loop_distance_meters_ << "m)" << std::endl;
                    }
                    continue;
                }
            }
        }
        
        num_checked++;
        float similarity = computeSimilarity(current_desc, kf.descriptor);
        
        if (similarity > max_similarity) {
            max_similarity = similarity;
            max_sim_index = i;
        }
        
        if (similarity > weak_threshold_) {
            num_weak_candidates++;
            if (debug_mode_) {
                // std::cout << "[LoopClosure]  Weak candidate: frame " << kf.id 
                //          << ", similarity: " << similarity << std::endl;
            }
        }
    }
    
    if (debug_mode_) {
        std::cout << "[LoopClosure] Checked " << num_checked 
                 << " keyframes, max similarity: " << max_similarity 
                 << ", strong threshold: " << strong_threshold_ << std::endl;
    }
    
    if (max_similarity < strong_threshold_) {
        current_similarity_ = max_similarity;
        return false;
    }
    
    KeyFrame& candidate_kf = keyframes_[max_sim_index];
    current_similarity_ = max_similarity;
    candidate_desc_ = candidate_kf.descriptor.clone();
    current_desc_ = current_desc.clone();
    
    if (debug_mode_) {
        std::cout << "[LoopClosure] Strong match found: current frame " 
                 << current_kf.id << " vs candidate " << candidate_kf.id 
                 << ", similarity: " << max_similarity << std::endl;
    }
    
    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(candidate_kf.orb_descriptor, current_kf.orb_descriptor, matches)) {
        if (debug_mode_) {
            std::cout << "[LoopClosure] ORB matching failed" << std::endl;
        }
        return false;
    }
    
    current_match_count_ = matches.size();
    
    if (debug_mode_) {
        std::cout << "[LoopClosure] ORB matches found: " << matches.size() 
                 << " (min required: " << min_match_count_ << ")" << std::endl;
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
        return false;
    }
    
    // Where the current frame SHOULD be in world coordinates, according to the loop match:
    cv::Mat T_current_estimated_world = candidate_kf.full_pose * T_pnp.inv();
    
    // Where odometry (with accumulated drift) currently thinks the current frame is:
    cv::Mat T_current_naive_world = current_kf.full_pose;
    
    // Calculate correction delta
    cv::Mat T_delta = T_current_estimated_world * T_current_naive_world.inv();
    
    // Extract rotation and translation from correction
    cv::Mat R_delta = T_delta(cv::Rect(0, 0, 3, 3));
    cv::Mat t_delta = T_delta(cv::Rect(3, 0, 1, 3));
    
    // Check if correction is too large (loop between two far images is not reliable)
    cv::Mat rvec_delta;
    cv::Rodrigues(R_delta, rvec_delta);
    double pose_distance_rad = cv::norm(rvec_delta);
    double pose_distance_deg = pose_distance_rad * 180.0 / CV_PI;

    if (pose_distance_deg > max_pose_distance_between_loop_keyframes_) {
        return false;
    }

    // Check pose difference between new and old calculation
    cv::Mat T_diff = T_current_naive_world * T_current_estimated_world.inv();
    cv::Mat rvec_diff;
    cv::Rodrigues(T_diff(cv::Rect(0, 0, 3, 3)), rvec_diff);
    double pose_diff = cv::norm(rvec_diff);

    if (pose_diff > max_pose_differnece_between_old_new_) {
        return false;
    }

    if (pose_diff < 0.01) {
        needs_correction_ = false;
        return false;
    }
    
    needs_correction_ = true;
    
    if (debug_mode_) {
        std::cout << "[LoopClosure] Pose correction delta: rotation=" << pose_distance_deg << "deg, translation norm=" << cv::norm(t_delta) << std::endl;
        std::cout << "[LoopClosure] Pose difference (new vs old): " << pose_diff << std::endl;
    }
    
    loop_rotation_ = R_delta.clone();
    loop_translation_ = t_delta.clone();
    loop_correction_ = T_delta.clone();
    
    candidate_keyframe_id_ = candidate_kf.id;
    candidate_index_ = static_cast<int>(max_sim_index);
    loop_detected_ = true;
    
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
    for (int i = candidate_index_ + 1; i <= current_index; i++) {
        double alpha = static_cast<double>(i - candidate_index_) / static_cast<double>(span);
        
        cv::Mat rvec_partial = rvec_delta * alpha;
        cv::Mat R_partial;
        cv::Rodrigues(rvec_partial, R_partial);
        cv::Mat t_partial = t_delta * alpha;
        
        cv::Mat T_partial = cv::Mat::eye(4, 4, CV_64F);
        R_partial.copyTo(T_partial(cv::Rect(0, 0, 3, 3)));
        t_partial.copyTo(T_partial(cv::Rect(3, 0, 1, 3)));
        
        KeyFrame& kf = keyframes_[i];
        kf.full_pose = T_partial * kf.full_pose;
        kf.rotation = kf.full_pose(cv::Rect(0, 0, 3, 3)).clone();
        kf.translation = kf.full_pose(cv::Rect(3, 0, 1, 3)).clone();
        corrected_count++;
    }
    
    needs_correction_ = false;
    
    if (debug_mode_) {
        std::cout << "[LoopClosure] Interpolated correction applied to " << corrected_count
                   << " keyframes between candidate " << keyframes_[candidate_index_].id
                   << " and current " << keyframes_[current_index].id << std::endl;
    }
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
