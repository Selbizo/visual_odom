#include "visualOdometry.h"
using namespace cv;
using namespace std;

cv::Mat euler2rot(cv::Mat& rotationMatrix, const cv::Mat & euler)
{

  double x = euler.at<double>(0);
  double y = euler.at<double>(1);
  double z = euler.at<double>(2);

  // Assuming the angles are in radians.
  double ch = cos(z);
  double sh = sin(z);
  double ca = cos(y);
  double sa = sin(y);
  double cb = cos(x);
  double sb = sin(x);

  double m00, m01, m02, m10, m11, m12, m20, m21, m22;

  m00 = ch * ca;
  m01 = sh*sb - ch*sa*cb;
  m02 = ch*sa*sb + sh*cb;
  m10 = sa;
  m11 = ca*cb;
  m12 = -ca*sb;
  m20 = -sh*ca;
  m21 = sh*sa*cb + ch*sb;
  m22 = -sh*sa*sb + ch*cb;

  rotationMatrix.at<double>(0,0) = m00;
  rotationMatrix.at<double>(0,1) = m01;
  rotationMatrix.at<double>(0,2) = m02;
  rotationMatrix.at<double>(1,0) = m10;
  rotationMatrix.at<double>(1,1) = m11;
  rotationMatrix.at<double>(1,2) = m12;
  rotationMatrix.at<double>(2,0) = m20;
  rotationMatrix.at<double>(2,1) = m21;
  rotationMatrix.at<double>(2,2) = m22;

  return rotationMatrix;
}

void checkValidMatch(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_return, std::vector<bool>& status, int threshold)
{
    int offset;
    for (int i = 0; i < points.size(); i++)
    {
        offset = std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
        // std::cout << offset << ", ";

        if(offset > threshold)
        {
            status.push_back(false);
        }
        else
        {
            status.push_back(true);
        }
    }
}

void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status)
{
    // ИСПРАВЛЕНИЕ: используем reserve + swap вместо O(n^2) erase в цикле
    int n = points.size();
    std::vector<cv::Point2f> valid_points;
    valid_points.reserve(n);
    
    for (int i = 0; i < n; i++)
    {
        if (status[i])
        {
            valid_points.push_back(points[i]);
        }
    }
    
    points.swap(valid_points);
}



void matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>&  pointsLeft_t0, 
                      std::vector<cv::Point2f>&  pointsRight_t0, 
                      std::vector<cv::Point2f>&  pointsLeft_t1, 
                      std::vector<cv::Point2f>&  pointsRight_t1,
                      double crop)
{
    // ----------------------------
    // Feature detection using FAST
    // ----------------------------
    std::vector<cv::Point2f>  pointsLeftReturn_t0;   // feature points to check cicular mathcing validation

    if (currentVOFeatures.size() < 2000)
    {
        // append new features with old features
        appendNewFeatures(imageLeft_t0, currentVOFeatures);   
        // std::cout << "Current feature set size: " << currentVOFeatures.points.size() << std::endl;
    }

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------
    int bucket_size = imageLeft_t0.rows/20;
    int features_per_bucket = 1;
    bucketingFeatures(imageLeft_t0, currentVOFeatures, bucket_size, features_per_bucket, crop);

    pointsLeft_t0 = currentVOFeatures.points;
    
    #if USE_CUDA
    	circularMatching_gpu(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #else
	    circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #endif
    std::vector<bool> status;
    checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 0);

    removeInvalidPoints(pointsLeft_t0, status);
    removeInvalidPoints(pointsLeft_t1, status);
    removeInvalidPoints(pointsRight_t0, status);
    removeInvalidPoints(pointsRight_t1, status);

    currentVOFeatures.points = pointsLeft_t1;
}

void matchingFeaturesStab(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>&  pointsLeft_t0, 
                      std::vector<cv::Point2f>&  pointsRight_t0, 
                      std::vector<cv::Point2f>&  pointsLeft_t1, 
                      std::vector<cv::Point2f>&  pointsRight_t1,
                      Ptr<cuda::CornersDetector>& d_features,
                      double crop)
{
    // ----------------------------
    // Feature detection using FAST
    // ----------------------------
    std::vector<cv::Point2f>  pointsLeftReturn_t0;   // feature points to check cicular mathcing validation

    // ИСПРАВЛЕНИЕ: ограничиваем максимальный размер feature set
    // Если features > 2000, очищаем периферийные и добавляем только если < 1500
    if (currentVOFeatures.size() > 2000) {
        // Слишком много features — очищаем и начинаем заново
        currentVOFeatures.clear();
    }
    
    if (currentVOFeatures.size() < 1500)
    {
        // append new features with old features
        appendNewFeatures(d_features, imageLeft_t0, currentVOFeatures);   
        // std::cout << "Current feature set size: " << currentVOFeatures.size() << std::endl;
    }

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------

    int bucket_size = imageLeft_t0.rows/21;
    int features_per_bucket = 10;
    bucketingFeatures(imageLeft_t0, currentVOFeatures, bucket_size, features_per_bucket, crop);

    pointsLeft_t0 = currentVOFeatures.points;
    
    #if USE_CUDA
    	circularMatching_gpu(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #else
	    circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #endif
    std::vector<bool> status;
    checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 0);

    removeInvalidPoints(pointsLeft_t0, status);
    removeInvalidPoints(pointsLeft_t1, status);
    removeInvalidPoints(pointsRight_t0, status);
    removeInvalidPoints(pointsRight_t1, status);

    currentVOFeatures.points = pointsLeft_t1;
}

void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::Point2f>&  pointsLeft_t0,
                         std::vector<cv::Point2f>&  pointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation,
                         unsigned int frame_skip,
                         bool mono_rotation)
{

      // Calculate frame to frame transformation

      // -----------------------------------------------------------
      // Rotation(R) estimation using Nister's Five Points Algorithm
      // -----------------------------------------------------------
      double focal = projMatrl.at<float>(0, 0);
      cv::Point2d principle_point(projMatrl.at<float>(0, 2), projMatrl.at<float>(1, 2));

      //recovering the pose and the essential cv::matrix
      cv::Mat E, mask;
      cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
      if(mono_rotation)
      {
      	E = cv::findEssentialMat(pointsLeft_t0, pointsLeft_t1, focal, principle_point, cv::RANSAC, 0.999, 1.0, mask);
      	cv::recoverPose(E, pointsLeft_t0, pointsLeft_t1, rotation, translation_mono, focal, principle_point, mask);
      	// std::cout << "recoverPose rotation: " << rotation << std::endl;
      }
      // ------------------------------------------------
      // Translation (t) estimation by use solvePnPRansac
      // ------------------------------------------------
      cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);   
      cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
      cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
                                                   projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2),
                                                   projMatrl.at<float>(2, 0), projMatrl.at<float>(2, 1), projMatrl.at<float>(2, 2));
                                                   // projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 3));

      int iterationsCount = 500;        // number of Ransac iterations.
      float reprojectionError = .5;    // maximum allowed distance to consider it an inlier.
      float confidence = 0.999;          // RANSAC successful confidence.
      bool useExtrinsicGuess = true;
      int flags =cv::SOLVEPNP_ITERATIVE;

      #if 1
      cv::Mat inliers;
      cv::solvePnPRansac( points3D_t0, pointsLeft_t1, intrinsic_matrix, distCoeffs, rvec, translation,
                          useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                          inliers, flags );
      #endif
      #if 0
      std::vector<int> inliers;
      cv::cuda::solvePnPRansac(points3D_t0.t(), cv::Mat(1, (int)pointsLeft_t1.size(), CV_32FC2, &pointsLeft_t1[0]),
                            intrinsic_matrix, cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)),
                            rvec, translation, false, 200, 0.5, 20, &inliers);
      #endif
      if (!mono_rotation)
      {
        //cv::multiply(rvec, cv::Mat::ones(3,1, CV_64FC1)*2, rvec);
        cv::Rodrigues(rvec, rotation);
      }

    //   std::cout << "[trackingFrame2Frame] inliers size: " << inliers.size() << std::endl;

}

namespace {

double normalizeAngle(double angle)
{
    while (angle > CV_PI)
    {
        angle -= 2.0 * CV_PI;
    }
    while (angle < -CV_PI)
    {
        angle += 2.0 * CV_PI;
    }
    return angle;
}

cv::Mat makeRigidTransform(const cv::Mat& rotation, const cv::Mat& translation)
{
    cv::Mat rigidTransform = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat rotation64;
    cv::Mat translation64;
    rotation.convertTo(rotation64, CV_64F);
    translation.convertTo(translation64, CV_64F);

    rotation64.copyTo(rigidTransform(cv::Rect(0, 0, 3, 3)));
    rigidTransform.at<double>(0, 3) = translation64.at<double>(0, 0);
    rigidTransform.at<double>(1, 3) = translation64.at<double>(1, 0);
    rigidTransform.at<double>(2, 3) = translation64.at<double>(2, 0);

    return rigidTransform;
}

cv::Mat makePoseFromState(const PoseGraphNode3D& node)
{
    cv::Mat euler = (cv::Mat_<double>(3, 1) << 0.0, 0.0, node.yaw);
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    euler2rot(rotation, euler);

    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    rotation.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    pose.at<double>(0, 3) = node.x;
    pose.at<double>(1, 3) = node.y;
    pose.at<double>(2, 3) = node.z;
    return pose;
}

} // namespace

bool optimizePoseGraph(std::vector<PoseGraphNode3D>& nodes,
                       const std::vector<PoseGraphEdge3D>& edges,
                       int iterations)
{
    if (nodes.empty() || edges.empty())
    {
        return false;
    }

    for (int iter = 0; iter < iterations; ++iter)
    {
        bool changed = false;
        for (const PoseGraphEdge3D& edge : edges)
        {
            if (edge.from < 0 || edge.to < 0 || edge.from >= static_cast<int>(nodes.size()) || edge.to >= static_cast<int>(nodes.size()))
            {
                continue;
            }

            PoseGraphNode3D& fromNode = nodes[edge.from];
            PoseGraphNode3D& toNode = nodes[edge.to];

            const double dx = (toNode.x - fromNode.x) - edge.dx;
            const double dy = (toNode.y - fromNode.y) - edge.dy;
            const double dz = (toNode.z - fromNode.z) - edge.dz;
            const double dyaw = normalizeAngle((toNode.yaw - fromNode.yaw) - edge.dyaw);

            if (std::abs(dx) > 1e-6 || std::abs(dy) > 1e-6 || std::abs(dz) > 1e-6 || std::abs(dyaw) > 1e-6)
            {
                changed = true;
                toNode.x -= dx * 0.5;
                toNode.y -= dy * 0.5;
                toNode.z -= dz * 0.5;
                toNode.yaw -= dyaw * 0.5;
                toNode.yaw = normalizeAngle(toNode.yaw);
            }
        }

        if (!changed)
        {
            break;
        }
    }

    for (PoseGraphNode3D& node : nodes)
    {
        node.pose = makePoseFromState(node);
    }

    return true;
}

bool addKeyframeAndCheckLoop(const cv::Mat& imageGray,
                             int frameId,
                             const cv::Mat& projMatL,
                             const cv::Mat& projMatR,
                             const cv::Mat& worldPose,
                             std::vector<Frame>& keyframes,
                             cv::Mat& loopTransform,
                             int& matchedFrameId)
{
    loopTransform = cv::Mat::eye(4, 4, CV_64F);
    matchedFrameId = -1;

    if (imageGray.empty())
    {
        return false;
    }

    // Extract descriptors for this frame
    // Note: ORB is only used here for visualization/debugging purposes - actual feature detection uses CUDA CornerDetector
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(15, 1.2f, 8, 15, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20); // Create ORB detector with parameters: nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=HARRIS_SCORE, patchSize=31, fastThreshold=20
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(imageGray, cv::noArray(), keypoints, descriptors);
    // === DEBUG: Visualize ORB keypoints on grayscale frame ===
    {
        static bool initWin = false;
        if (!initWin) {
            cv::namedWindow("ORB Keypoints Debug", cv::WINDOW_AUTOSIZE);
            initWin = true;
        }
        cv::Mat visImg = imageGray.clone();
        if (visImg.channels() == 1) cv::cvtColor(visImg, visImg, cv::COLOR_GRAY2BGR);
        cv::drawKeypoints(visImg, keypoints, visImg, cv::Scalar(100, 255, 100), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("ORB Keypoints Debug", visImg);
        if (frameId >= 137 && frameId <= 185 || frameId >= 1582 && frameId <= 1628) {
            cv::imwrite("/home/selbizo/CV/StabAndSLAM/visual_odom/src/OutputResults/ORB_Keypoints_Frame" + std::to_string(frameId) + ".jpg", visImg);
        }
    }
    // =========================================================


    Frame currentFrame(frameId, projMatL, projMatR, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
    currentFrame.setImage(imageGray);
    currentFrame.setKeypoints(keypoints);
    currentFrame.setDescriptors(descriptors);
    currentFrame.setPose(worldPose);

    // Get current position from pose
    double curX = worldPose.at<double>(0, 3);
    double curY = worldPose.at<double>(1, 3);
    double curZ = worldPose.at<double>(2, 3);

    // Extract ABSOLUTE rotation from worldPose (cumulative from start)
    cv::Mat currentAbsR = worldPose(cv::Rect(0, 0, 3, 3)).clone();

    // ========================================
    // TURN-BASED KEYFRAME SELECTION:
    // Add KF only after cumulative turn > 50° from last KF
    // This ensures KF is added at the END of a turn, not during
    // Also add 3 frames with step = 7 after each turn to ensure better coverage
    // ========================================
    const double TURN_THRESHOLD = 80.0;  // degrees
    const int MIN_FRAME_GAP_FOR_LOOP = 500;       // Min frames between loop candidates
    const double LOOP_SPATIAL_THRESHOLD = 30.0;  // Max odometry distance for loop candidate (meters) - increased from 15m
    const double SPATIAL_PROXIMITY_THRESHOLD = 30.0;  // "Close" distance for spatial weighting
    const double LOOP_MATCH_THRESHOLD = 0.05;  // Very low — almost any match passes
    const int LOOP_MIN_INLIERS = 50;
    const int MAX_KEYFRAMES = 30;
    
    // Track turn detection for post-turn frame addition
    static int lastTurnFrameId = -1;
    static int postTurnCount = 0;
    const int FRAMES_AFTER_TURN = 3;  // Number of frames to add after turn
    const int FRAME_STEP_AFTER_TURN = 7;     // Step size for frames after turn
    
    // Check if we should add a new KF based on turn angle OR spatial proximity
    bool shouldAddKF = false;
    bool forceLoopCheck = false;  // Force loop check even without significant turn
    
    if (keyframes.empty())
    {
        shouldAddKF = true;
    }
    else
    {
        Frame& lastKF = keyframes.back();
        cv::Mat lastKFR = lastKF.m_worldRotation;
        
        if (lastKFR.empty() || lastKFR.rows < 3)
        {
            shouldAddKF = true;
        }
        else
        {
            // Compute cumulative turn from last KF
            cv::Mat R_delta = lastKFR.t() * currentAbsR;
            cv::Vec3f deltaEuler = rotationMatrixToEulerAngles(R_delta);
            double turnAngleDeg = cv::norm(deltaEuler) * 180.0 / CV_PI;
            
            if (turnAngleDeg > TURN_THRESHOLD)
            {
                shouldAddKF = true;
                lastTurnFrameId = frameId;
                postTurnCount = 0;
                std::cout << " addKeyframeAndCheckLoop() [Turn] frame=" << frameId 
                          << " turnAngle=" << turnAngleDeg << "°" << std::endl;
            }
            else if (lastTurnFrameId != -1 && postTurnCount < FRAMES_AFTER_TURN) {
                // We're in the post-turn period, add extra frames at regular intervals
                int framesSinceTurn = frameId - lastTurnFrameId;
                if (framesSinceTurn > 0 && framesSinceTurn % FRAME_STEP_AFTER_TURN == 0) {
                    shouldAddKF = true;
                    postTurnCount++;
                    std::cout << " addKeyframeAndCheckLoop() [Post-Turn] frame=" << frameId 
                              << " (post-turn frame " << postTurnCount << ")" << std::endl;
                }
            }
        }
        
        // === NEW: Check spatial proximity to force loop closure detection ===
        if (!shouldAddKF) {
            // Calculate distance from current position to last KF position
            double kfX = keyframes.back().m_worldTranslation.at<double>(0, 0);
            double kfY = keyframes.back().m_worldTranslation.at<double>(1, 0);
            double kfZ = keyframes.back().m_worldTranslation.at<double>(2, 0);
            double spatialDistToLastKF = std::sqrt(
                std::pow(curX - kfX, 2) + 
                std::pow(curY - kfY, 2) + 
                std::pow(curZ - kfZ, 2));
            
            // Also check distance to all existing KFs for potential loop closure
            double minSpatialDist = spatialDistToLastKF;
            int closestKFIdx = keyframes.size() - 1;
            
            for (size_t i = 0; i < keyframes.size(); ++i) {
                double kfX_i = keyframes[i].m_worldTranslation.at<double>(0, 0);
                double kfY_i = keyframes[i].m_worldTranslation.at<double>(1, 0);
                double kfZ_i = keyframes[i].m_worldTranslation.at<double>(2, 0);
                double dist = std::sqrt(
                    std::pow(curX - kfX_i, 2) + 
                    std::pow(curY - kfY_i, 2) + 
                    std::pow(curZ - kfZ_i, 2));
                
                if (dist < minSpatialDist) {
                    minSpatialDist = dist;
                    closestKFIdx = static_cast<int>(i);
                }
            }
            
            // If we're close to a KF that's far enough in frame count → force loop check
            int frameGap = std::abs(frameId - keyframes[closestKFIdx].m_frameId);
            if (minSpatialDist < 30.0 && frameGap > MIN_FRAME_GAP_FOR_LOOP) {
                shouldAddKF = true;
                forceLoopCheck = true;
                std::cout << " addKeyframeAndCheckLoop() [SPATIAL] frame=" << frameId 
                          << " spatialDist=" << minSpatialDist << "m to KF[" 
                          << keyframes[closestKFIdx].m_frameId << "] gap=" << frameGap 
                          << " → forcing loop check" << std::endl;
            }
        }
    }
    
    // If not enough turn AND not close to any KF, skip this frame entirely
    if (!shouldAddKF)
    {
        return false;
    }
    
    // Trim old KFs
    while (keyframes.size() > MAX_KEYFRAMES)
    {
        keyframes.erase(keyframes.begin());
    }
    
    std::cout << " addKeyframeAndCheckLoop() [UNIQUE] frame=" << frameId << " → adding as KF (KF count=" << keyframes.size() << ")" << std::endl;
    
    // ========================================
    // LOOP CLOSURE CHECK:
    // Before adding as new KF, check if it matches an OLD KF (frame gap > 500)
    // Prioritize checking closest keyframes first for better loop closure detection
    // ========================================
    
    // Collect potential loop closure candidates with their spatial distances
    struct LoopCandidate {
        int index;
        double spatialDist;
    };
    std::vector<LoopCandidate> candidates;
    candidates.reserve(keyframes.size());
    
    for (size_t i = 0; i < keyframes.size(); ++i) {
        const Frame& kf = keyframes[i];
        int frameGap = std::abs(frameId - static_cast<int>(kf.m_frameId));
        
        // Must have enough frame gap to be a valid loop candidate
        if (frameGap < MIN_FRAME_GAP_FOR_LOOP) continue;
        if (kf.m_descriptors.empty() || kf.m_descriptors.rows < 10) continue;
        
        // Calculate spatial distance
        double kfX = kf.m_worldTranslation.at<double>(0, 0);
        double kfY = kf.m_worldTranslation.at<double>(1, 0);
        double kfZ = kf.m_worldTranslation.at<double>(2, 0);
        double spatialDist = std::sqrt(
            std::pow(curX - kfX, 2) +
            std::pow(curY - kfY, 2) +
            std::pow(curZ - kfZ, 2));
        
        // Must be within spatial threshold to be a candidate
        if (spatialDist <= LOOP_SPATIAL_THRESHOLD) {
            candidates.push_back({static_cast<int>(i), spatialDist});
        }
    }
    
    // Sort candidates by spatial distance (closest first) for prioritized checking
    std::sort(candidates.begin(), candidates.end(),
              [](const LoopCandidate& a, const LoopCandidate& b) {
                  return a.spatialDist < b.spatialDist;
              });
    
    // Check candidates in order of proximity (closest first)
    for (const auto& candidate : candidates) {
        const Frame& kf = keyframes[candidate.index];
        int frameGap = std::abs(frameId - static_cast<int>(kf.m_frameId));
        double spatialDist = candidate.spatialDist;
        
        // SPATIAL WEIGHTING: boost score for close matches
        double spatialWeight = 1.0;
        if (spatialDist < SPATIAL_PROXIMITY_THRESHOLD) {
            spatialWeight = 2.0;  // Close match → higher priority
        }
        
        std::cout << " addKeyframeAndCheckLoop() [LoopCheck-PRIORITIZE] vs KF[" 
                  << kf.m_frameId << "] gap=" << frameGap 
                  << " spatialDist=" << spatialDist << " weight=" << spatialWeight << std::endl;
        
        // Descriptor matching
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors, kf.m_descriptors, matches);
        
        // Count good matches
        int goodMatches = 0;
        for (const auto& m : matches) {
            if (m.distance < 80) {  // Increased threshold from 50 to 80 to catch weaker matches
                goodMatches++;
            }
        }
        
        double matchRatio = static_cast<double>(goodMatches) / std::min(descriptors.rows, kf.m_descriptors.rows);
        
        std::cout << " addKeyframeAndCheckLoop() [LoopCheck] vs KF[" << kf.m_frameId 
                  << "] gap=" << frameGap << " goodMatches=" << goodMatches 
                  << " ratio=" << matchRatio << " weight=" << spatialWeight << std::endl;
        
        // Use weighted ratio for loop detection
        double weightedRatio = matchRatio * spatialWeight;
        
        std::cout << " addKeyframeAndCheckLoop() [LoopCheck] weightedRatio=" << weightedRatio 
                  << " threshold=" << LOOP_MATCH_THRESHOLD << std::endl;
        
        // Visualize best matches if good matches > 30%
        if (goodMatches > 30 && frameId >= 1582 && frameId <= 1628) {
            cv::BFMatcher matcher(cv::NORM_HAMMING, false);
            std::vector<cv::DMatch> all_matches;
            matcher.match(descriptors, kf.m_descriptors, all_matches);
            
            // Sort matches by distance (best first)
            std::sort(all_matches.begin(), all_matches.end(), 
                      [](const cv::DMatch& a, const cv::DMatch& b) {
                          return a.distance < b.distance;
                      });
            
            // Take top 30 matches for visualization
            int numVisMatches = std::min(30, static_cast<int>(all_matches.size()));
            std::vector<cv::DMatch> visMatches(all_matches.begin(), all_matches.begin() + numVisMatches);
            
            // Create visualization image
            cv::Mat visImg;
            cv::drawMatches(imageGray, keypoints, 
                           kf.m_image, kf.m_keypoints,
                           visMatches, visImg, 
                           cv::Scalar::all(-1), cv::Scalar::all(-1),
                           std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            // Save visualization
            std::string filename = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/OutputResults/BestMatches_Frame" + 
                               std::to_string(frameId) + "_vs_KF" + std::to_string(kf.m_frameId) + ".jpg";
            cv::imwrite(filename, visImg);
            
            std::cout << "  *** Best matches visualization saved for frame=" << frameId 
                    << " vs KF[" << kf.m_frameId << "] with " << numVisMatches << " matches" << std::endl;
        }
        
        if (weightedRatio >= LOOP_MATCH_THRESHOLD) {
            // Potential loop closure — verify with geometric check
            // Apply RANSAC filtering to handle point permutation issues
            std::vector<cv::Point2f> prevPoints, currPoints;
            
            // For better matching, we'll use RANSAC-based approach
            if (matches.size() >= 30) {
                // Filter matches using RANSAC for better geometric verification
                std::vector<cv::DMatch> filteredMatches;
                for (const auto& m : matches) {
                    if (m.distance < 80) {  // Use the same threshold as above
                        filteredMatches.push_back(m);
                    }
                }
                
                // If we have enough good matches, apply RANSAC filtering
                if (filteredMatches.size() >= 30) {
                    // Create point vectors for geometric verification
                    for (const auto& m : filteredMatches) {
                        prevPoints.push_back(kf.m_keypoints[m.queryIdx].pt);
                        currPoints.push_back(keypoints[m.trainIdx].pt);
                    }
                } else {
                    // Fallback to original approach if not enough matches
                    for (const auto& m : matches) {
                        if (m.distance < 80) {
                            prevPoints.push_back(kf.m_keypoints[m.queryIdx].pt);
                            currPoints.push_back(keypoints[m.trainIdx].pt);
                        }
                    }
                }
            } else {
                // Fallback to original approach for small number of matches
                for (const auto& m : matches) {
                    if (m.distance < 80) {
                        prevPoints.push_back(kf.m_keypoints[m.queryIdx].pt);
                        currPoints.push_back(keypoints[m.trainIdx].pt);
                    }
                }
            }
            
            std::cout << " addKeyframeAndCheckLoop() [Geometric] prevPoints=" << prevPoints.size() 
                      << " currPoints=" << currPoints.size() << std::endl;
            
            if (prevPoints.size() >= 30) {
                const double focal = projMatL.at<float>(0, 0);
                const cv::Point2d principalPoint(projMatL.at<float>(0, 2), projMatL.at<float>(1, 2));
                
                cv::Mat mask;
                cv::Mat E = cv::findEssentialMat(currPoints, prevPoints, focal, principalPoint, cv::RANSAC, 0.999, 1.0, mask);
                cv::Mat R, t;
                cv::recoverPose(E, currPoints, prevPoints, R, t, focal, principalPoint, mask);
                
                int inliers = cv::countNonZero(mask);
                std::cout << "    [Geometric] inliers=" << inliers 
                          << " threshold=" << LOOP_MIN_INLIERS << std::endl;
                
                if (inliers >= LOOP_MIN_INLIERS) {
                    std::cout << "  *** LOOP CLOSURE DETECTED: frame=" << frameId 
                              << " <-> KF[" << kf.m_frameId << "] inliers=" << inliers << std::endl;
                    loopTransform = makeRigidTransform(R, t);
                    matchedFrameId = kf.m_frameId;
                    keyframes.push_back(currentFrame);
                    return true;
                }
            }
        }
    }
    
    // No loop closure — add as regular keyframe
    keyframes.push_back(currentFrame);
    return false;
}

void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<cv::Point2f>&  pointsLeft_t0,
                     std::vector<cv::Point2f>&  pointsLeft_t1,
                     std::string name)
{
      // -----------------------------------------
      // Display feature racking
      // -----------------------------------------
      int radius = 2;
      cv::Mat vis;
      cv::cvtColor(imageLeft_t1, vis, cv::COLOR_GRAY2BGR, 3);


      for (int i = 0; i < pointsLeft_t0.size(); i++)
      {
          cv::circle(vis, cv::Point(pointsLeft_t0[i].x, pointsLeft_t0[i].y), radius, CV_RGB(0, 0, 100));
      }

      for (int i = 0; i < pointsLeft_t1.size(); i++)
      {
          cv::circle(vis, cv::Point(pointsLeft_t1[i].x, pointsLeft_t1[i].y), radius+1, CV_RGB(200, 40, 40), -1);
      }

      for (int i = 0; i < pointsLeft_t1.size(); i++)
      {
          cv::line(vis, pointsLeft_t0[i], pointsLeft_t1[i], CV_RGB(10, 100, 10));
      }
    //   cv::waitKey(200);
      cv::imshow(name, vis );  
}

double calculatePoseGraphResidual(const std::vector<PoseGraphNode3D>& nodes,
                                       const std::vector<PoseGraphEdge3D>& edges)
{
    double residual = 0.0;
    for (const auto& edge : edges) {
        if (edge.from >= 0 && edge.to >= 0 && 
            edge.from < static_cast<int>(nodes.size()) && 
            edge.to < static_cast<int>(nodes.size())) {
            residual += std::abs(edge.dx) + std::abs(edge.dy) + std::abs(edge.dz) + std::abs(edge.dyaw);
        }
    }
    return residual;
}
