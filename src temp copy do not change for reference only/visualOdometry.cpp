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

    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(imageGray, cv::noArray(), keypoints, descriptors);

    if (keypoints.empty() || descriptors.empty())
    {
        return false;
    }

    Frame currentFrame(frameId, projMatL, projMatR, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
    currentFrame.setImage(imageGray);
    currentFrame.setKeypoints(keypoints);
    currentFrame.setDescriptors(descriptors);
    currentFrame.setPose(worldPose);

    // Get current position from pose
    double curX = worldPose.at<double>(0, 3);
    double curY = worldPose.at<double>(1, 3);
    double curZ = worldPose.at<double>(2, 3);

    // Check if we should add this frame as a keyframe
    // Only add if sufficiently far from the last keyframe
    bool shouldAddAsKeyframe = false;
    if (keyframes.empty())
    {
        shouldAddAsKeyframe = true;
    }
    else
    {
        Frame& lastKeyframe = keyframes.back();
        double lastX = lastKeyframe.m_worldTranslation.at<double>(0, 0);
        double lastY = lastKeyframe.m_worldTranslation.at<double>(1, 0);
        double lastZ = lastKeyframe.m_worldTranslation.at<double>(2, 0);
        double dist = std::sqrt(
            std::pow(curX - lastX, 2) +
            std::pow(curY - lastY, 2) +
            std::pow(curZ - lastZ, 2));

        // Add as keyframe only if moved at least 3 meters
        if (dist > 3.0)
        {
            shouldAddAsKeyframe = true;
        }
    }

    if (!shouldAddAsKeyframe)
    {
        return false;
    }

    // Now check for loop closure against existing keyframes
    const double focal = projMatL.at<float>(0, 0);
    const cv::Point2d principalPoint(projMatL.at<float>(0, 2), projMatL.at<float>(1, 2));

    // Minimum frame gap: require >500 frames between current and candidate
    const int MIN_FRAME_GAP = 500;
    // Maximum spatial distance: skip candidates too far away (>100m) — different location
    const double MAX_SPATIAL_DISTANCE = 100.0;
    // Minimum inlier count for loop detection
    const int MIN_INLIERS = 35;
    // Minimum match ratio for loop detection
    const double MIN_MATCH_RATIO = 0.25;

    double bestScore = 0.0;
    int bestMatchedFrameId = -1;
    cv::Mat bestLoopTransform = cv::Mat::eye(4, 4, CV_64F);

    for (const Frame& previousFrame : keyframes)
    {
        if (previousFrame.m_descriptors.empty() || previousFrame.m_keypoints.empty())
        {
            continue;
        }

        // Check minimum frame gap
        if (std::abs(frameId - previousFrame.m_frameId) < MIN_FRAME_GAP)
        {
            continue;
        }

        // Check maximum spatial distance: skip candidates too far away (>100m)
        // This avoids comparing frames from completely different locations
        double prevX = previousFrame.m_worldTranslation.at<double>(0, 0);
        double prevY = previousFrame.m_worldTranslation.at<double>(1, 0);
        double prevZ = previousFrame.m_worldTranslation.at<double>(2, 0);
        double spatialDist = std::sqrt(
            std::pow(curX - prevX, 2) +
            std::pow(curY - prevY, 2) +
            std::pow(curZ - prevZ, 2));

        if (spatialDist > MAX_SPATIAL_DISTANCE)
        {
            continue;
        }

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(previousFrame.m_descriptors, descriptors, matches);

        // Check match ratio
        double matchRatio = static_cast<double>(matches.size()) / std::min(previousFrame.m_descriptors.rows, descriptors.rows);
        if (matchRatio < MIN_MATCH_RATIO)
        {
            continue;
        }

        std::vector<cv::Point2f> prevPoints;
        std::vector<cv::Point2f> currPoints;
        prevPoints.reserve(matches.size());
        currPoints.reserve(matches.size());

        for (const cv::DMatch& match : matches)
        {
            prevPoints.push_back(previousFrame.m_keypoints[match.queryIdx].pt);
            currPoints.push_back(keypoints[match.trainIdx].pt);
        }

        std::vector<uchar> mask;
        cv::Mat essentialMatrix = cv::findEssentialMat(prevPoints, currPoints, focal, principalPoint, cv::RANSAC, 0.999, 1.0, mask);
        cv::Mat rotation;
        cv::Mat translation;
        cv::recoverPose(essentialMatrix, prevPoints, currPoints, rotation, translation, focal, principalPoint, mask);

        if (cv::norm(translation) < 0.01)
        {
            continue;
        }

        const int inlierCount = cv::countNonZero(mask);
        if (inlierCount < MIN_INLIERS)
        {
            continue;
        }

        // Score based on inlier count and match ratio
        double score = static_cast<double>(inlierCount) * matchRatio;
        if (score > bestScore)
        {
            bestScore = score;
            bestMatchedFrameId = previousFrame.m_frameId;
            bestLoopTransform = makeRigidTransform(rotation, translation);
        }
    }

    // Only accept if we found a good match
    if (bestMatchedFrameId >= 0 && bestScore > 50.0)
    {
        loopTransform = bestLoopTransform;
        matchedFrameId = bestMatchedFrameId;
        keyframes.push_back(currentFrame);
        return true;
    }

    // Add as regular keyframe
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
          cv::circle(vis, cv::Point(pointsLeft_t0[i].x, pointsLeft_t0[i].y), radius, CV_RGB(100, 0, 0));
      }

      for (int i = 0; i < pointsLeft_t1.size(); i++)
      {
          cv::circle(vis, cv::Point(pointsLeft_t1[i].x, pointsLeft_t1[i].y), radius+1, CV_RGB(200, 0, 0), -1);
      }

      for (int i = 0; i < pointsLeft_t1.size(); i++)
      {
          cv::line(vis, pointsLeft_t0[i], pointsLeft_t1[i], CV_RGB(10, 100, 10));
      }
    //   cv::waitKey(200);
      cv::imshow(name, vis );  
}
