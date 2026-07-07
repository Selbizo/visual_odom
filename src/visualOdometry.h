#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#if USE_CUDA
  #include <opencv2/cudaoptflow.hpp>
  #include <opencv2/cudaimgproc.hpp>
  #include <opencv2/cudaarithm.hpp>
  #include <opencv2/cudalegacy.hpp>
#endif

#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include "feature.h"
#include "bucket.h"
#include "utils.h"
#include "Frame.h"



void matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>&  pointsLeft_t0, 
                      std::vector<cv::Point2f>&  pointsRight_t0, 
                      std::vector<cv::Point2f>&  pointsLeft_t1, 
                      std::vector<cv::Point2f>&  pointsRight_t1,
                      double crop);

void matchingFeaturesStab(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>&  pointsLeft_t0, 
                      std::vector<cv::Point2f>&  pointsRight_t0, 
                      std::vector<cv::Point2f>&  pointsLeft_t1, 
                      std::vector<cv::Point2f>&  pointsRight_t1,
                      cv::Ptr<cv::cuda::CornersDetector>& d_features,
                      double crop);

void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::Point2f>&  pointsLeft_t0,
                         std::vector<cv::Point2f>&  pointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation,
                         unsigned int frame_skip,
                         bool mono_rotation=true);

void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<cv::Point2f>&  pointsLeft_t0,
                     std::vector<cv::Point2f>&  pointsLeft_t1,
                     std::string name);

struct PoseGraphNode3D
{
    int frameId = -1;
    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double yaw = 0.0;
};

struct PoseGraphEdge3D
{
    int from = -1;
    int to = -1;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double dyaw = 0.0;
};

bool optimizePoseGraph(std::vector<PoseGraphNode3D>& nodes,
                       const std::vector<PoseGraphEdge3D>& edges,
                       int iterations = 10);

bool addKeyframeAndCheckLoop(const cv::Mat& imageGray,
                             int frameId,
                             const cv::Mat& projMatL,
                             const cv::Mat& projMatR,
                             const cv::Mat& worldPose,
                             std::vector<Frame>& keyframes,
                             cv::Mat& loopTransform,
                             int& matchedFrameId);

#endif
