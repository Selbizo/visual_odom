#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

class Frame
{
public:
    // Constructors

    Frame();

    Frame(int frameId, const cv::Mat projMatL, const cv::Mat projMatR, cv::Mat worldRotation, cv::Mat worldTranslation);


    void setFeatures(std::vector<cv::Point2f> pointsFeatureLeft, std::vector<cv::Point2f> pointsFeatureRight);
    void setPose(const cv::Mat& worldPose);
    void setPose(const cv::Mat& worldRotation, const cv::Mat& worldTranslation);
    void setImage(const cv::Mat& image);
    void setKeypoints(const std::vector<cv::KeyPoint>& keypoints);
    void setDescriptors(const cv::Mat& descriptors);

    void triangulateFeaturePoints(cv::Mat& points4D);


    int m_frameId;
    cv::Mat m_projMatL, m_projMatR;

    cv::Mat m_worldRotation, m_worldTranslation;
    cv::Mat m_image;
    std::vector<cv::KeyPoint> m_keypoints;
    cv::Mat m_descriptors;

    std::vector<cv::Point2f> m_pointsFeatureLeft, m_pointsFeatureRight;
    

// private:

};


#endif
