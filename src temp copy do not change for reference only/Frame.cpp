#include "Frame.h"

Frame::Frame()
    : m_frameId(-1)
{}


Frame::Frame(int frameId, const cv::Mat projMatL, const cv::Mat projMatR, cv::Mat worldRotation, cv::Mat worldTranslation)
    : m_frameId(frameId)
{
    m_projMatL = projMatL;
    m_projMatR = projMatR;
    m_worldRotation = worldRotation;
    m_worldTranslation = worldTranslation;
}



void Frame::setFeatures(std::vector<cv::Point2f> pointsFeatureLeft, std::vector<cv::Point2f> pointsFeatureRight)
{
    m_pointsFeatureLeft = pointsFeatureLeft;
    m_pointsFeatureRight = pointsFeatureRight;

}

void Frame::setPose(const cv::Mat& worldPose)
{
    if (worldPose.rows >= 4 && worldPose.cols >= 4)
    {
        cv::Mat rotation = worldPose(cv::Rect(0, 0, 3, 3));
        rotation.copyTo(m_worldRotation);

        m_worldTranslation = cv::Mat::zeros(3, 1, CV_64F);
        m_worldTranslation.at<double>(0, 0) = worldPose.at<double>(0, 3);
        m_worldTranslation.at<double>(1, 0) = worldPose.at<double>(1, 3);
        m_worldTranslation.at<double>(2, 0) = worldPose.at<double>(2, 3);
    }
}

void Frame::setPose(const cv::Mat& worldRotation, const cv::Mat& worldTranslation)
{
    m_worldRotation = worldRotation;
    m_worldTranslation = worldTranslation;
}

void Frame::setImage(const cv::Mat& image)
{
    image.copyTo(m_image);
}

void Frame::setKeypoints(const std::vector<cv::KeyPoint>& keypoints)
{
    m_keypoints = keypoints;
}

void Frame::setDescriptors(const cv::Mat& descriptors)
{
    descriptors.copyTo(m_descriptors);
}

void Frame::triangulateFeaturePoints(cv::Mat& points4D)
{
    cv::triangulatePoints( m_projMatL,  m_projMatR,  m_pointsFeatureLeft,  m_pointsFeatureRight,  points4D);
}


