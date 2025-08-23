// Функции отвечающие за стабилизацию
#pragma once


// подключение необходимых модулей OpenCV
#include <opencv2/core.hpp>          
#include <opencv2/imgproc.hpp>       
#include <opencv2/videoio.hpp>       
#include <opencv2/core/cuda.hpp>     
#include <opencv2/cudaarithm.hpp>    
#include <opencv2/cudaimgproc.hpp>   
#include <opencv2/calib3d.hpp>
#include <opencv2/cudawarping.hpp>

#include <vector>    // std::vector
#include <iostream>  // std::cout
#include "basicFunctions.hpp"

using namespace cv;
using namespace std;

void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldCompressed, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gOldCompressed, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d,
	vector <TransformParam>& transforms, Mat& T, const int compression);

void iir(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, Mat& frame);




void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldCompressed, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gOldCompressed, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, Point2f& meanP0,
	vector <TransformParam>& transforms, Mat& T, const int compression);

void addFramePoints(cuda::GpuMat& gOldGray, vector<Point2f>& p0,
	Ptr<cuda::CornersDetector>& d_features, cuda::GpuMat& gMaskSearchSmall);

void addFramePoints(cuda::GpuMat& gOldGray, vector<Point2f>& p0,
	Ptr<cuda::CornersDetector>& d_features);

void removeFramePoints(vector<Point2f>& p0, double minDistance);

void iirAdaptiveOld(vector<TransformParam>& transforms, double& tau_stab, 
	Rect& roi, const int a, const int b, const double c, double& kSwitch);

void iirAdaptiveHighPass(vector<TransformParam>& transforms, double& tau_stab, 
	Rect& roi, const int a, const int b, const double c, double& kSwitch, 
	vector<TransformParam>& movement, vector<TransformParam>& movementKalman);

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, 
	Rect& roi, const int a, const int b, const double c, double& kSwitch, 
	vector<TransformParam>& movement, vector<TransformParam>& movementKalman);