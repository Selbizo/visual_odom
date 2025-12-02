#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include "feature.h"
#include "utils.h"
#include "evaluate_odometry.h"
#include "visualOdometry.h"
#include "Frame.h"

#include "camera_object.h"
#include "rgbd_standalone.h"

#include "basicFunctions.h"
#include "stabilizationFunctions.h"


using namespace std;
using namespace cv;

// int main(int argc, char **argv)
int main()
{
    #if USE_CUDA
        printf("CUDA is Enabled\n");
    #endif

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Для отображения надписей на кадре~~~~~~~~~~~~~~~~~~~~~~~~~~~
	int fontFace = FONT_HERSHEY_SIMPLEX;

	//double fontScale = 1.0*min(a,b)/1080;
	double fontScale = 0.7;

	setlocale(LC_ALL, "RU");

	vector <Point> textOrg(20);
    vector <Point> textOrgCrop(20);
	vector <Point> textOrgStab(20);
	vector <Point> textOrgOrig(20);

    for (int i = 0; i < 20; i++)
    {
        textOrg[i].x = 5;
        textOrg[i].y = 5 + 30 * fontScale * (i + 1);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    bool display_ground_truth = false;
    bool use_intel_rgbd = false;
    bool use_camera = false;
    std::vector<Matrix> pose_matrix_gt;
    
    // if(argc == 4)
    // {   display_ground_truth = true;
    //     cerr << "Display ground truth trajectory" << endl;
    //     // load ground truth pose
    //     //string filename_pose = string(argv[3]); ///home/selbizo/CV/dataset/sequences/00/ ../calibration/kitti00.yaml
    //     string filename_pose = string("/home/selbizo/CV/dataset/sequences/00/");
    //     pose_matrix_gt = loadPoses(filename_pose);
    // }

    //string filename_pose = string("/home/selbizo/CV/dataset/sequences/00/");
    //pose_matrix_gt = loadPoses(filename_pose);
    // if(argc < 3)
    // {
    //     cerr << "Usage: ./run path_to_sequence(rgbd for using intel rgbd) path_to_calibration [optional]path_to_ground_truth_pose" << endl;
    //     return 1;
    // }

    // Sequence
    //string filepath = string(argv[1]);
    string filepath = string("/home/selbizo/CV/dataset/sequences/00/");
    cout << "Filepath: " << filepath << endl;

    if(filepath == "rgbd") use_intel_rgbd = true;
    if(filepath == "camera") use_camera = true;

    // Camera calibration
    //string strSettingPath = string(argv[2]);
    string strSettingPath = string("../calibration/kitti00.yaml");
    cout << "Calibration Filepath: " << strSettingPath << endl;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    int frame_skip = 1;
    
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float bf = fSettings["Camera.bf"];


    double MaxShake = 10.0;
    double framePart = 0.94;
    fx = fx/framePart;
    fy = fy/framePart;
    cx = cx/framePart;
    cy = cy/framePart;

    bf = bf/framePart;
    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
    cout << "P_left: " << endl << projMatrl << endl;
    cout << "P_right: " << endl << projMatrr << endl;

    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);
    
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

    std::cout << "frame_pose " << frame_pose << std::endl;
    cv::Mat trajectory = cv::Mat::zeros(1500, 1500, CV_8UC3);
    cv::Mat trajectory_biased = cv::Mat::zeros(500, 500, CV_8UC3);
    FeatureSet currentVOFeatures;
    FeatureSet currentVOFeatures_stab;
    cv::Mat points4D, points3D;
    int init_frame_id = 0;

    //--------------------------------
    // Initialize variables VideoShake
    //--------------------------------
    Mat Shake(2, 3, CV_64F);
    TransformParam noiseIn = { 0.0, 0.0, 0.0 };
    vector <TransformParam> noiseOut(2);

	for (int i = 0; i < noiseOut.size();i++)
	{
		noiseOut[i] = {0.0, 0.0, 0.0};
	}
    vector <TransformParam> X(1+NCoef), Y(1 + NCoef);
    
    //--------------------------------
    // END Initialize variables VideoShake
    //--------------------------------


    //--------------------------------
    // Initialize variables VideoStab
    //--------------------------------
    
    //initialisaton
	std::vector <std::string> folderPath(4);

	// Создадим массив случайных цветов для цветов характерных точек
	std::vector<cv::Scalar> colors;
	cv::RNG rng;
    createPointColors(colors, rng);
	// детектор для поиска характерных точек
	Ptr<cuda::CornersDetector> d_features;
	Ptr<cuda::CornersDetector> d_features_small;
	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse;
	createDetectors(d_features, d_features_small, d_pyrLK_sparse);
    
	//create current arguments and arrays
	Mat oldFrameLeft, oldGrayLeft, errLeft;
	Mat oldFrameRight, oldGrayRight, errRight;
	
	vector<Point2f> p0Left, p1Left, good_newLeft;
	vector<Point2f> p0Right, p1Right, good_newRight;
	cuda::GpuMat gP0Left, gP1Left;
	cuda::GpuMat gP0Right, gP1Right;
    
	Point2f dLeft = Point2f(0.0f, 0.0f);
	Point2f dRight = Point2f(0.0f, 0.0f);
	Point2f meanP0Left = Point2f(0.0f, 0.0f);
	Point2f meanP0Right = Point2f(0.0f, 0.0f);
    
	Mat TLeft, TStabLeft(2, 3, CV_64F), TStabInvLeft(2, 3, CV_64F), TSearchPointsLeft(2, 3, CV_64F);
	Mat TRight, TStabRight(2, 3, CV_64F), TStabInvRight(2, 3, CV_64F), TSearchPointsRight(2, 3, CV_64F);
	cuda::GpuMat gTLeft, gTStabLeft(2, 3, CV_64F);
	cuda::GpuMat gTRight, gTStabRight(2, 3, CV_64F);
    
	vector<uchar> statusLeft, statusRight;

	cuda::GpuMat gStatusLeft, gErrLeft, gStatusRight, gErrRight;
	
	double tauStab = 20.0;
	double gain = 0.7;
	//double framePart = 0.95;

	const unsigned int firSize = 4;
    vector <TransformParam> transforms(firSize), movement(firSize), movementKalman(firSize);

	for (int i = 0; i < firSize;i++)
	{
        transforms[i] = {0.0, 0.0, 0.0};
        movement[i] = {0.0, 0.0, 0.0};
        movementKalman[i] = {0.0, 0.0, 0.0};        
    }
     

	//init KF

	// System dimensions
	const int state_dim = 9;  // vx, vy, ax, ay
	const int meas_dim = 3;   // vx, vy

	// Create system matrices
	double FPS = 30.0;
	double dt = 1; //1/ FPS;
	double dt2 = dt*dt/2;
	cv::Mat A = (cv::Mat_<double>(state_dim, state_dim) <<
		1,	0,	dt,	0,	dt2,0,	0,	0,	0,	//vx	
		0,	1,	0,	dt,	0,	dt2,0,	0,	0,	//vy
		0,	0,	1,	0,	dt,	0,	0,	0,	0,	//ax
		0,	0,	0,	1,	0,	dt,	0,	0,	0,	//ay
		0,	0,	0,	0,	1,	0,	0,	0,	0,	//a2x
		0,	0,	0,	0,	0,	1,	0,	0,	0,	//a2y
		0,	0,	0,	0,	0,	0,	1,	dt,	dt2,//vroll
		0,	0,	0,	0,	0,	0,	0,	1,	dt,	//aroll
		0,	0,	0,	0,	0,	0,	0,	0,	1	//a2roll 
		);

	cv::Mat C = (cv::Mat_<double>(meas_dim, state_dim) <<
		1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0
		);
	
	cv::Mat Q = cv::Mat::eye(state_dim, state_dim, CV_64F) * 0.00001;	//low value
	cv::Mat R = cv::Mat::eye(meas_dim, meas_dim, CV_64F) * 10000.0;		//high value
	cv::Mat P = cv::Mat::eye(state_dim, state_dim, CV_64F) * 1.0;
	
	// Create KF
	KalmanFilterCV kf(dt, A, C, Q, R, P);

	// Initialize with first measurement
	cv::Mat x0 = (cv::Mat_<double>(state_dim, 1) << 0,0,0, 0,0,0, 0,0,0);
	kf.init(0, x0);

	// переменные для фильтра Виннера
	Mat Hw, h, gray_wiener;
	cuda::GpuMat gHw, gH, gGrayWiener;

	bool wiener = false;
	bool threadwiener = false;
	double nsr = 0.01;
	double qWiener = 8.0;
	double LEN = 0;
	double THETA = 0.0;

	//для обработки трех каналов по Виннеру
	vector<Mat> channels(3), channelsWiener(3);
	Mat frame_wiener;

	vector<cuda::GpuMat> gChannels(3), gChannelsWiener(3);
	cuda::GpuMat gFrameWiener;

	// ~~~~~~~~~~~~~~ для счетчика кадров в секунду ~~~~~~~~~~~~~~~//
	unsigned int frameCnt = 0;
	double seconds = 0.05;
	double secondsGPUPing = 0.0;
	double secondsFullPing = 0.0;
	clock_t start = clock();
	clock_t end = clock();

	clock_t startFullPing = clock();
	clock_t endFullPing = clock();

	clock_t startGPUPing = clock();
	clock_t endGPUPing = clock();
    //------------------------------------
    // END Initialize variables VideoStab
    //------------------------------------
    

    // ------------------------
    // Load first images
    // ------------------------
    cv::Mat imageRight_t0,  imageLeft_t0, imageLeft_stab_t0, imageRight_stab_t0;
    CameraBase *pCamera = NULL;
    cv::VideoCapture captureLeft, captureRight;
    // cv::VideoCapture captureLeft("http://192.168.8.106:4747/video?640x480");
    // cv::VideoCapture captureRight("http://192.168.8.107:4747/video?640x480");
    
    cv::Mat imageLeft_t0_color, imageRight_t0_color;
    
    if(use_intel_rgbd)
    {   
        pCamera = new Intel_V4L2;
        for (int throw_frames = 10 ; throw_frames >=0 ; throw_frames--)
        pCamera->getLRFrames(imageLeft_t0,imageRight_t0);
    }
    else if (use_camera &&! use_intel_rgbd)
    {
        captureLeft >> imageLeft_t0_color;
        cvtColor(imageLeft_t0_color, imageLeft_t0, cv::COLOR_BGR2GRAY);
        
        //imageLeft_t0_color.copyTo(imageRight_t0_color);
        // imageLeft_t0.copyTo(imageRight_t0);
        captureRight >> imageRight_t0_color;
        cvtColor(imageRight_t0_color, imageRight_t0, cv::COLOR_BGR2GRAY);
    }
    else
    {
        // cv::Mat imageLeft_t0_color;
        loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath);
        
        // cv::Mat imageRight_t0_color;  
        loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);
    }
    imageLeft_t0.copyTo(imageLeft_stab_t0);
    imageRight_t0.copyTo(imageRight_stab_t0);
    
    clock_t t_a, t_b;

    //init sizes of frames

	const int a = imageLeft_t0.cols;
	const int b = imageLeft_t0.rows;
	const double c = sqrt(a * a + b * b);
	const double atan_ba = atan2(b, a);

    //переменные для запоминания кадров и характерных точек
	Mat frameShowOrigLeft(a, b, CV_8UC3),
        frameShowOrigRight(a, b, CV_8UC3), 
        frameOutLeft(a, b, CV_8UC3),
        frameOutRight(a, b, CV_8UC3);
	cuda::GpuMat gFrameStabilizedLeft(a, b, CV_8UC3),
                 gFrameStabilizedRight(a, b, CV_8UC3);

	cuda::GpuMat gFrameLeft(a,b, CV_8UC3),
                 gFrameRight(a,b, CV_8UC3), 
                 gFrameShowOrigLeft(a, b, CV_8UC3),
                 gFrameShowOrigRight(a, b, CV_8UC3),
		gGrayLeft(a/compression, b / compression, CV_8UC1),
        gGrayRight(a/compression, b / compression, CV_8UC1), 
		gCompressedLeft(a / compression, b / compression, CV_8UC3),
        gCompressedRight(a / compression, b / compression, CV_8UC3);

	cuda::GpuMat gOldFrameLeft(a, b, CV_8UC3),
                 gOldFrameRight(a, b, CV_8UC3), 
		gOldGrayLeft(a / compression, b / compression, CV_8UC1),
        gOldGrayRight(a / compression, b / compression, CV_8UC1),
		gOldCompressedLeft(a / compression, b / compression, CV_8UC3),
        gOldCompressedRight(a / compression, b / compression, CV_8UC3);
	
    cuda::GpuMat gToShowLeft(a, b, CV_8UC3),
                 gToShowRight(a, b, CV_8UC3);

	cuda::GpuMat gRoiGrayLeft, gRoiGrayRight;

	Rect roi(
		a * ((1.0 - framePart) / 2.0),
		b * ((1.0 - framePart) / 2.0),
		a * framePart,
		b * framePart
	);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~для вывода изображения на дисплей~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Mat frameStabilizatedCropResizedLeft(a, b, CV_8UC3), frame_cropLeft,
	    frameStabilizatedCropResizedRight(a, b, CV_8UC3), frame_cropRight;
    cuda::GpuMat 
		gFrameStabilizatedCropLeft(roi.width, roi.height, CV_8UC3),
        gFrameStabilizatedCropRight(roi.width, roi.height, CV_8UC3), 
		gFrameRoiLeft(roi.width, roi.height, CV_8UC3),
        gFrameRoiRight(roi.width, roi.height, CV_8UC3),
		gFrameOutLeft(a, b, CV_8UC3),
        gFrameOutRight(a, b, CV_8UC3),
		gFrameStabilizatedCropResizedLeft(a, b, CV_8UC3),
        gFrameStabilizatedCropResizedRight(a, b, CV_8UC3),
		gWriterFrameToShowLeft(a, b, CV_8UC3),
        gWriterFrameToShowRight(a, b, CV_8UC3);


	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создадим маску для нахождения точек~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Mat maskSearchLeft = Mat::zeros(cv::Size(b / compression , a / compression ), CV_8U);
	Mat maskSearchRight = Mat::zeros(cv::Size(b / compression , a / compression ), CV_8U);
	
    cv::rectangle(maskSearchLeft, Rect(b * (1.0 - 0.5) / compression / 2, b * (1.0 - 0.5) / compression / 2, a * 0.5, b * 0.5 / compression ), 
		Scalar(255), FILLED); // Прямоугольная маска
    cv::rectangle(maskSearchRight, Rect(a * (1.0 - 0.5) / compression / 2, b * (1.0 - 0.5) / compression / 2, a * 0.5, b * 0.5 / compression ), 
		Scalar(255), FILLED); // Прямоугольная маска
    
	// cv::rectangle(maskSearchLeft, Rect(b * (1.0 - 0.4) / compression / 2, b * (1.0 - 0.4) / compression / 2, a * 0.4, b * 0.4 / compression),
	// 	Scalar(0), FILLED);
	// cv::rectangle(maskSearchRight, Rect(a * (1.0 - 0.4) / compression / 2, b * (1.0 - 0.4) / compression / 2, a * 0.4, b * 0.4 / compression),
	// 	Scalar(0), FILLED);
    
	cuda::GpuMat gMaskSearchLeft(maskSearchLeft);
    cuda::GpuMat gMaskSearchRight(maskSearchRight);

	Mat maskSearchSmallLeft = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	Mat maskSearchSmallRight = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	
    cv::rectangle(maskSearchSmallLeft, Rect(a * (1.0 - 0.3) / compression / 2, b * (1.0 - 0.3) / compression / 2, max(a,b) * 0.3 / compression, max(a,b) * 0.3 / compression),
		Scalar(255), FILLED); // Прямоугольная маска

    cv::rectangle(maskSearchSmallRight, Rect(a * (1.0 - 0.3) / compression / 2, b * (1.0 - 0.3) / compression / 2, max(a,b) * 0.3 / compression, max(a,b) * 0.3 / compression),
		Scalar(255), FILLED); // Прямоугольная маска
    
	cuda::GpuMat gMaskSearchSmallLeft(maskSearchSmallLeft);
	cuda::GpuMat gMaskSearchSmallRight(maskSearchSmallRight);
    cuda::GpuMat gMaskSearchSmallRoiLeft, gMaskSearchSmallRoiRight;

	Mat roiMaskLeft = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	cv::rectangle(roiMaskLeft, Rect(a * (1.0 - 0.4) / compression / 2, b * (1.0 - 0.4) / compression / 2, a * 0.4, b * 0.4 / compression),
		Scalar(255), FILLED); // Прямоугольная маска

	Mat roiMaskRight = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	cv::rectangle(roiMaskRight, Rect(a * (1.0 - 0.4) / compression / 2, b * (1.0 - 0.4) / compression / 2, a * 0.4, b * 0.4 / compression),
		Scalar(255), FILLED); // Прямоугольная маска
	//cuda::GpuMat gRoiMask(roiMask);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем GpuMat для мнимой части фильтра Винера~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cuda::GpuMat zeroMatHLeft(cv::Size(a, b), CV_32F, Scalar(0)), complexHLeft;
	cuda::GpuMat zeroMatHRight(cv::Size(a, b), CV_32F, Scalar(0)), complexHRight;
	
	Ptr<cuda::DFT> forwardDFTLeft = cuda::createDFT(cv::Size(a, b), DFT_SCALE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> inverseDFTLeft = cuda::createDFT(cv::Size(a, b), DFT_INVERSE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> forwardDFTRight = cuda::createDFT(cv::Size(a, b), DFT_SCALE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> inverseDFTRight = cuda::createDFT(cv::Size(a, b), DFT_INVERSE | DFT_COMPLEX_INPUT);

    //------------------------------------------
    // First frame VidStab
    //------------------------------------------




    //------------------------------------------
    // END Initialize variables VideoStab
    //------------------------------------------




    // -----------------------------------------
    // Run visual odometry
    // -----------------------------------------
    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;
    
    
    // Добавляем переменные для интерполяции
    bool use_interpolation = false;
    cv::Mat last_valid_rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat last_valid_translation = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat interpolated_rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat interpolated_translation = cv::Mat::zeros(3, 1, CV_64F);
    int interpolation_frames = 0;
    const int max_interpolation_frames = 100; // Максимальное количество кадров для интерполяции
    
    // Добавляем переменные для визуальной одометрии
    cv::Mat imageLeft_t1_color, imageRight_t1_color;  
    cv::Mat imageRight_t1,  imageLeft_t1;
    cv::Mat imageRight_stab_t1,  imageLeft_stab_t1;
    cv::Mat points3D_t0_stab, points4D_t0_stab;
    cv::Vec3f rotation_euler_stab;
    cv::Mat state;
    
    //std::vector<cv::Point2f> oldPointsLeft_t0;
    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;
    cv::Mat rotation_stab = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_stab = cv::Mat::zeros(3, 1, CV_64F);

    //std::vector<cv::Point2f> oldPointsLeft_t0_stab;
    std::vector<cv::Point2f> pointsLeft_t0_stab, pointsRight_t0_stab, pointsLeft_t1_stab, pointsRight_t1_stab;
    double crop = framePart;

    cv::Vec3f rotation_euler;
    cv::Mat points3D_t0, points4D_t0;
    cv::Mat rigid_body_transformation;
    
    for (int frame_id = init_frame_id+1; frame_id < 50000; frame_id+=frame_skip)
    {
        imageRight_t1.release();
        imageLeft_t1.release();
        if(use_intel_rgbd)
        {
            pCamera->getLRFrames(imageLeft_t1,imageRight_t1);
        }
        else if (use_camera &&! use_intel_rgbd)
        {
            captureLeft >> imageLeft_t1_color;
            cvtColor(imageLeft_t1_color, imageLeft_t1, cv::COLOR_BGR2GRAY);
            cvtColor(imageLeft_t1_color, imageLeft_t1, cv::COLOR_BGR2GRAY);

            captureRight >> imageRight_t1_color;
            cvtColor(imageRight_t1_color, imageRight_t1, cv::COLOR_BGR2GRAY);
        }
        else
        {
            loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id%1+1, filepath);  //%1+1
            loadImageRight(imageRight_t1_color, imageRight_t1, frame_id%1+1, filepath);   
        }

        if (frame_id < 80 && frame_skip < 0)
            frame_skip = 1;
        if (frame_id > 13000 && frame_skip > 0)
            frame_skip = -1;
        noiseIn.dx = (double)(rng.uniform(-MaxShake, MaxShake))*0.0 + MaxShake*sin(frame_id*DEG_TO_RAD*40.0);
        //noiseIn.dy = (double)(rng.uniform(-MaxShake, MaxShake))*0.0 + MaxShake*cos(frame_id*DEG_TO_RAD*20.0);
        //noiseIn.da = (double)(rng.uniform(-sqrt(MaxShake)/1000, sqrt(MaxShake)/1000)) + 3.0*sqrt(MaxShake)/1000*sin(frame_id*DEG_TO_RAD*10.0);

        //noiseOut[0] = iirNoise(noiseIn, X,Y);
        noiseOut[0] = noiseIn;

        noiseOut[0].getTransform(Shake);
        cv::warpAffine(imageLeft_t1, imageLeft_t1, Shake, imageLeft_t1.size());
        cv::warpAffine(imageRight_t1, imageRight_t1, Shake, imageRight_t1.size());
        
		// imageLeft_t1 = imageLeft_t1(roi);
		// imageRight_t1 = imageRight_t1(roi);
        
        cv::resize(imageLeft_t1, imageLeft_t1, cv::Size(a, b), 0.0, 0.0, cv::INTER_CUBIC);
        cv::resize(imageRight_t1, imageRight_t1, cv::Size(a, b), 0.0, 0.0, cv::INTER_CUBIC);
        
        pointsLeft_t0_stab.clear();
        pointsRight_t0_stab.clear();
        pointsLeft_t1_stab.clear();
        pointsRight_t1_stab.clear();
        
        matchingFeaturesStab( imageLeft_t0, imageRight_t0,
                          imageLeft_t1, imageRight_t1, 
                          currentVOFeatures_stab,
                          pointsLeft_t0_stab, 
                          pointsRight_t0_stab, 
                          pointsLeft_t1_stab, 
                          pointsRight_t1_stab,
                          d_features,
                          0.6);

        cv::Mat tempImagForTest;
        imageLeft_t1.copyTo(tempImagForTest);

        getBiasAndRotation(pointsLeft_t0_stab, pointsLeft_t1_stab, dLeft, meanP0Left, transforms, TLeft, compression); //перемещение между кадрами оценивается как первая производная
        // std::cout << std::endl << "1 - TLeft = " << std::endl << TLeft<< std::endl;
                
        // points3D_t0_stab.release();
        // points4D_t0_stab.release();
        // if (pointsLeft_t0_stab.size()>5)
        // {
        //     cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0_stab,  pointsRight_t0_stab,  points4D_t0_stab);
        //     cv::convertPointsFromHomogeneous(points4D_t0_stab.t(), points3D_t0_stab);
        //     trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0_stab, pointsLeft_t1_stab, points3D_t0_stab, rotation_stab, translation_stab, frame_skip, false);
        //     cv::Mat temp_TLeft = (cv::Mat_<double>(2, 3) << 
        //     rotation_stab.at<double>(0, 0), rotation_stab.at<double>(0, 1), rotation_stab.at<double>(0, 2),
        //     rotation_stab.at<double>(1, 0), rotation_stab.at<double>(1, 1), rotation_stab.at<double>(1, 2));
        //     //transforms[1] = TransformParam(-temp_TLeft.at<double>(0, 2)*compression, -temp_TLeft.at<double>(1, 2)*compression, -atan2(temp_TLeft.at<double>(1, 0), temp_TLeft.at<double>(0, 0)));
        //     std::cout << "2 - TLeft = " << std::endl << temp_TLeft<< std::endl;
        //     std::cout << "3 - rotation_stab = " << std::endl << rotation_stab << std::endl;
        //     rotation_euler_stab = rotationMatrixToEulerAngles(rotation_stab);
        //     rotation_euler_stab[0] = rotation_euler_stab[0];
        //     rotation_euler_stab[1] = rotation_euler_stab[1];
        //     std::cout << "4 - rotation_euler_stab = " << std::endl << rotation_euler_stab << std::endl;
        // }
        //transforms[1] = TransformParam(-rotation_euler_stab[0]*fx*compression, -rotation_euler_stab[1]*fx*compression, -rotation_euler_stab[2]);
        
        iirAdaptiveHighPass(transforms, tauStab, roi, a, b, c, gain, movement, movementKalman); //интегрирование первой производной (получение смещения)
        if (gain < 1.0)
        {
            gain *=1.05;
            gain+=0.01;
        } 
        if (gain > 1.0)
        {
            gain = 1.0;
        }
        //showServiceInfoSmall(tempImagForTest, 1.0, 1.0, true, true, true, transforms, movementKalman, tauStab, gain, framePart, pointsLeft_t0_stab.max_size(), 1, 1.0, 1.0, 1.0, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab, fontFace, fontScale, colorBLACK);
        
        displayTracking(tempImagForTest, pointsLeft_t0_stab, pointsLeft_t1_stab, "1) test stab point area");

        //kf.update((cv::Mat_<double>(3, 1) << transforms[1].dx, transforms[1].dy, transforms[1].da));
        kf.update((cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0));

        cv::Mat state = kf.state();
        
        movementKalman[1].dx = state.at<double>(0, 0); //скорость
        movementKalman[1].dy = state.at<double>(1, 0); //скорость
        movementKalman[1].da = state.at<double>(6, 0); //скорость

        movementKalman[2].dx = state.at<double>(2, 0); //ускорение
        movementKalman[2].dy = state.at<double>(3, 0); //ускорение
        movementKalman[2].da = state.at<double>(7, 0); //ускорение

        movementKalman[3].dx = state.at<double>(4, 0); //вторая производная ускорения
        movementKalman[3].dy = state.at<double>(5, 0); //вторая производная ускорения
        movementKalman[3].da = state.at<double>(8, 0); //вторая производная ускорения

        transforms[0].getTransform(TStabLeft, a, b, c, atan_ba, framePart); // получение текущего компенсирующего преобразования
        transforms[0].getTransformInvert(TStabInvLeft, a, b, c, atan_ba, framePart); // получение текущего обратного компенсирующего преобразования для отрисовки маски
        
        // cout << "transforms[1]" << transforms[1].dx << " : " << transforms[1].dy << " : " << transforms[1].da << endl;
        // cout << "transforms[0]" << transforms[0].dx << " : " << transforms[0].dy << " : " << transforms[0].da << endl;
        // cout << "noiseOut[0]" << noiseOut[0].dx << " : " << noiseOut[0].dy << " : " << noiseOut[0].da << endl;
        // cout << "TStabLeft" <<TStabLeft.at<double>(0, 2) << " : " << TStabLeft.at<double>(1, 2) <<endl;

        gFrameLeft.upload(imageLeft_t1);
        gFrameRight.upload(imageRight_t1);

        cuda::warpAffine(gFrameLeft,  gFrameStabilizedLeft,  TStabLeft, cv::Size(a, b)); //8ms
        cuda::warpAffine(gFrameRight, gFrameStabilizedRight, TStabLeft, cv::Size(a, b)); //8ms

        gFrameStabilizatedCropLeft = gFrameStabilizedLeft(roi);
        gFrameStabilizatedCropRight = gFrameStabilizedRight(roi);

        //cuda::resize(gFrameStabilizatedCropLeft, gImageLeft_t0, cv::Size(a,b));
        cv::cuda::resize(gFrameStabilizatedCropLeft, gWriterFrameToShowLeft, cv::Size(a, b), 0.0, 0.0, cv::INTER_NEAREST);
        cv::cuda::resize(gFrameStabilizatedCropRight, gWriterFrameToShowRight, cv::Size(a, b), 0.0, 0.0, cv::INTER_NEAREST);
        gWriterFrameToShowLeft.download(imageLeft_stab_t1);
        gWriterFrameToShowRight.download(imageRight_stab_t1);
        showServiceInfoSmall(imageLeft_t1_color, 1.0, 1.0, true, true, true, transforms, movementKalman, tauStab, gain, framePart, pointsLeft_t0_stab.max_size(), 1, 1.0, 1.0, 1.0, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab, fontFace, fontScale, colorGREEN);
        
        imshow("imageLeft_t1_color", imageLeft_t1_color);
        //imshow("imageLeft_stab_t0", imageLeft_stab_t0);

        t_a = clock();

        //oldPointsLeft_t0 = currentVOFeatures.points;

        pointsLeft_t0.clear();
        pointsRight_t0.clear();
        pointsLeft_t1.clear();
        pointsRight_t1.clear();
        
        matchingFeatures( gain > 0.5 ? imageLeft_stab_t0 : imageLeft_t0, gain > 0.5 ? imageRight_stab_t0 : imageRight_t0,
                          gain > 0.5 ? imageLeft_stab_t1 : imageLeft_t1, gain > 0.5 ? imageRight_stab_t1 : imageRight_t1,
                          currentVOFeatures,
                          pointsLeft_t0, 
                          pointsRight_t0, 
                          pointsLeft_t1, 
                          pointsRight_t1,
                          1.0); //не доворачивает повороты

        imageLeft_t1.copyTo(imageLeft_t0);
        imageRight_t1.copyTo(imageRight_t0);

        imageLeft_stab_t1.copyTo(imageLeft_stab_t0);
        imageRight_stab_t1.copyTo(imageRight_stab_t0);

        // Проверяем количество найденных точек
        if (pointsLeft_t0.size() < 30 || pointsLeft_t1.size() < 30) {
            if (!use_interpolation && pointsLeft_t0.size() >= 15) {
                // Сохраняем последние валидные параметры движения перед началом интерполяции
                last_valid_rotation = rotation.clone();
                last_valid_translation = translation.clone();
                use_interpolation = true;
                interpolation_frames = 0;
            }
        
            if (use_interpolation) {
                // Используем линейную интерполяцию
                if (interpolation_frames < max_interpolation_frames) {
                    double alpha = (double)(interpolation_frames + 1) / (max_interpolation_frames + 1);
                    interpolated_rotation = last_valid_rotation * (1.0 - alpha) + rotation * alpha;
                    interpolated_translation = last_valid_translation * (1.0 - alpha) + translation * alpha;
                    
                    // Используем интерполированные значения
                    rotation = interpolated_rotation.clone();
                    translation = interpolated_translation.clone();
                    interpolation_frames++;
                    
                    std::cout << "[Info] Using interpolation, frames: " << interpolation_frames 
                              << ", points found: " << pointsLeft_t0.size() << std::endl;
                } else {
                   // Сбрасываем интерполяцию если слишком долго не находим точки
                    use_interpolation = false;
                    std::cout << "[Warning] Interpolation timeout, resetting..." << std::endl;
                }
            } else {
                // Пропускаем кадр если точек слишком мало и интерполяция не активна
                std::cout << "[Warning] Too few points (" << pointsLeft_t0.size() 
                          << "), skipping frame..." << std::endl;
                continue;
            }
        } else {
            // Достаточно точек - нормальная обработка
            if (use_interpolation) {
                use_interpolation = false;
                std::cout << "[Info] Enough points found, stopping interpolation" << std::endl;
            }
        

            // ---------------------
            // Triangulate 3D Points
            // ---------------------
            points3D_t0.release();
            points4D_t0.release();
            cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
            cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

            // ---------------------
            // Tracking transformation
            // ---------------------
            clock_t tic_gpu = clock();
            trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, 
                           points3D_t0, rotation, translation, frame_skip, false);
            clock_t toc_gpu = clock();
        
            // Сохраняем валидные параметры движения
            last_valid_rotation = rotation.clone();
            last_valid_translation = translation.clone();
        }

        // cv::Mat tempImage;
        // cv::addWeighted(imageLeft_t1, 0.25, imageRight_t1, 0.25, 1.4, tempImage);

        displayTracking(imageLeft_stab_t1, pointsLeft_t0, pointsLeft_t1, "vis_left"); //show input image

        // displayTracking(imageRight_t1, pointsRight_t0, pointsRight_t1, "vis_right"); //show input image
        // displayTracking(tempImage, pointsRight_t0, pointsLeft_t0, "vis_both"); //show input image

        // ------------------------------------------------
        // Integrating and display
        // ------------------------------------------------
        rotation_euler = rotationMatrixToEulerAngles(rotation);

        rigid_body_transformation.release();

        if(abs(rotation_euler[1])<0.4*MaxShake*abs(frame_skip) && abs(rotation_euler[0])<0.4*MaxShake*abs(frame_skip) && abs(rotation_euler[2])<0.4*MaxShake*abs(frame_skip))
        {
            integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, 
                               rotation, translation);
        } else {
            std::cout << "Too large rotation" << std::endl;
        }
    
        t_b = clock();
        float frame_time = 1000*(double)(t_b-t_a)/CLOCKS_PER_SEC;
        float fps = 1000/frame_time;

        cv::Mat xyz = frame_pose.col(3).clone();
        display(frame_id, trajectory, trajectory_biased, xyz, pose_matrix_gt, fps, display_ground_truth);

        int key = cv::waitKey(1);
        if (key == 'w')
            {
                //MaxShake *= 1.1;
                //cout << "MaxShake = " << MaxShake << endl;
                frame_skip++;
                cout << "frame_skip = " << frame_skip << endl;
            }
        else if (key == 's' && frame_skip > 1)
            {
                //MaxShake /= 1.1;
                //cout << "MaxShake = " << MaxShake << endl;
                frame_skip--;
                cout << "frame_skip = " << frame_skip << endl;
            }
        else if (key == 'p' || frame_id%1500 == 0)
            {
            string trajectory_picture_1 = "trajectory_Shake_";
            string trajectory_picture_2 = "FrameSkip_";
            string trajectory_picture_3 = ".jpg";
            string trajectory_picture = trajectory_picture_1 + to_string(MaxShake) + trajectory_picture_2 + to_string(frame_skip) + trajectory_picture_3;
            cv::imwrite(trajectory_picture, trajectory);
            cout << frame_id << endl;
            }
        else if (key == 27 || frame_id > 17500){

            break;
        }
    }
    return 0; //banana PI
}

