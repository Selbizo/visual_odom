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

//#include "ConfigVideoStab.h"
#include "basicFunctions.h"
#include "stabilizationFunctions.h"
// #include "wienerFilter.h"
// #include "basicStructs.h"


using namespace std;
using namespace cv;

// int main(int argc, char **argv)
int main()
{

    #if USE_CUDA
        printf("CUDA is Enabled\n");
    #endif
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
    unsigned int frame_skip = 1;
    
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float bf = fSettings["Camera.bf"];

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
    cv::Mat trajectory = cv::Mat::zeros(1400, 1400, CV_8UC3);
    cv::Mat trajectory_biased = cv::Mat::zeros(900, 900, CV_8UC3);
    FeatureSet currentVOFeatures;
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
	Mat oldFrame, oldGray, err;
	
	vector<Point2f> p0, p1, good_new;
	cuda::GpuMat gP0, gP1;
    
	Point2f d = Point2f(0.0f, 0.0f);
	Point2f meanP0 = Point2f(0.0f, 0.0f);
    
	Mat T, TStab(2, 3, CV_64F), TStabInv(2, 3, CV_64F), TSearchPoints(2, 3, CV_64F);
	cuda::GpuMat gT, gTStab(2, 3, CV_64F);
    
	vector<uchar> status;
	cuda::GpuMat gStatus, gErr;
	
	double tauStab = 100.0;
	double kSwitch = 0.01;
	double framePart = 0.8;
    
	const unsigned int firSize = 4;
    vector <TransformParam> transforms(firSize);
    vector <TransformParam> movement(firSize);
    vector <TransformParam> movementKalman(firSize);

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
    cv::Mat imageRight_t0,  imageLeft_t0;
    CameraBase *pCamera = NULL;
    cv::VideoCapture captureLeft;
    cv::VideoCapture captureRight;
    // cv::VideoCapture captureLeft("http://192.168.8.106:4747/video?640x480");
    // cv::VideoCapture captureRight("http://192.168.8.107:4747/video?640x480");
    
    
    if(use_intel_rgbd)
    {   
        pCamera = new Intel_V4L2;
        for (int throw_frames = 10 ; throw_frames >=0 ; throw_frames--)
        pCamera->getLRFrames(imageLeft_t0,imageRight_t0);
    }
    else if (use_camera &&! use_intel_rgbd)
    {
        cv::Mat imageLeft_t0_color;
        cv::Mat imageRight_t0_color;  
        captureLeft >> imageLeft_t0_color;
        cvtColor(imageLeft_t0_color, imageLeft_t0, cv::COLOR_BGR2GRAY);

        //imageLeft_t0_color.copyTo(imageRight_t0_color);
        // imageLeft_t0.copyTo(imageRight_t0);
        captureRight >> imageRight_t0_color;
        cvtColor(imageRight_t0_color, imageRight_t0, cv::COLOR_BGR2GRAY);
    }
    else
    {
        cv::Mat imageLeft_t0_color;
        loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath);
        
        cv::Mat imageRight_t0_color;  
        loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);
    }
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
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    noiseIn.dx = (double)(rng.uniform(-10.0, 10.0))/4;// / 32 + noiseIn.dx * 31 / 32;
    noiseIn.dy = (double)(rng.uniform(-10.0, 10.0))/4;//    / 32 + noiseIn.dy * 31 / 32;
    noiseIn.da = (double)(rng.uniform(-1.0, 1.0));// / 32 + noiseIn.da * 31 / 32;

    noiseOut[0] = iirNoise(noiseIn, X,Y);

    noiseOut[0].getTransform(Shake);
    cv::warpAffine(imageLeft_t0, imageLeft_t0, Shake, imageLeft_t0.size());

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

    for (int frame_id = init_frame_id+1; frame_id < 4800000; frame_id+=frame_skip)
    //for(;;)
    {

        //std::cout << std::endl << "frame id " << frame_id << std::endl;
        // ------------
        // Load images
        // ------------
        cv::Mat imageRight_t1,  imageLeft_t1;
        if(use_intel_rgbd)
        {
            pCamera->getLRFrames(imageLeft_t1,imageRight_t1);
        }
            else if (use_camera &&! use_intel_rgbd)
        {
            cv::Mat imageLeft_t1_color;
            cv::Mat imageRight_t1_color;  
            captureLeft >> imageLeft_t1_color;
            cvtColor(imageLeft_t1_color, imageLeft_t1, cv::COLOR_BGR2GRAY);

            //imageLeft_t1_color.copyTo(imageRight_t1_color);
            //imageLeft_t1.copyTo(imageRight_t1);
            //cv::VideoCapture captureRight(0);
            captureRight >> imageRight_t1_color;
            cvtColor(imageRight_t1_color, imageRight_t1, cv::COLOR_BGR2GRAY);
        }
        else
        {
            cv::Mat imageLeft_t1_color;
            loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id, filepath);  
            cv::Mat imageRight_t1_color;  
            loadImageRight(imageRight_t1_color, imageRight_t1, frame_id, filepath);      
        }

        noiseIn.dx = (double)(rng.uniform(-20.0, 20.0));
        noiseIn.dy = (double)(rng.uniform(-20.0, 20.0));
        noiseIn.da = (double)(rng.uniform(-0.1, 0.1));

        noiseIn.dx = (double)(rng.uniform(-40.0, 40.0));
        noiseIn.dy = (double)(rng.uniform(-40.0, 40.0));
        // noiseIn.da = (double)(rng.uniform(-0.1, 0.1));

        noiseOut[0] = iirNoise(noiseIn, X,Y);

        noiseOut[0].getTransform(Shake);
        cv::warpAffine(imageLeft_t0, imageLeft_t0, Shake, imageLeft_t0.size());
        cv::warpAffine(imageRight_t0, imageRight_t0, Shake, imageRight_t0.size());

        //video stab begins
        

        //video stab ends


        t_a = clock();
        std::vector<cv::Point2f> oldPointsLeft_t0 = currentVOFeatures.points;


        std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  
        matchingFeatures( imageLeft_t0, imageRight_t0,
                          imageLeft_t1, imageRight_t1, 
                          currentVOFeatures,
                          pointsLeft_t0, 
                          pointsRight_t0, 
                          pointsLeft_t1, 
                          pointsRight_t1);  

        imageLeft_t0 = imageLeft_t1;
        imageRight_t0 = imageRight_t1;

        std::vector<cv::Point2f>& currentPointsLeft_t0 = pointsLeft_t0;
        std::vector<cv::Point2f>& currentPointsLeft_t1 = pointsLeft_t1;
        
        std::vector<cv::Point2f> newPoints;
        std::vector<bool> valid; // valid new points are ture

        // ---------------------
        // Triangulate 3D Points
        // ---------------------
        cv::Mat points3D_t0, points4D_t0;
        cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
        cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

/*        cv::Mat points3D_t1, points4D_t1;
        cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t1,  pointsRight_t1,  points4D_t1);
        cv::convertPointsFromHomogeneous(points4D_t1.t(), points3D_t1);*/

        // ---------------------
        // Tracking transfomation
        // ---------------------
	clock_t tic_gpu = clock();
        trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, points3D_t0, rotation, translation, frame_skip, false);
	clock_t toc_gpu = clock();
	// std::cerr << "tracking frame 2 frame: " << float(toc_gpu - tic_gpu)/CLOCKS_PER_SEC*1000 << "ms" << std::endl;
        displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);


/*        points4D = points4D_t0;
        frame_pose.convertTo(frame_pose32, CV_32F);
        points4D = frame_pose32 * points4D;
        cv::convertPointsFromHomogeneous(points4D.t(), points3D);*/

        // ------------------------------------------------
        // Intergrating and display
        // ------------------------------------------------

        cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);


        cv::Mat rigid_body_transformation;

        if(abs(rotation_euler[1])<0.1 && abs(rotation_euler[0])<0.1 && abs(rotation_euler[2])<0.1)
        {
            integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation);

        } else {

            std::cout << "Too large rotation"  << std::endl;
        }
        t_b = clock();
        float frame_time = 1000*(double)(t_b-t_a)/CLOCKS_PER_SEC*2;
        float fps = 1000/frame_time;
        //cout << "[Info] frame times (ms): " << frame_time << endl;
        //cout << "[Info] FPS: " << fps << endl;

        // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;
        // std::cout << "rotation: " << rotation_euler << std::endl;
        // std::cout << "translation: " << translation.t() << std::endl;
        // std::cout << "frame_pose" << frame_pose << std::endl;


        cv::Mat xyz = frame_pose.col(3).clone();
        display(frame_id, trajectory, trajectory_biased, xyz, pose_matrix_gt, fps, display_ground_truth);

        int key = cv::waitKey(3);
        if (key == 27)
            break;
    }

    return 0;
}

