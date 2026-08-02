#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <vector>
#include <ctime>
#include <string>

#include "feature.h"
#include "utils.h"
#include "evaluate_odometry.h"
#include "visualOdometry.h"
#include "loopclosure.h"

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
    
    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    bool computeTest = false;
    bool display_ground_truth = false;
    bool use_intel_rgbd = false;
    bool use_camera = false;
    std::vector<Matrix> pose_matrix_gt;
    
    // Sequence
    constexpr const char* filepath = "/home/selbizo/CV/dataset/sequences/00/";
    cout << "Filepath: " << filepath << endl;

    constexpr const char* rgbd_str = "rgbd";
    constexpr const char* camera_str = "camera";
    
    if(filepath == rgbd_str) use_intel_rgbd = true;
    else if(filepath == camera_str) use_camera = true;

    // Camera calibration
    constexpr const char* strSettingPath = "../calibration/kitti00.yaml";
    cout << "Calibration Filepath: " << strSettingPath << endl;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    if(!fSettings.isOpened()) {
        cerr << "Error: Cannot open calibration file " << strSettingPath << endl;
        return -1;
    }
    int frame_skip = 1;

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
    cv::Mat trajectory = cv::Mat::zeros(3000, 3000, CV_8UC3);
    cv::Mat trajectory_biased = cv::Mat::zeros(400, 400, CV_8UC3);
    FeatureSet currentVOFeatures;
    FeatureSet currentVOFeatures_stab;
    cv::Mat points4D, points3D;
    int init_frame_id = 0; //126
    int local_loop_ceiling = 4470;
    
    bool loop_detected = false;
    int last_loop_frame_id = -100;

    //--------------------------------
    // Initialize variables VideoShake
    //--------------------------------
    bool shakeEnabled = false;
    Mat Shake(2, 3, CV_64F);
    TransformParam noiseIn = { 0.0, 0.0, 0.0 };
    vector <TransformParam> noiseOut(2);

	for (auto& noise : noiseOut) noise = {0.0, 0.0, 0.0};
    
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
    
    bool stabEnabled = false;
    /*Заключение: 
    стабилизация ухудшает точность VO, но улучшает визуальное качество видео. 
    Включать стабилизацию или нет - зависит от задачи.
    */
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

	const unsigned int firSize = 4;
    vector <TransformParam> transforms(firSize), movement(firSize), movementKalman(firSize);

	for (auto& t : transforms) t = {0.0, 0.0, 0.0};
	for (auto& m : movement) m = {0.0, 0.0, 0.0};
	for (auto& k : movementKalman) k = {0.0, 0.0, 0.0};
     
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
    cv::Mat imageRight_t0, imageLeft_t0, imageLeft_stab_t0, imageRight_stab_t0;
    CameraBase* pCamera = nullptr;
    cv::VideoCapture captureLeft, captureRight;
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
    const double c = std::sqrt(a * a + b * b);
    const double atan_ba = std::atan2(b, a);
    
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float bf = fSettings["Camera.bf"];
    
    double framePart = 1.0;
    if(framePart < 1.0) {
        fx /= framePart;
        fy /= framePart;
        cx -= a * (1.0 - framePart) / 2.0;
        cy -= b * (1.0 - framePart) / 2.0;
        bf /= framePart;
    }
    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
    cout << "P_left: " << endl << projMatrl << endl;
    cout << "P_right: " << endl << projMatrr << endl;
    
    LoopClosure loopClosure;
    loopClosure.setCameraParameters(projMatrl, projMatrr);
    loopClosure.setParameters(30, 0.6f, 0.55f, 10, 10, 50.0f);
    loopClosure.setMaxPoseDistance(30.0f);
    loopClosure.setMinLoopGap(50);
    loopClosure.setMaxLoopDistance(200.0f);
    loopClosure.setMinKeyframeDistance(50.0f);
    loopClosure.setDebugMode(true);
    
    bool check_loopclosure = true;

    double MaxShake = b * (1.0 - framePart) / 2.0;

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
	
    const Rect rect_search(0, 0, a * 0.5, b * 0.5 / compression);
    cv::rectangle(maskSearchLeft, rect_search, Scalar(255), FILLED);
    cv::rectangle(maskSearchRight, rect_search, Scalar(255), FILLED);
    
	cuda::GpuMat gMaskSearchLeft(maskSearchLeft);
    cuda::GpuMat gMaskSearchRight(maskSearchRight);

	Mat maskSearchSmallLeft = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	Mat maskSearchSmallRight = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	
    const int max_dim = std::max(a, b);
    const Rect rect_small(0, 0, max_dim * 0.3 / compression, max_dim * 0.3 / compression);
    cv::rectangle(maskSearchSmallLeft, rect_small, Scalar(255), FILLED);
    cv::rectangle(maskSearchSmallRight, rect_small, Scalar(255), FILLED);
    
	cuda::GpuMat gMaskSearchSmallLeft(maskSearchSmallLeft);
	cuda::GpuMat gMaskSearchSmallRight(maskSearchSmallRight);
    cuda::GpuMat gMaskSearchSmallRoiLeft, gMaskSearchSmallRoiRight;

	Mat roiMaskLeft = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	Mat roiMaskRight = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
    const Rect rect_roi(0, 0, a * 0.4, b * 0.4 / compression);
    cv::rectangle(roiMaskLeft, rect_roi, Scalar(255), FILLED);
    cv::rectangle(roiMaskRight, rect_roi, Scalar(255), FILLED);
	//cuda::GpuMat gRoiMask(roiMask);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем GpuMat для мнимой части фильтра Винера~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cuda::GpuMat zeroMatHLeft(cv::Size(a, b), CV_32F, Scalar(0)), complexHLeft;
	cuda::GpuMat zeroMatHRight(cv::Size(a, b), CV_32F, Scalar(0)), complexHRight;
	
	Ptr<cuda::DFT> forwardDFTLeft = cuda::createDFT(cv::Size(a, b), DFT_SCALE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> inverseDFTLeft = cuda::createDFT(cv::Size(a, b), DFT_INVERSE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> forwardDFTRight = cuda::createDFT(cv::Size(a, b), DFT_SCALE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> inverseDFTRight = cuda::createDFT(cv::Size(a, b), DFT_INVERSE | DFT_COMPLEX_INPUT);


    // -----------------------------------------
    // Run visual odometry
    // -----------------------------------------
    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;
    
    
    // Добавляем переменные для интерполяции
    bool use_interpolation = false;
    cv::Mat last_valid_rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat last_valid_translation = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat interpolated_rotation, interpolated_translation;
    int interpolation_frames = 0;
    const int max_interpolation_frames = 100;
    
    // Добавляем переменные для визуальной одометрии
    cv::Mat imageLeft_t1_color, imageRight_t1_color;  
    cv::Mat imageRight_t1,  imageLeft_t1;
    cv::Mat imageRight_stab_t1,  imageLeft_stab_t1;
    cv::Mat points3D_t0_stab, points4D_t0_stab;
    cv::Vec3f rotation_euler_stab;
    cv::Mat state;
    
    // Переменные для отслеживания поворотов и добавления ключевых кадров
    cv::Mat last_rotation_euler_mat = cv::Mat::zeros(3, 1, CV_64F);
    int frames_since_last_keyframe = 0;
    const int normal_keyframe_interval = 10;
    
    //std::vector<cv::Point2f> oldPointsLeft_t0;
    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;
    cv::Mat rotation_stab = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_stab = cv::Mat::zeros(3, 1, CV_64F);

    //std::vector<cv::Point2f> oldPointsLeft_t0_stab;
    std::vector<cv::Point2f> pointsLeft_t0_stab, pointsRight_t0_stab, pointsLeft_t1_stab, pointsRight_t1_stab;
    // double crop = framePart;

    cv::Vec3f rotation_euler;
    cv::Mat points3D_t0, points4D_t0;
    cv::Mat rigid_body_transformation;
    
    for (int frame_id = init_frame_id+1; frame_id < 15000; frame_id+=frame_skip)
    {

        imageRight_t1.release();
        imageLeft_t1.release();
        imageRight_t1_color.release();
        imageLeft_t1_color.release();
        
        if(use_intel_rgbd) {
            pCamera->getLRFrames(imageLeft_t1, imageRight_t1);
        } else if(use_camera) {
            captureLeft >> imageLeft_t1_color;
            cvtColor(imageLeft_t1_color, imageLeft_t1, cv::COLOR_BGR2GRAY);
            captureRight >> imageRight_t1_color;
            cvtColor(imageRight_t1_color, imageRight_t1, cv::COLOR_BGR2GRAY);
        } else {
            const int adjusted_frame = (frame_id < local_loop_ceiling) 
                ? frame_id % local_loop_ceiling 
                : frame_id % local_loop_ceiling + init_frame_id;
            loadImageLeft(imageLeft_t1_color, imageLeft_t1, adjusted_frame, filepath);
            loadImageRight(imageRight_t1_color, imageRight_t1, adjusted_frame, filepath);
        }
        if(shakeEnabled) {
            noiseIn.dx = MaxShake * sin(frame_id * DEG_TO_RAD * 50.0);
            noiseIn.dy = MaxShake * cos(frame_id * DEG_TO_RAD * 41.0);
            noiseIn.da = 3.0 * sqrt(MaxShake) / 1000.0 * sin(frame_id * DEG_TO_RAD * 10.0);
            
            noiseOut[0] = noiseIn;
            noiseOut[0].getTransform(Shake);
            
            cv::warpAffine(imageLeft_t1, imageLeft_t1, Shake, imageLeft_t1.size());
            cv::warpAffine(imageRight_t1, imageRight_t1, Shake, imageRight_t1.size());
        }
        if (stabEnabled == true) //для исследования влияния стабилизации видео на визуальную одометрию можно включить стабилизацию видео
        {
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
                            0.3);

            cv::Mat tempImagForTest;
            imageLeft_t1.copyTo(tempImagForTest);

            getBiasAndRotation(pointsLeft_t0_stab, pointsLeft_t1_stab, dLeft, meanP0Left, transforms, TLeft, compression); //перемещение между кадрами оценивается как первая производная
            std::cout << std::endl << "1 - TLeft = " << std::endl << TLeft<< std::endl;
                    
            points3D_t0_stab.release();
            points4D_t0_stab.release();
            if (pointsLeft_t0_stab.size()>5)
            {
                cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0_stab,  pointsRight_t0_stab,  points4D_t0_stab);
                cv::convertPointsFromHomogeneous(points4D_t0_stab.t(), points3D_t0_stab);
                trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0_stab, pointsLeft_t1_stab, points3D_t0_stab, rotation_stab, translation_stab, frame_skip, false);
                cv::Mat temp_TLeft = (cv::Mat_<double>(2, 3) << 
                rotation_stab.at<double>(0, 0), rotation_stab.at<double>(0, 1), rotation_stab.at<double>(0, 2),
                rotation_stab.at<double>(1, 0), rotation_stab.at<double>(1, 1), rotation_stab.at<double>(1, 2));
                cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
                                                projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2),
                                                projMatrl.at<float>(2, 0), projMatrl.at<float>(2, 1), projMatrl.at<float>(2, 2));

            }
                    
            if(gain < 1.0) {
                gain = gain * 1.05 + 0.01;
                if(gain > 1.0) gain = 1.0;
            }
            iirAdaptive(transforms, tauStab, roi, a, b, c, gain, movement, movementKalman); //интегрирование первой производной (получение смещения)

            displayTracking(tempImagForTest, pointsLeft_t0_stab, pointsLeft_t1_stab, "1) before stab feture points map");

            transforms[0].getTransform(TStabLeft, a, b, c, atan_ba, framePart); // получение текущего компенсирующего преобразования
            transforms[0].getTransformInvert(TStabInvLeft, a, b, c, atan_ba, framePart); // получение текущего обратного компенсирующего преобразования для отрисовки маски

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
        } else if (framePart < 1.0)
        {
            gFrameLeft.upload(imageLeft_t1);
            gFrameRight.upload(imageRight_t1);

            gFrameStabilizatedCropLeft = gFrameLeft(roi);
            gFrameStabilizatedCropRight = gFrameRight(roi);
            cv::cuda::resize(gFrameStabilizatedCropLeft, gWriterFrameToShowLeft, cv::Size(a, b), 0.0, 0.0, cv::INTER_LINEAR);
            cv::cuda::resize(gFrameStabilizatedCropRight, gWriterFrameToShowRight, cv::Size(a, b), 0.0, 0.0, cv::INTER_LINEAR);
            gWriterFrameToShowLeft.download(imageLeft_t1);
            gWriterFrameToShowRight.download(imageRight_t1);
        }
        if (computeTest == true)
        {
            showServiceInfoSmall(imageLeft_t1_color, 1.0, 1.0, true, true, true, transforms, movementKalman, tauStab, gain, framePart, pointsLeft_t0_stab.max_size(), 1, 1.0, 1.0, 1.0, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab, fontFace, fontScale, colorGREEN);
            imshow("imageLeft_t1_color", imageLeft_t1_color);

        }

        t_a = clock();

        pointsLeft_t0.clear();
        pointsRight_t0.clear();
        pointsLeft_t1.clear();
        pointsRight_t1.clear();
        
        matchingFeatures( stabEnabled && gain > 0.5 ? imageLeft_stab_t0 : imageLeft_t0, stabEnabled && gain > 0.5 ? imageRight_stab_t0 : imageRight_t0,
                          stabEnabled && gain > 0.5 ? imageLeft_stab_t1 : imageLeft_t1, stabEnabled && gain > 0.5 ? imageRight_stab_t1 : imageRight_t1,
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
        if(pointsLeft_t0.size() < 30 || pointsLeft_t1.size() < 30) {
            if(!use_interpolation && pointsLeft_t0.size() >= 15) {
                last_valid_rotation = rotation.clone();
                last_valid_translation = translation.clone();
                use_interpolation = true;
                interpolation_frames = 0;
            }
        
            if(use_interpolation && interpolation_frames < max_interpolation_frames) {
                const double alpha = static_cast<double>(interpolation_frames + 1) / (max_interpolation_frames + 1);
                interpolated_rotation = last_valid_rotation * (1.0 - alpha) + rotation * alpha;
                interpolated_translation = last_valid_translation * (1.0 - alpha) + translation * alpha;
                
                rotation = interpolated_rotation.clone();
                translation = interpolated_translation.clone();
                interpolation_frames++;
                
                std::cout << "[Info] Using interpolation, frames: " << interpolation_frames 
                          << ", points found: " << pointsLeft_t0.size() << std::endl;
            } else {
                if(!use_interpolation) {
                    std::cout << "[Warning] Too few points (" << pointsLeft_t0.size() 
                              << "), skipping frame..." << std::endl;
                } else {
                    std::cout << "[Warning] Interpolation timeout, resetting..." << std::endl;
                    use_interpolation = false;
                }
                continue;
            }
        } else {
            if(use_interpolation) {
                use_interpolation = false;
                std::cout << "[Info] Enough points found, stopping interpolation" << std::endl;
            }

            // ---------------------
            // Triangulate 3D Points
            // ---------------------
            points3D_t0.release();
            points4D_t0.release();
            cv::triangulatePoints(projMatrl, projMatrr, pointsLeft_t0, pointsRight_t0, points4D_t0);
            cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

            // ---------------------
            // Tracking transformation
            // ---------------------
            clock_t tic_gpu = clock();
            trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, 
                           points3D_t0, rotation, translation, frame_skip, false);
            clock_t toc_gpu = clock();
        
            last_valid_rotation = rotation.clone();
            last_valid_translation = translation.clone();
        }



        displayTracking(stabEnabled ? imageLeft_stab_t1 : imageLeft_t1, pointsLeft_t0, pointsLeft_t1, "vis_left"); //show input image


        
        if(loop_detected) {
            const cv::Mat R_corr = loopClosure.getLoopRotation();
            const cv::Mat t_corr = loopClosure.getLoopTranslation();
            const int candidate_id = loopClosure.getCandidateKeyframeId();
            
            const double t_norm = cv::norm(t_corr);
            if(t_norm <= 1000) {
                cv::Mat T_correction = cv::Mat::eye(4, 4, CV_64F);
                R_corr.copyTo(T_correction(cv::Rect(0, 0, 3, 3)));
                t_corr.copyTo(T_correction(cv::Rect(3, 0, 1, 3)));
                
                T_correction = T_correction * frame_pose;
                T_correction.copyTo(frame_pose);
                
                loopClosure.applyCorrectionToKeyframes();
                setLoopClosureCorrection(candidate_id, frame_id, R_corr, t_corr);
            }
            
            loop_detected = false;
        }

        // Integrating and display

        
        rotation_euler = rotationMatrixToEulerAngles(rotation);

        rigid_body_transformation.release();

        if(abs(rotation_euler[1]) < 0.4*(MaxShake + 2)*abs(frame_skip) && 
           abs(rotation_euler[0]) < 0.4*(MaxShake + 2)*abs(frame_skip) && 
           abs(rotation_euler[2]) < 0.4*(MaxShake + 2)*abs(frame_skip)) {
            integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation);
        } else {
            std::cout << "Too large rotation" << std::endl;
        }

        // Проверяем поворот для определения необходимости добавления ключевого кадра
        bool needs_keyframe_after_turn = false;
        if (frame_id > init_frame_id + 1) {
            cv::Mat current_euler_mat = cv::Mat(rotation_euler).clone();
            current_euler_mat.convertTo(current_euler_mat, CV_64F);
            
            double yaw_change = fabs(current_euler_mat.at<double>(0, 0) - last_rotation_euler_mat.at<double>(0, 0));
            if (yaw_change > 60.0 * DEG_TO_RAD) {
                needs_keyframe_after_turn = true;
            }
            last_rotation_euler_mat = current_euler_mat.clone();
        }

        bool should_add_keyframe = false;
        if (points3D_t0.rows >= 30) {
            frames_since_last_keyframe++;
            
            if (needs_keyframe_after_turn) {
                should_add_keyframe = true;
                frames_since_last_keyframe = 0;
            } else if (frames_since_last_keyframe >= normal_keyframe_interval) {
                should_add_keyframe = true;
                frames_since_last_keyframe = 0;
            }
        }

        // Loop closure detection - добавляем каждый кадр с дескриптором
        loopClosure.addFrame(frame_id, imageLeft_t1, imageRight_t1,
                            pointsLeft_t0, pointsRight_t0,
                            rotation, translation, points3D_t0, frame_pose, should_add_keyframe);
        
        if(should_add_keyframe) {
            std::cout << "[Keyframe] Frame " << frame_id << " added as keyframe" << std::endl;
        }
        
        if (check_loopclosure && loopClosure.detectLoop()) {
            int candidate_id = loopClosure.getCandidateKeyframeId();
            int min_loop_interval = 20;
            if (frame_id - last_loop_frame_id >= min_loop_interval) {
                loop_detected = true;
                last_loop_frame_id = frame_id;
                std::cout << "[LoopClosure] Frame " << frame_id << " <-> Frame " << candidate_id << " loop detected!" << std::endl;
                
                const cv::Mat current_desc = loopClosure.getCurrentDescriptor();
                const cv::Mat candidate_desc = loopClosure.getCandidateDescriptor();
                float similarity = loopClosure.getCurrentSimilarity();
                std::cout << "[LoopClosure] Similarity: " << similarity << ", Candidate ID: " << candidate_id << std::endl;
                
                int num_matches = loopClosure.getMatchCount();
                std::cout << "[LoopClosure] Number of matches: " << num_matches << std::endl;
            } else {
                std::cout << "[LoopClosure] Frame " << frame_id << " skip (last loop at " << last_loop_frame_id << ")" << std::endl;
            }
        } else if (check_loopclosure && loopClosure.getCurrentSimilarity() > 0.5f) {
            float max_sim = loopClosure.getCurrentSimilarity();
            int num_checked = loopClosure.getKeyframeCount() - 1;
            std::cout << "[LoopClosure] Frame " << frame_id << " checked " << num_checked 
                     << " keyframes, max similarity: " << max_sim << std::endl;
        }
    
        t_b = clock();
        float frame_time = 1000*(double)(t_b-t_a)/CLOCKS_PER_SEC;
        float fps = 1000/frame_time;

        cv::Mat xyz = frame_pose.col(3).clone(); 
        //где-то здесь нужно 

        display(frame_id, trajectory, trajectory_biased, xyz, fps);

        const int key = cv::waitKey(1);
        if(key == 'w') {
            frame_skip++;
            cout << "frame_skip = " << frame_skip << endl;
        } else if(key == 's' && frame_skip > 1) {
            frame_skip--;
            cout << "frame_skip = " << frame_skip << endl;
        } else if(key == 'p' || key == 27 || frame_id > 15800) {
            if(key == 'p' || frame_id % 1000 == 0) {
                const string traj_file = string("trajectory_Shake_") + 
                    to_string(MaxShake) + "_FrameSkip_" + to_string(frame_skip) + ".jpg";
                cv::imwrite(traj_file, trajectory);
            }
            if(key == 27 || frame_id > 1580) break;
        }
    }
    return 0;
}

