#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <fstream>
#include <string>
#include <iomanip>

#include "feature.h"
#include "utils.h"
#include "evaluate_odometry.h"
#include "visualOdometry.h"
#include "Frame.h"

#include "camera_object.h"
#include "rgbd_standalone.h"

#include "basicFunctions.h"
#include "stabilizationFunctions.h"
#include "kalmanSplitter.h"
#include "VideoStabilizationPipeline.h"


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
    

    // Sequence
    //string filepath = string(argv[1]);
    string filepath = string("/home/selbizo/CV/dataset/sequences/00/");
    cout << "Filepath: " << filepath << endl;

    if(filepath == "rgbd") use_intel_rgbd = true;
    if(filepath == "camera") use_camera = true;



    // ========================================
    // Инициализация VideoStabilizationPipeline
    // (будет выполнена после загрузки первого кадра, когда известны размеры)
    // ========================================

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
	
	double tauStab = 5.0;
	double gain = 0.7;
	//double framePart = 0.95;

   

    // ------------------------
    // Load first images
    // ------------------------
    cv::Mat imageRight_t0,  imageLeft_t0, imageLeft_stab_t0, imageRight_stab_t0;
    CameraBase *pCamera = NULL;
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
        loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath);
        loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);
    }
    imageLeft_t0.copyTo(imageLeft_stab_t0);
    imageRight_t0.copyTo(imageRight_stab_t0);

    clock_t t_a, t_b;

    //init sizes of frames
	const int a = imageLeft_t0.cols;
	const int b = imageLeft_t0.rows;

    // Camera calibration
    // Use absolute path to avoid working directory issues
    string strSettingPath = string("/home/selbizo/CV/StabAndSLAM/visual_odom/calibration/kitti00.yaml");
    cout << "Calibration Filepath: " << strSettingPath << endl;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    if (!fSettings.isOpened())
    {
        cerr << "ERROR: Failed to open calibration file: " << strSettingPath << endl;
        return 1;
    }
    int frame_skip = 1;
    
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float bf = fSettings["Camera.bf"];


    
    double framePart = 0.99;
    float dx = a * (1.0 - framePart) / 2.0;
    float dy = b * (1.0 - framePart) / 2.0;
    fx = fx/framePart;
    fy = fy/framePart;
    cx = cx - dx;
    cy = cy - dy;

    bf = bf/framePart;
    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
    cout << "P_left: " << endl << projMatrl << endl;
    cout << "P_right: " << endl << projMatrr << endl;

    double MaxShake = b * (1.0 - framePart) / 2.0;


    // ========================================
    // VideoStabilizationPipeline — высокоуровневый контур стабилизации
    // ========================================
    VideoStabilizationPipeline stabPipeline;
    bool usePipeline = true;  // переключатель: true = pipeline, false = старый код

    // ========================================
    // Инициализация VideoStabilizationPipeline
    // ========================================
    if (usePipeline) {
        stabPipeline.init(fx, fy, cx, cy, bf, a, b, compression, framePart);
        std::cout << "[Pipeline] Initialized with fx=" << fx << " fy=" << fy 
                  << " cx=" << cx << " cy=" << cy << " bf=" << bf << std::endl;
    }


    // ========================================
    // KalmanSplitter: разделение motion на low-freq (VO) и high-freq (stab)
    // ========================================
    KalmanSplitter kalmanSplitter;
    
    // Переменные для логирования
    int logFrameInterval = 30;
    std::ofstream kalmanLogFile;
    kalmanLogFile.open("/home/selbizo/CV/StabAndSLAM/visual_odom/src/OutputResults/KalmanSplitterLog.csv");
    if (kalmanLogFile.is_open()) {
        kalmanLogFile << "frame\tmode\tinnov_norm\tq_trans\tq_rot\tr_trans\tr_rot\thigh_energy\tlow_dx\tlow_dy\tlow_da" << std::endl;
    }


    // -----------------------------------------
    // Run visual odometry
    // -----------------------------------------

    
    
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
    
    // KalmanSplitter результат для текущего кадра
    bool isTurning = false;
    
    // Бенчмарк-лог (вне цикла)
    std::ofstream perfLog;
    perfLog.open("/home/selbizo/CV/StabAndSLAM/visual_odom/src/OutputResults/PerformanceLog.csv");
    if (perfLog.is_open()) {
        perfLog << "frame_id,features_size,matching_stab_ms,matching_ms,triangulate_ms,tracking_ms,total_ms,fps\n";
    }
    
    // Детальный бенчмарк-лог для отладки замедлений
    std::ofstream benchDetailLog;
    benchDetailLog.open("/home/selbizo/CV/StabAndSLAM/visual_odom/src/OutputResults/BenchDetailLog.csv");
    if (benchDetailLog.is_open()) {
        benchDetailLog << "frame_id,phase,duration_ms,cumulative_ms\n";
    }
    
    // Счётчик кадров для логов
    int frameCount = 0;
    
    // Макрос для логирования таймингов
    auto LOG_TIMING = [&](const char* phase, double ms) {
        if (benchDetailLog.is_open()) {
            benchDetailLog << frameCount << "," << phase << "," 
                          << std::fixed << std::setprecision(2) << ms << ",0\n";
        }
    };
    
    for (int frame_id = init_frame_id+1; frame_id < 50000; frame_id+=frame_skip)
    {
        clock_t frame_start = clock();
        double frame_cumulative_ms = 0.0;
        
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
            loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id%10000+1, filepath);  //%1+1
            loadImageRight(imageRight_t1_color, imageRight_t1, frame_id%10000+1, filepath);   
        }
        
        double t_load = 1000.0 * (double)(clock() - frame_start) / CLOCKS_PER_SEC;
        frame_cumulative_ms += t_load;
        LOG_TIMING("load_images", t_load);

        if (frame_id < 80 && frame_skip < 0)
            frame_skip = 1;
        if (frame_id > 13000 && frame_skip > 0)
            frame_skip = -1;
      
        if (true)
        {
            noiseIn.dx = (double)(rng.uniform(-MaxShake, MaxShake))*0.1 + MaxShake*sin(frame_id*DEG_TO_RAD*40.0);
            noiseIn.dy = (double)(rng.uniform(-MaxShake, MaxShake))*0.1 + MaxShake*cos(frame_id*DEG_TO_RAD*34.0);
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
        }

        // ========================================
        // VideoStabilizationPipeline — высокоуровневый интерфейс
        // ========================================
        StabilizationResult stabResult;
        KalmanMotionComponents kalmanResult;
        bool isTurning = false;
        
        if (usePipeline) {
            // Высокоуровневый вызов — один метод вместо ~160 строк
            stabResult = stabPipeline.processFrame(imageLeft_t1, imageRight_t1);
            
            // Получаем kalmanResult из pipeline
            auto debugInfo = stabPipeline.getDebugInfo();
            kalmanResult.low_dx = debugInfo.low_dx;
            kalmanResult.low_dy = debugInfo.low_dy;
            kalmanResult.low_da = debugInfo.low_da;
            kalmanResult.high_dx = debugInfo.high_dx;
            kalmanResult.high_dy = debugInfo.high_dy;
            kalmanResult.high_da = debugInfo.high_da;
            kalmanResult.mode = debugInfo.mode;
            kalmanResult.confidence = 1.0 - debugInfo.innovationNorm / 50.0;
            kalmanResult.confidence = std::max(0.0, std::min(1.0, kalmanResult.confidence));
            
            // Копируем результаты
            stabResult.stabilizedLeft.copyTo(imageLeft_stab_t1);
            stabResult.stabilizedRight.copyTo(imageRight_stab_t1);
            stabResult.TStabLeft.copyTo(TStabLeft);
            stabResult.TStabInvLeft.copyTo(TStabInvLeft);
            
            // Обновляем transforms для обратной совместимости
            std::vector<TransformParam> p_transforms(3, TransformParam(0, 0, 0));
            std::vector<TransformParam> p_movementKalman(3, TransformParam(0, 0, 0));
            p_transforms[0] = TransformParam(stabResult.movementHigh.dx, stabResult.movementHigh.dy, stabResult.movementHigh.da);
            p_transforms[1] = TransformParam(debugInfo.meas_dx, debugInfo.meas_dy, debugInfo.meas_da);
            p_movementKalman[1] = TransformParam(debugInfo.low_dx, debugInfo.low_dy, debugInfo.low_da);
            p_movementKalman[2] = TransformParam(debugInfo.high_dx, debugInfo.high_dy, debugInfo.high_da);
            
            // Используем локальные переменные для showServiceInfo
            double p_gain = stabPipeline.getGain();
            double p_tauStab = stabPipeline.getTauStab();
            
            gain = p_gain;
            tauStab = p_tauStab;
            
            isTurning = (kalmanResult.mode == KalmanMotionComponents::MotionMode::TURNING);
        }
        // Визуализация результатов стабилизации
        // ========================================
        {
            double shakeEnergy = kalmanResult.high_dx * kalmanResult.high_dx + kalmanResult.high_dy * kalmanResult.high_dy;
            double totalEnergy = kalmanResult.low_dx * kalmanResult.low_dx + kalmanResult.low_dy * kalmanResult.low_dy + shakeEnergy;
            double stabPercent = totalEnergy > 0.01 ? (1.0 - shakeEnergy / totalEnergy) * 100.0 : 0.0;
            stabPercent = std::max(0.0, std::min(100.0, stabPercent));
            
            const char* modeStr[] = {"UNKNOWN", "TURNING", "STRAIGHT", "SHAKE_ONLY"};
            
            char infoLine[256];
            snprintf(infoLine, sizeof(infoLine), 
                "Stab: %5.1f%% | Mode: %s | Shake: %5.1f px | Low: %5.1f px",
                stabPercent, modeStr[static_cast<int>(kalmanResult.mode)],
                std::sqrt(shakeEnergy), std::sqrt(kalmanResult.low_dx * kalmanResult.low_dx + kalmanResult.low_dy * kalmanResult.low_dy));
            
            cv::putText(imageLeft_t1_color, infoLine, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            // Индикатор режима цветом
            cv::Scalar modeColor = cv::Scalar(0, 255, 0);  // зелёный = STRAIGHT
            if (kalmanResult.mode == KalmanMotionComponents::MotionMode::TURNING) {
                modeColor = cv::Scalar(0, 0, 255);  // красный = TURNING
            } else if (kalmanResult.mode == KalmanMotionComponents::MotionMode::SHAKE_ONLY) {
                modeColor = cv::Scalar(255, 255, 0);  // жёлтый = SHAKE_ONLY
            }
            cv::rectangle(imageLeft_t1_color, cv::Rect(10, 5, 250, 25), modeColor, -1);
        }
        
        // ========================================
        // Адаптивный выбор кадров для VO
        // ========================================
        bool useStabForVO = !isTurning && kalmanResult.confidence > 0.3;
        
        // Для pipeline — imageLeft_stab_t1 уже установлен выше
        if (usePipeline) {
            // Вызываем showServiceInfo с pipeline-данными
            std::vector<TransformParam> p_transforms(3, TransformParam(0, 0, 0));
            std::vector<TransformParam> p_movementKalman(3, TransformParam(0, 0, 0));
            p_transforms[0] = TransformParam(stabResult.movementHigh.dx, stabResult.movementHigh.dy, stabResult.movementHigh.da);
            p_transforms[1] = TransformParam(kalmanResult.high_dx, kalmanResult.high_dy, kalmanResult.high_da);
            p_movementKalman[1] = TransformParam(kalmanResult.low_dx, kalmanResult.low_dy, kalmanResult.low_da);
            p_movementKalman[2] = TransformParam(kalmanResult.high_dx, kalmanResult.high_dy, kalmanResult.high_da);
            
            double p_gain = std::max(0.7, std::min(1.0, 1.0 - kalmanResult.innovationNorm / 50.0));
            double p_tauStab = 5.0;
            
            showServiceInfoSmall(imageLeft_t1_color, 1.0, 1.0, true, true, true, 
                p_transforms, p_movementKalman, p_tauStab, p_gain, framePart, 
                static_cast<int>(stabResult.featuresFound ? 500 : 0), 1, 1.0, 1.0, 1.0, 
                a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab, fontFace, fontScale, colorGREEN);
        }
    
    clock_t t_match_start = clock();
    {
        t_match_start = clock();
        matchingFeatures(
            useStabForVO ? imageLeft_stab_t0 : imageLeft_t0,
            useStabForVO ? imageRight_stab_t0 : imageRight_t0,
            useStabForVO ? imageLeft_stab_t1 : imageLeft_t1,
            useStabForVO ? imageRight_stab_t1 : imageRight_t1,
            currentVOFeatures,
            pointsLeft_t0, 
            pointsRight_t0, 
            pointsLeft_t1, 
            pointsRight_t1,
            1.0);
        double t_match = 1000.0 * (double)(clock() - t_match_start) / CLOCKS_PER_SEC;
        LOG_TIMING("matching_features", t_match);
    }
    double matching_ms = 1000.0 * (double)(clock() - t_match_start) / CLOCKS_PER_SEC;
    
    // Инициализация таймингов для бенчмарка
    double triangulate_ms = 0.0;
    double tracking_ms = 0.0;
    double matchingStab_ms = 0.0;
    
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
        {
            clock_t t_triang_start = clock();
            points3D_t0.release();
            points4D_t0.release();
            cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
            cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);
            clock_t t_triang_end = clock();
            triangulate_ms = 1000.0 * (double)(t_triang_end - t_triang_start) / CLOCKS_PER_SEC;
            LOG_TIMING("triangulate", triangulate_ms);
        }

        // ---------------------
        // Tracking transformation
        // ---------------------
        {
            clock_t tic_gpu = clock();
            trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, 
                           points3D_t0, rotation, translation, frame_skip, false);
            clock_t toc_gpu = clock();
            tracking_ms = 1000.0 * (double)(toc_gpu - tic_gpu) / CLOCKS_PER_SEC;
            LOG_TIMING("tracking", tracking_ms);
        }
    
        // Сохраняем валидные параметры движения
        last_valid_rotation = rotation.clone();
        last_valid_translation = translation.clone();
        
        // Сохраняем тайминги для бенчмарка
        matching_ms = 1000.0 * (double)(clock() - t_match_start) / CLOCKS_PER_SEC;
    }

    displayTracking(imageLeft_stab_t1, pointsLeft_t0, pointsLeft_t1, "imageLeft_stab_t1"); //show input image
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
        
        // Итоговый тайминг кадра
        double frame_total_ms = 1000.0 * (double)(clock() - frame_start) / CLOCKS_PER_SEC;
        frame_cumulative_ms += frame_total_ms;
        frameCount++;
        

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
    
    // Закрытие бенчмарк-лога
    if (perfLog.is_open()) {
        perfLog.close();
        std::cout << "Performance log saved to: /home/selbizo/CV/StabAndSLAM/visual_odom/src/OutputResults/PerformanceLog.csv" << std::endl;
    }
    
    // Закрытие детального бенчмарк-лога
    if (benchDetailLog.is_open()) {
        benchDetailLog.close();
        std::cout << "Detailed benchmark log saved to: /home/selbizo/CV/StabAndSLAM/visual_odom/src/OutputResults/BenchDetailLog.csv" << std::endl;
    }
    
    std::cout << "\n[SUMMARY] Processed " << frameCount << " frames" << std::endl;
    
    return 0;
}