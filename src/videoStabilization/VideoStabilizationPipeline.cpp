// ============================================================
// VideoStabilizationPipeline — реализация
// ============================================================
// Инкапсулирует весь контур стабилизации:
//   matchingFeaturesStab → getBiasAndRotation → iirAdaptiveHighPass
//   → KalmanSplitter → warpAffine → crop/resize
// ============================================================

#include "VideoStabilizationPipeline.h"
#include "stabilizationFunctions.h"
#include "../feature.h"
#include "../visualOdometry.h"
#include "../utils.h"
#include "../evaluate/matrix.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

// ============================================================
// Конструктор
// ============================================================
VideoStabilizationPipeline::VideoStabilizationPipeline()
    : tauStab_(5.0), gain_(0.7), frame_skip_(1)
{
    transforms_.resize(3, TransformParam(0, 0, 0));
    movement_.resize(2, TransformParam(0, 0, 0));
    movementKalman_.resize(3, TransformParam(0, 0, 0));
}

VideoStabilizationPipeline::~VideoStabilizationPipeline() {}

// ============================================================
// Инициализация с параметрами камеры
// ============================================================
void VideoStabilizationPipeline::init(double fx, double fy, double cx, double cy, double bf,
                                      int imageWidth, int imageHeight,
                                      int compression, double framePart)
{
    config_.compression = compression;
    config_.framePart = framePart;
    imgWidth_ = imageWidth;
    imgHeight_ = imageHeight;

    // Камера: P_left / P_right
    projMatrL_ = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
    projMatrR_ = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0, 0., 1., 0.);

    // ROI — центральная область
    roi_ = cv::Rect(
        imageWidth * ((1.0 - framePart) / 2.0),
        imageHeight * ((1.0 - framePart) / 2.0),
        imageWidth * framePart,
        imageHeight * framePart
    );

    c_ = std::sqrt((double)imageWidth * imageWidth + (double)imageHeight * imageHeight);
    atan_ba_ = std::atan2(imageHeight, imageWidth);

    // Инициализация CUDA
    initCUDA();
    initBuffers();

    std::cout << "[VideoStabPipeline] Initialized: " << imageWidth << "x" << imageHeight
              << " ROI:" << roi_ << " compression:" << compression << std::endl;
}

// ============================================================
// Инициализация CUDA-детекторов
// ============================================================
void VideoStabilizationPipeline::initCUDA()
{
    createDetectors(d_features_, d_features_small_, d_pyrLK_sparse_);
}

// ============================================================
// Инициализация GpuMat-буферов
// ============================================================
void VideoStabilizationPipeline::initBuffers()
{
    int a = imgWidth_;
    int b = imgHeight_;
    int comp = config_.compression;
    int cw = a / comp;
    int ch = b / comp;

    gFrameLeft_.create(b, a, CV_8UC1);
    gFrameRight_.create(b, a, CV_8UC1);
    gCompressedLeft_.create(ch, cw, CV_8UC1);
    gCompressedRight_.create(ch, cw, CV_8UC1);
    gGrayLeft_.create(ch, cw, CV_8UC1);
    gGrayRight_.create(ch, cw, CV_8UC1);

    gOldFrameLeft_.create(b, a, CV_8UC1);
    gOldFrameRight_.create(b, a, CV_8UC1);
    gOldGrayLeft_.create(ch, cw, CV_8UC1);
    gOldGrayRight_.create(ch, cw, CV_8UC1);
    gOldCompressedLeft_.create(ch, cw, CV_8UC3);
    gOldCompressedRight_.create(ch, cw, CV_8UC3);

    // p0 buffers для детекции
    gP0Left_.create(1, 0, CV_32FC2);
    gP0Right_.create(1, 0, CV_32FC2);

    // Маски
    Mat maskSearchL = Mat::zeros(cv::Size(cw, ch), CV_8U);
    Mat maskSearchR = Mat::zeros(cv::Size(cw, ch), CV_8U);
    cv::rectangle(maskSearchL, cv::Rect(cw * (1.0 - 0.5) / 2, ch * (1.0 - 0.5) / 2, cw * 0.5, ch * 0.5), Scalar(255), cv::FILLED);
    cv::rectangle(maskSearchR, cv::Rect(cw * (1.0 - 0.5) / 2, ch * (1.0 - 0.5) / 2, cw * 0.5, ch * 0.5), Scalar(255), cv::FILLED);
    gMaskSearchLeft_.upload(maskSearchL);
    gMaskSearchRight_.upload(maskSearchR);

    Mat maskSmallL = Mat::zeros(cv::Size(cw, ch), CV_8U);
    Mat maskSmallR = Mat::zeros(cv::Size(cw, ch), CV_8U);
    cv::rectangle(maskSmallL, cv::Rect(cw * (1.0 - 0.3) / 2, ch * (1.0 - 0.3) / 2, std::max(a, b) * 0.3 / comp, std::max(a, b) * 0.3 / comp), Scalar(255), cv::FILLED);
    cv::rectangle(maskSmallR, cv::Rect(cw * (1.0 - 0.3) / 2, ch * (1.0 - 0.3) / 2, std::max(a, b) * 0.3 / comp, std::max(a, b) * 0.3 / comp), Scalar(255), cv::FILLED);
    gMaskSearchSmallLeft_.upload(maskSmallL);
    gMaskSearchSmallRight_.upload(maskSmallR);

    // Буферы для Kalman
    kalmanSplitter_ = KalmanSplitter();

    // Инициализация результатов
    rotation_stab_ = cv::Mat::eye(3, 3, CV_64F);
    translation_stab_ = cv::Mat::zeros(3, 1, CV_64F);
}

// ============================================================
// processFrame — основной интерфейс
// ============================================================
StabilizationResult VideoStabilizationPipeline::processFrame(const cv::Mat& leftGray, const cv::Mat& rightGray)
{
    auto frameStart = std::chrono::high_resolution_clock::now();
    StabilizationResult result;
    result.enabled = enabled_;

    if (!enabled_) {
        // Zero-copy: просто копируем входные кадры
        leftGray.copyTo(result.stabilizedLeft);
        rightGray.copyTo(result.stabilizedRight);
        result.mode = KalmanMotionComponents::MotionMode::UNKNOWN;
        result.elapsedMs = 0.0;
        return result;
    }

    // ---- Шаг 1: matchingFeaturesStab (feature tracking на стабилизированных кадрах) ----
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Загружаем в GPU
        gFrameLeft_.upload(leftGray);
        gFrameRight_.upload(rightGray);

        // Сжимаем
        cuda::resize(gFrameLeft_, gCompressedLeft_, cv::Size(imgWidth_ / config_.compression, imgHeight_ / config_.compression), 0, 0, cv::INTER_LINEAR);
        cuda::resize(gFrameRight_, gCompressedRight_, cv::Size(imgWidth_ / config_.compression, imgHeight_ / config_.compression), 0, 0, cv::INTER_LINEAR);

        // Уже grayscale — просто копируем
        gCompressedLeft_.copyTo(gGrayLeft_);
        gCompressedRight_.copyTo(gGrayRight_);

        // Загружаем старые кадры для сравнения
        gOldFrameLeft_.upload(leftGray);
        gOldFrameRight_.upload(rightGray);

        cuda::resize(gOldFrameLeft_, gOldCompressedLeft_, cv::Size(imgWidth_ / config_.compression, imgHeight_ / config_.compression), 0, 0, cv::INTER_LINEAR);
        gOldCompressedLeft_.copyTo(gOldGrayLeft_);

        cuda::resize(gOldFrameRight_, gOldCompressedRight_, cv::Size(imgWidth_ / config_.compression, imgHeight_ / config_.compression), 0, 0, cv::INTER_LINEAR);
        gOldCompressedRight_.copyTo(gOldGrayRight_);
    }

    // ---- Шаг 2: Вычисление motion из optical flow ----
    computeStabilizationTransform(leftGray, rightGray);

    // ---- Шаг 3: KalmanSplitter ----
    updateKalmanSplitter();

    // ---- Шаг 4: Вычисление TStabLeft ----
    {
        // Адаптивная стабилизация
        bool isTurning = (kalmanSplitter_.getModeOverride() == KalmanMotionComponents::MotionMode::TURNING) ||
                         (modeOverride_ == KalmanMotionComponents::MotionMode::TURNING);

        double stab_dx = movementKalman_[2].dx;
        double stab_dy = movementKalman_[2].dy;
        double stab_da = isTurning ? 0.0 : movementKalman_[2].da;

        transforms_[0] = TransformParam(-stab_dx, -stab_dy, -stab_da);
        
        // Выделяем память для матриц трансформации
        result.TStabLeft.create(2, 3, CV_64F);
        result.TStabInvLeft.create(2, 3, CV_64F);
        
        transforms_[0].getTransform(result.TStabLeft, imgWidth_, imgHeight_, c_, atan_ba_, config_.framePart);
        transforms_[0].getTransformInvert(result.TStabInvLeft, imgWidth_, imgHeight_, c_, atan_ba_, config_.framePart);
    }

    // ---- Шаг 5: warpAffine + crop + resize ----
    applyWarpAffine(leftGray, rightGray, result);

    // ---- Шаг 6: Заполнение метаданных ----
    {
        result.mode = kalmanSplitter_.getMode();
        double shakeEnergy = movementKalman_[2].dx * movementKalman_[2].dx + movementKalman_[2].dy * movementKalman_[2].dy;
        double lowEnergy = movementKalman_[1].dx * movementKalman_[1].dx + movementKalman_[1].dy * movementKalman_[1].dy;
        result.shakeEnergy = shakeEnergy;
        result.lowEnergy = lowEnergy;
        double totalEnergy = lowEnergy + shakeEnergy;
        result.stabPercent = totalEnergy > 0.01 ? (1.0 - shakeEnergy / totalEnergy) * 100.0 : 0.0;
        result.stabPercent = std::max(0.0, std::min(100.0, result.stabPercent));

        result.movementHigh = movementKalman_[2];
        result.movementLow = movementKalman_[1];
        result.featuresFound = (p0Left_.size() >= 15);
    }

    auto frameEnd = std::chrono::high_resolution_clock::now();
    result.elapsedMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

    return result;
}

// ============================================================
// computeStabilizationTransform — matching + bias estimation
// ============================================================
void VideoStabilizationPipeline::computeStabilizationTransform(const cv::Mat& leftGray, const cv::Mat& rightGray)
{
    // Упрощённый подход: используем optical flow между gOldGray и gGray
    // для оценки движения между кадрами

    // Sparse optical flow
    std::vector<cv::Point2f> p0, p1;
    std::vector<uchar> status;
    std::vector<float> err;

    // Если есть старые точки — используем LK flow
    if (!p0Left_.empty()) {
        std::vector<float> err;

        cv::Mat oldGrayCPU, grayCPU;
        gOldGrayLeft_.download(oldGrayCPU);
        gGrayLeft_.download(grayCPU);
        
        cv::calcOpticalFlowPyrLK(oldGrayCPU, grayCPU, p0Left_, p1Left_, statusLeft_, err,
                                  cv::Size(config_.winSize, config_.winSize), config_.maxLevel,
                                  cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, config_.iters, 0.01));

        // Фильтрация
        int n = statusLeft_.size();
        int goodCount = 0;
        for (int i = 0; i < n; i++) {
            if (statusLeft_[i] != 0 && p1Left_[i].x >= 0 && p1Left_[i].y >= 0) {
                goodCount++;
            }
        }

        if (goodCount >= 15) {
            // Успешный трекинг
            cv::Mat tempT;
            getBiasAndRotation(p0Left_, p1Left_, dLeft_, meanP0Left_, transforms_, tempT, config_.compression);
        } else {
            // Мало точек — детектируем заново
            p0Left_.clear();
            p1Left_.clear();
            statusLeft_.clear();

            d_features_->detect(gGrayLeft_, gP0Left_, gMaskSearchLeft_);
            if (!gP0Left_.empty()) {
                gP0Left_.download(p0Left_);
            }

            // Сброс transforms
            transforms_[1] = TransformParam(0, 0, 0);
        }
    } else {
        // Первая инициализация — детектируем точки
        d_features_->detect(gGrayLeft_, gP0Left_, gMaskSearchLeft_);
        if (!gP0Left_.empty()) {
            gP0Left_.download(p0Left_);
            p1Left_ = p0Left_; // для первого кадра
            statusLeft_.assign(p0Left_.size(), 1);
            transforms_[1] = TransformParam(0, 0, 0);
        }
    }

    // Обновляем "старые" буферы
    gOldGrayLeft_.copyTo(gGrayLeft_);
}

// ============================================================
// applyWarpAffine — GPU warp + crop + resize
// ============================================================
void VideoStabilizationPipeline::applyWarpAffine(const cv::Mat& leftGray, const cv::Mat& rightGray, StabilizationResult& result)
{
    int a = imgWidth_;
    int b = imgHeight_;

    // Загружаем в GPU
    cuda::GpuMat gLeft(b, a, CV_8UC1), gRight(b, a, CV_8UC1);
    gLeft.upload(leftGray);
    gRight.upload(rightGray);

    // Warp
    cuda::GpuMat gStabLeft, gStabRight;
    cuda::warpAffine(gLeft, gStabLeft, result.TStabLeft, cv::Size(a, b));
    cuda::warpAffine(gRight, gStabRight, result.TStabLeft, cv::Size(a, b));

    // Crop ROI
    cuda::GpuMat gCropLeft = gStabLeft(roi_);
    cuda::GpuMat gCropRight = gStabRight(roi_);

    // Resize back to original
    cuda::GpuMat gResizedLeft(a, b, CV_8UC1), gResizedRight(a, b, CV_8UC1);
    cv::cuda::resize(gCropLeft, gResizedLeft, cv::Size(a, b), 0, 0, cv::INTER_NEAREST);
    cv::cuda::resize(gCropRight, gResizedRight, cv::Size(a, b), 0, 0, cv::INTER_NEAREST);

    // Download
    gResizedLeft.download(result.stabilizedLeft);
    gResizedRight.download(result.stabilizedRight);
    gCropLeft.download(result.stabilizedLeftCrop);
    gCropRight.download(result.stabilizedRightCrop);
}

// ============================================================
// adaptGain — адаптация gain
// ============================================================
void VideoStabilizationPipeline::adaptGain()
{
    if (gain_ < 1.0) {
        gain_ *= 1.05;
        gain_ += 0.01;
    }
    if (gain_ > 1.0) {
        gain_ = 1.0;
    }
}

// ============================================================
// updateKalmanSplitter — обновление KalmanSplitter
// ============================================================
void VideoStabilizationPipeline::updateKalmanSplitter()
{
    auto kalmanResult = kalmanSplitter_.update(
        transforms_[1].dx,
        transforms_[1].dy,
        transforms_[1].da
    );

    // Применяем override режима
    if (modeOverride_ != KalmanMotionComponents::MotionMode::UNKNOWN) {
        kalmanResult.mode = modeOverride_;
    }

    // Заполняем movementKalman
    movementKalman_[1].dx = kalmanResult.low_dx;
    movementKalman_[1].dy = kalmanResult.low_dy;
    movementKalman_[1].da = kalmanResult.low_da;

    movementKalman_[2].dx = kalmanResult.high_dx;
    movementKalman_[2].dy = kalmanResult.high_dy;
    movementKalman_[2].da = kalmanResult.high_da;

    adaptGain();
}

// ============================================================
// getDebugInfo
// ============================================================
KalmanSplitter::DebugInfo VideoStabilizationPipeline::getDebugInfo() const
{
    return kalmanSplitter_.getDebugInfo();
}

// ============================================================
// setModeOverride
// ============================================================
void VideoStabilizationPipeline::setModeOverride(KalmanMotionComponents::MotionMode mode)
{
    modeOverride_ = mode;
}

// ============================================================
// reset
// ============================================================
void VideoStabilizationPipeline::reset()
{
    p0Left_.clear();
    p1Left_.clear();
    statusLeft_.clear();
    kalmanSplitter_.reset();
    modeOverride_ = KalmanMotionComponents::MotionMode::UNKNOWN;
    gain_ = config_.gainInit;
    transforms_ = std::vector<TransformParam>(3, TransformParam(0, 0, 0));
    movementKalman_ = std::vector<TransformParam>(3, TransformParam(0, 0, 0));
}
