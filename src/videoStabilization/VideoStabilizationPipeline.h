#pragma once

// ============================================================
// VideoStabilizationPipeline — высокоуровневый контур стабилизации
// ============================================================
// Инкапсулирует весь pipeline:
//   matchingFeaturesStab → getBiasAndRotation → iirAdaptiveHighPass
//   → KalmanSplitter → warpAffine → crop/resize
// ============================================================

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaoptflow.hpp"

#include <vector>
#include <string>
#include <chrono>

#include "basicStructs.h"
#include "kalmanSplitter.h"

// ============================================================
// Результат стабилизации одного кадра
// ============================================================
struct StabilizationResult {
    cv::Mat stabilizedLeft;
    cv::Mat stabilizedRight;
    cv::Mat stabilizedLeftCrop;   // crop ROI после стабилизации
    cv::Mat stabilizedRightCrop;

    KalmanMotionComponents::MotionMode mode = KalmanMotionComponents::MotionMode::UNKNOWN;
    double shakeEnergy = 0.0;
    double lowEnergy = 0.0;
    double stabPercent = 0.0;
    bool enabled = false;
    bool featuresFound = false;
    double elapsedMs = 0.0;

    // Для обратной совместимости
    cv::Mat TStabLeft;
    cv::Mat TStabInvLeft;
    TransformParam movementHigh;
    TransformParam movementLow;
};

// ============================================================
// Конфигурация пайплайна
// ============================================================
struct VideoStabConfig {
    int compression = 1;
    double framePart = 0.94;
    double tauStab = 5.0;
    double maxShake = 2.0;
    double gainInit = 0.7;
    double gainStep = 0.01;
    double gainMax = 1.0;
    int maxCorners = 500;
    double qualityLevel = 0.01;
    double minDistance = 1.0;
    int winSize = 31;
    int maxLevel = 2;
    int iters = 30;
    bool useCUDA = true;

    VideoStabConfig() {}
};

class VideoStabilizationPipeline {
public:
    VideoStabilizationPipeline();
    ~VideoStabilizationPipeline();

    /**
     * Инициализация с параметрами камеры
     * @param fx, fy, cx, cy  фокусное расстояние и оптический центр
     * @param bf               базлайн (для стерео)
     * @param imageWidth, imageHeight  размер изображений
     * @param compression      коэффициент сжатия
     * @param framePart        доля кадра, используемая для трекинга
     */
    void init(double fx, double fy, double cx, double cy, double bf,
              int imageWidth, int imageHeight,
              int compression = 1, double framePart = 0.94);

    /**
     * Обработать пару кадров (левый + правый, серые)
     * @return StabilizationResult
     */
    StabilizationResult processFrame(const cv::Mat& leftGray, const cv::Mat& rightGray);

    /** Включить/выключить стабилизацию */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    /** Ручной override режима KalmanSplitter */
    void setModeOverride(KalmanMotionComponents::MotionMode mode);
    void clearModeOverride() { modeOverride_ = KalmanMotionComponents::MotionMode::UNKNOWN; }

    /** Получить отладочную информацию */
    KalmanSplitter::DebugInfo getDebugInfo() const;

    /** Сбросить состояние (при потере трекинга) */
    void reset();

    /** Получить текущий frame_skip */
    int getFrameSkip() const { return frame_skip_; }
    void setFrameSkip(int val) { frame_skip_ = val; }

    /** Получить projMatrL/R */
    cv::Mat getProjMatL() const { return projMatrL_; }
    cv::Mat getProjMatR() const { return projMatrR_; }

    /** Получить ROI */
    cv::Rect getROI() const { return roi_; }

    /** Получить параметры для визуализации */
    double getGain() const { return gain_; }
    double getTauStab() const { return tauStab_; }
    const std::vector<TransformParam>& getMovementKalman() const { return movementKalman_; }
    const std::vector<TransformParam>& getMovement() const { return movement_; }
    const std::vector<TransformParam>& getTransforms() const { return transforms_; }

private:
    // ---- Конфигурация ----
    VideoStabConfig config_;
    bool enabled_ = true;
    KalmanMotionComponents::MotionMode modeOverride_ = KalmanMotionComponents::MotionMode::UNKNOWN;

    // ---- Камера ----
    cv::Mat projMatrL_, projMatrR_;
    int imgWidth_, imgHeight_;
    cv::Rect roi_;
    double c_, atan_ba_;

    // ---- KalmanSplitter ----
    KalmanSplitter kalmanSplitter_;

    // ---- Параметры фильтра ----
    double tauStab_;
    double gain_;
    int frame_skip_;

    // ---- Буферы transforms/movement ----
    std::vector<TransformParam> transforms_;  // [0]=stab, [1]=raw, [2]=high
    std::vector<TransformParam> movement_;
    std::vector<TransformParam> movementKalman_;

    // ---- CUDA-ресурсы ----
    cv::Ptr<cv::cuda::CornersDetector> d_features_;
    cv::Ptr<cv::cuda::CornersDetector> d_features_small_;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse_;

    // ---- GpuMat-буферы ----
    cv::cuda::GpuMat gFrameLeft_, gFrameRight_;
    cv::cuda::GpuMat gCompressedLeft_, gCompressedRight_;
    cv::cuda::GpuMat gGrayLeft_, gGrayRight_;
    cv::cuda::GpuMat gOldFrameLeft_, gOldFrameRight_;
    cv::cuda::GpuMat gOldGrayLeft_, gOldGrayRight_;
    cv::cuda::GpuMat gOldCompressedLeft_, gOldCompressedRight_;
    cv::cuda::GpuMat gP0Left_, gP0Right_;
    cv::cuda::GpuMat gMaskSearchLeft_, gMaskSearchRight_;
    cv::cuda::GpuMat gMaskSearchSmallLeft_, gMaskSearchSmallRight_;

    // ---- Point buffers ----
    std::vector<cv::Point2f> p0Left_, p1Left_, p0Right_, p1Right_;
    std::vector<uchar> statusLeft_, statusRight_;

    // ---- Результаты ----
    cv::Mat rotation_stab_, translation_stab_;
    cv::Mat points3D_t0_stab_, points4D_t0_stab_;
    cv::Point2f dLeft_, dRight_, meanP0Left_, meanP0Right_;

    // ---- Тайминг ----
    std::chrono::high_resolution_clock::time_point startTime_;

    // ---- Внутренние методы ----
    void initCUDA();
    void initBuffers();
    void computeStabilizationTransform(const cv::Mat& leftGray, const cv::Mat& rightGray);
    void applyWarpAffine(const cv::Mat& leftGray, const cv::Mat& rightGray, StabilizationResult& result);
    void adaptGain();
    void updateKalmanSplitter();
};
