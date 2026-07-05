#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// ============================================================
// KalmanSplitter — разделение motion на low-freq (VO) и high-freq (stab)
// ============================================================
// Два независимых фильтра:
//   1. TranslationFilter:  state = [px, py, vx, vy, ax, ay]ᵀ
//                          meas = [dx, dy] (из optical flow)
//   2. RotationFilter:     state = [yaw, vyaw, ayaw]ᵀ
//                          meas = [da] (из optical flow)
//
// Адаптивная настройка Q, R на основе innovation (остатка)
// Детекция режимов: поворот / прямое движение / тряска
// ============================================================

struct KalmanMotionComponents {
    // Low-frequency (предсказание Калмана) — идёт в VO
    double low_dx = 0.0;
    double low_dy = 0.0;
    double low_da = 0.0;

    // High-frequency (остаток = measurement - prediction) — идёт в stab
    double high_dx = 0.0;
    double high_dy = 0.0;
    double high_da = 0.0;

    // Скорость поворота из Калмана (vyaw) — для детекции поворота
    double vyaw = 0.0;

    // Детекция режимов
    enum class MotionMode {
        TURNING,        // автомобиль поворачивает — stab НЕ компенсирует поворот
        STRAIGHT,       // прямое движение — stab компенсирует только тряску
        SHAKE_ONLY,     // только тряска, нет направленного движения
        UNKNOWN
    };
    MotionMode mode = MotionMode::UNKNOWN;

    // Доверие к фильтрации (0..1)
    double confidence = 0.0;

    // Innovation norm для мониторинга
    double innovationNorm = 0.0;
};

class KalmanSplitter {
public:
    KalmanSplitter();
    ~KalmanSplitter();

    /**
     * Обновить фильтр и получить разделённые компоненты motion
     * @param meas_dx  измеренное смещение по X (из optical flow / transforms)
     * @param meas_dy  измеренное смещение по Y
     * @param meas_da  измеренный поворот (из optical flow / transforms)
     * @return разделённые компоненты
     */
    KalmanMotionComponents update(double meas_dx, double meas_dy, double meas_da);

    /** Сбросить фильтр (при потере трекинга) */
    void reset();

    /** Получить текущее состояние для логирования */
    struct DebugInfo {
        double meas_dx, meas_dy, meas_da;
        double low_dx, low_dy, low_da;
        double high_dx, high_dy, high_da;
        double innovationNorm;
        double qTranslation, qRotation;
        double rTranslation, rRotation;
        KalmanMotionComponents::MotionMode mode;
    };
    DebugInfo getDebugInfo() const;

private:
    // ---- Translation filter ----
    // state: [px, py, vx, vy, ax, ay] (6x1)
    cv::Mat A_trans_;
    cv::Mat P_trans_;
    cv::Mat Q_trans_;
    cv::Mat R_trans_;
    double qTransNominal_;
    double rTransNominal_;
    double qTransAdaptive_;
    double rTransAdaptive_;
    cv::Mat xTrans_;       // текущее состояние
    cv::Mat innovTrans_;   // innovation (остаток)
    double innovNormTrans_;

    // ---- Rotation filter ----
    // state: [yaw, vyaw, ayaw] (3x1)
    cv::Mat A_rot_;
    cv::Mat P_rot_;
    cv::Mat Q_rot_;
    cv::Mat R_rot_;
    double qRotNominal_;
    double rRotNominal_;
    double qRotAdaptive_;
    double rRotAdaptive_;
    cv::Mat xRot_;
    cv::Mat innovRot_;
    double innovNormRot_;

    // ---- Motion mode detection ----
    KalmanMotionComponents::MotionMode detectMotionMode(double low_dx, double low_dy, double low_da,
                                                         double high_dx, double high_dy, double high_da,
                                                         double vyaw) const;

    // ---- Adaptive Q, R ----
    void adaptQandR_translation(double innovNorm, double dt = 1.0/30.0);
    void adaptQandR_rotation(double innovNorm, double dt = 1.0/30.0);

    // ---- Helpers ----
    void initTranslationFilter();
    void initRotationFilter();

    // Параметры адаптации
    double alphaQ_ = 0.01;   // коэффициент адаптации Q
    double alphaR_ = 0.005;  // коэффициент адаптации R
    double innovThresholdLow_ = 2.0;   // порог для "хорошего" измерения
    double innovThresholdHigh_ = 15.0; // порог для "выброса" (тряска)

    // Параметры детекции режимов
    double turnThreshold_ = 0.015;     // рад/кадр ≈ 0.86°/кадр при 30fps
    double straightThreshold_ = 0.5;    // пикселей/кадр для прямого движения
    double shakeThreshold_ = 3.0;       // пикселей std для тряски
    double historySize_ = 30;           // окно для статистики
    vector<double> highEnergyHistory_;  // история high-freq энергии
};
