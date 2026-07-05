#include "kalmanSplitter.h"
#include <algorithm>
#include <cmath>
#include <numeric>

// ============================================================
// Конструктор / Деструктор
// ============================================================

KalmanSplitter::KalmanSplitter() {
    initTranslationFilter();
    initRotationFilter();
}

KalmanSplitter::~KalmanSplitter() {}

// ============================================================
// Инициализация Translation фильтра
// state = [px, py, vx, vy, ax, ay]^T (6x1)
// meas = [dx, dy] (2x1) — измеряем смещение за кадр
// ============================================================
void KalmanSplitter::initTranslationFilter() {
    double dt = 1.0 / 30.0;
    double dt2 = dt * dt;

    // State transition matrix (постоянное ускорение)
    A_trans_ = (cv::Mat_<double>(6, 6) <<
        1, 0, dt,  0,  dt2/2,    0,
        0, 1,  0, dt,     0, dt2/2,
        0, 0,  1,  0,     dt,    0,
        0, 0,  0,  1,     0,   dt,
        0, 0,  0,  0,     1,    0,
        0, 0,  0,  0,     0,    1);

    // Measurement matrix: y = H * x, где y = [dx, dy]
    cv::Mat H_trans = (cv::Mat_<double>(2, 6) <<
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0);

    // Начальное состояние: нулевое смещение
    xTrans_ = cv::Mat::zeros(6, 1, CV_64F);

    // Начальная ковариация
    P_trans_ = cv::Mat::eye(6, 6, CV_64F) * 10.0;

    // Process noise
    qTransNominal_ = 1.0;
    qTransAdaptive_ = qTransNominal_;
    Q_trans_ = cv::Mat::eye(6, 6, CV_64F) * qTransAdaptive_;

    // Measurement noise
    rTransNominal_ = 100.0;
    rTransAdaptive_ = rTransNominal_;
    R_trans_ = cv::Mat::eye(2, 2, CV_64F) * rTransAdaptive_;

    innovTrans_ = cv::Mat::zeros(2, 1, CV_64F);
    innovNormTrans_ = 0.0;
}

// ============================================================
// Инициализация Rotation фильтра
// state = [yaw, vyaw, ayaw]^T (3x1)
// meas = [da] (1x1)
// ============================================================
void KalmanSplitter::initRotationFilter() {
    double dt = 1.0 / 30.0;
    double dt2 = dt * dt;

    // State transition (постоянное угловое ускорение)
    A_rot_ = (cv::Mat_<double>(3, 3) <<
        1, dt, dt2/2,
        0,  1,   dt,
        0,  0,    1);

    // Measurement: измеряем da напрямую
    cv::Mat H_rot = (cv::Mat_<double>(1, 3) << 1, 0, 0);

    // Начальное состояние
    xRot_ = cv::Mat::zeros(3, 1, CV_64F);

    // Начальная ковариация
    P_rot_ = cv::Mat::eye(3, 3, CV_64F) * 5.0;

    // Process noise
    qRotNominal_ = 0.5;
    qRotAdaptive_ = qRotNominal_;
    Q_rot_ = cv::Mat::eye(3, 3, CV_64F) * qRotAdaptive_;

    // Measurement noise
    rRotNominal_ = 50.0;
    rRotAdaptive_ = rRotNominal_;
    R_rot_ = cv::Mat::eye(1, 1, CV_64F) * rRotAdaptive_;

    innovRot_ = cv::Mat::zeros(1, 1, CV_64F);
    innovNormRot_ = 0.0;
}

// ============================================================
// Обновление фильтра — основной метод
// ============================================================
KalmanMotionComponents KalmanSplitter::update(double meas_dx, double meas_dy, double meas_da) {
    KalmanMotionComponents result;

    // ---- Шаг 1: Предсказание (prediction) ----
    cv::Mat predTrans = A_trans_ * xTrans_;
    cv::Mat P_predTrans = A_trans_ * P_trans_ * A_trans_.t() + Q_trans_;

    cv::Mat predRot = A_rot_ * xRot_;
    cv::Mat P_predRot = A_rot_ * P_rot_ * A_rot_.t() + Q_rot_;

    // ---- Шаг 2: Innovation (остаток = measurement - prediction) ----
    cv::Mat measTrans = (cv::Mat_<double>(2, 1) << meas_dx, meas_dy);
    cv::Mat H_trans = (cv::Mat_<double>(2, 6) <<
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0);
    innovTrans_ = measTrans - H_trans * predTrans;

    cv::Mat measRot = (cv::Mat_<double>(1, 1) << meas_da);
    cv::Mat H_rot = (cv::Mat_<double>(1, 3) << 1, 0, 0);
    innovRot_ = measRot - H_rot * predRot;

    innovNormTrans_ = std::sqrt(innovTrans_.dot(innovTrans_));
    innovNormRot_ = std::sqrt(innovRot_.dot(innovRot_));

    // ---- Шаг 3: Kalman Gain и Update ----
    // Translation: K = P_pred * H^T * (H * P_pred * H^T + R)^-1
    cv::Mat S_trans = H_trans * P_predTrans * H_trans.t() + R_trans_;
    cv::Mat S_trans_inv;
    cv::invert(S_trans, S_trans_inv);
    cv::Mat K_trans = P_predTrans * H_trans.t() * S_trans_inv;
    xTrans_ = predTrans + K_trans * innovTrans_;
    cv::Mat I_KH = cv::Mat::eye(6, 6, CV_64F) - K_trans * H_trans;
    P_trans_ = I_KH * P_predTrans * I_KH.t() + K_trans * R_trans_ * K_trans.t();

    // Rotation: K = P_pred * H^T * (H * P_pred * H^T + R)^-1
    cv::Mat S_rot = H_rot * P_predRot * H_rot.t() + R_rot_;
    cv::Mat S_rot_inv;
    cv::invert(S_rot, S_rot_inv);
    cv::Mat K_rot = P_predRot * H_rot.t() * S_rot_inv;
    xRot_ = predRot + K_rot * innovRot_;
    cv::Mat I_KH_rot = cv::Mat::eye(3, 3, CV_64F) - K_rot * H_rot;
    P_rot_ = I_KH_rot * P_predRot * I_KH_rot.t() + K_rot * R_rot_ * K_rot.t();

    // ---- Шаг 4: Адаптивная настройка Q, R ----
    adaptQandR_translation(innovNormTrans_);
    adaptQandR_rotation(innovNormRot_);

    // ---- Шаг 5: Разделение на low-freq и high-freq ----
    result.low_dx = predTrans.at<double>(0, 0);
    result.low_dy = predTrans.at<double>(1, 0);
    result.low_da = predRot.at<double>(0, 0);

    result.high_dx = innovTrans_.at<double>(0, 0);
    result.high_dy = innovTrans_.at<double>(1, 0);
    result.high_da = innovRot_.at<double>(0, 0);

    // vyaw — скорость поворота из Калмана (для детекции поворота)
    result.vyaw = predRot.at<double>(1, 0);

    // ---- Шаг 6: Детекция режимов ----
    result.innovationNorm = innovNormTrans_;
    result.mode = detectMotionMode(result.low_dx, result.low_dy, result.low_da,
                                    result.high_dx, result.high_dy, result.high_da,
                                    result.vyaw);

    result.confidence = std::max(0.0, std::min(1.0, 1.0 - innovNormTrans_ / 50.0));

    // ---- Шаг 7: История для детекции тряски ----
    double highEnergy = result.high_dx * result.high_dx + result.high_dy * result.high_dy;
    highEnergyHistory_.push_back(highEnergy);
    if (highEnergyHistory_.size() > static_cast<size_t>(historySize_)) {
        highEnergyHistory_.erase(highEnergyHistory_.begin());
    }

    return result;
}

// ============================================================
// Детекция режимов движения
// ============================================================
KalmanMotionComponents::MotionMode KalmanSplitter::detectMotionMode(
    double low_dx, double low_dy, double low_da,
    double high_dx, double high_dy, double high_da,
    double vyaw) const {

    double lowSpeed = std::sqrt(low_dx * low_dx + low_dy * low_dy);
    double highEnergy = high_dx * high_dx + high_dy * high_dy;

    // Используем vyaw (скорость поворота из Калмана) для детекции поворота
    // Порог: ~0.01 рад/кадр ≈ 0.57°/кадр при 30fps
    double turnThresholdRad = 0.01;  // рад/кадр
    
    // Если есть значительная скорость поворота — режим "поворот"
    if (std::abs(vyaw) > turnThresholdRad) {
        return KalmanMotionComponents::MotionMode::TURNING;
    }

    // Если high-freq энергия велика — режим "тряска"
    if (highEnergy > shakeThreshold_ * shakeThreshold_) {
        return KalmanMotionComponents::MotionMode::STRAIGHT;
    }

    // Если low-freq скорость мала — возможно "тряска на месте"
    if (lowSpeed < 0.1) {
        return KalmanMotionComponents::MotionMode::SHAKE_ONLY;
    }

    return KalmanMotionComponents::MotionMode::STRAIGHT;
}

// ============================================================
// Адаптивная настройка Q (process noise)
// ============================================================
void KalmanSplitter::adaptQandR_translation(double innovNorm, double dt) {
    if (innovNorm > innovThresholdHigh_) {
        qTransAdaptive_ = qTransNominal_ * 3.0;
        rTransAdaptive_ = rTransNominal_ * 2.0;
    } else if (innovNorm > innovThresholdLow_) {
        qTransAdaptive_ = qTransNominal_ * 1.5;
        rTransAdaptive_ = rTransNominal_ * 1.2;
    } else {
        qTransAdaptive_ = qTransNominal_;
        rTransAdaptive_ = rTransNominal_;
    }

    qTransAdaptive_ = (1 - alphaQ_) * qTransAdaptive_ + alphaQ_ * qTransNominal_;
    rTransAdaptive_ = (1 - alphaR_) * rTransAdaptive_ + alphaR_ * rTransNominal_;

    Q_trans_ = cv::Mat::eye(6, 6, CV_64F) * qTransAdaptive_;
    R_trans_ = cv::Mat::eye(2, 2, CV_64F) * rTransAdaptive_;
}

void KalmanSplitter::adaptQandR_rotation(double innovNorm, double dt) {
    if (innovNorm > innovThresholdHigh_) {
        qRotAdaptive_ = qRotNominal_ * 5.0;
        rRotAdaptive_ = rRotNominal_ * 3.0;
    } else if (innovNorm > innovThresholdLow_) {
        qRotAdaptive_ = qRotNominal_ * 2.0;
        rRotAdaptive_ = rRotNominal_ * 1.5;
    } else {
        qRotAdaptive_ = qRotNominal_;
        rRotAdaptive_ = rRotNominal_;
    }

    qRotAdaptive_ = (1 - alphaQ_) * qRotAdaptive_ + alphaQ_ * qRotNominal_;
    rRotAdaptive_ = (1 - alphaR_) * rRotAdaptive_ + alphaR_ * rRotNominal_;

    Q_rot_ = cv::Mat::eye(3, 3, CV_64F) * qRotAdaptive_;
    R_rot_ = cv::Mat::eye(1, 1, CV_64F) * rRotAdaptive_;
}

// ============================================================
// Сброс фильтра
// ============================================================
void KalmanSplitter::reset() {
    xTrans_ = cv::Mat::zeros(6, 1, CV_64F);
    P_trans_ = cv::Mat::eye(6, 6, CV_64F) * 10.0;
    innovTrans_ = cv::Mat::zeros(2, 1, CV_64F);
    innovNormTrans_ = 0.0;

    xRot_ = cv::Mat::zeros(3, 1, CV_64F);
    P_rot_ = cv::Mat::eye(3, 3, CV_64F) * 5.0;
    innovRot_ = cv::Mat::zeros(1, 1, CV_64F);
    innovNormRot_ = 0.0;

    highEnergyHistory_.clear();
}

// ============================================================
// Debug info
// ============================================================
KalmanSplitter::DebugInfo KalmanSplitter::getDebugInfo() const {
    DebugInfo info;
    info.innovationNorm = innovNormTrans_;
    info.qTranslation = qTransAdaptive_;
    info.qRotation = qRotAdaptive_;
    info.rTranslation = rTransAdaptive_;
    info.rRotation = rRotAdaptive_;
    info.mode = KalmanMotionComponents::MotionMode::STRAIGHT;
    return info;
}
