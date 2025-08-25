//OpenCV
#include <opencv2/core.hpp>          // Mat, Scalar, Size, Rect, Point
#include <opencv2/imgproc.hpp>       // cvtColor, rectangle, ellipse, putText
#include <opencv2/highgui.hpp>       // imshow, imwrite
#include <opencv2/calib3d.hpp>       // findChessboardCorners, calibrateCamera
#include <opencv2/videoio.hpp>       // VideoCapture
#include <opencv2/cudaarithm.hpp>    // GpuMat, upload, download

// C++
#include <vector>    
#include <iostream>  
#include <thread>    
#include <mutex>     
#include <filesystem>

using namespace cv;
using namespace std;
//namespace fs = std::filesystem;

// 
const double DEG_TO_RAD = CV_PI / 180.0;
const double RAD_TO_DEG = 180.0 / CV_PI;

Scalar colorRED   (48, 62,  255);
Scalar colorYELLOW(5,  188, 251);
Scalar colorGREEN (82, 156,  23);
Scalar colorBLUE  (239,107,  23);
Scalar colorPURPLE(180,  0, 180);
Scalar colorWHITE (255,255, 255);
Scalar colorBLACK (0,    0,   0);


struct TransformParam
{
	TransformParam() {}
	TransformParam(double _dx, double _dy, double _da)
	{
		dx = _dx;
		dy = _dy;
		da = _da;
	}

	double dx;
	double dy;
	double da; // angle

	const void getTransform(Mat& T, double a, double b, double c, double atan_ba, double crop)
	{
		// Reconstruct transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);
		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;

	}

	const void getTransform(Mat& T)
	{
		// Reconstruct transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);
		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;

	}
	
	const void getTransformInvert(Mat& T, double a, double b, double c, double atan_ba, double crop)
	{
		// Reconstruct unverted transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(-da);
		T.at<double>(0, 1) = -sin(-da);
		T.at<double>(1, 0) = sin(-da);
		T.at<double>(1, 1) = cos(-da);
		T.at<double>(0, 2) = -dx;
		T.at<double>(1, 2) = -dy;

	}

	const void getTransformBoost(Mat& T, const int a, const int b, RNG rng)
	{
		T.at<double>(0, 0) = cos(0.0);
		T.at<double>(0, 1) = -sin(0.0);
		T.at<double>(1, 0) = sin(0.0);
		T.at<double>(1, 1) = cos(0.0);
		T.at<double>(0, 2) = a*atan((dx)*0.4)/4 + rng.uniform(-10, 10);
		T.at<double>(1, 2) = b*atan((dy)*0.4)/4 + rng.uniform(-10, 10);
	}
};


//int createFolders(vector <std::string>& folderPath);

void createPointColors(vector<Scalar>& colors, RNG& rng);


static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec);

static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec);

int camera_calibration(int argc, char** argv);

bool keyResponse(int& keyboard, Mat& frame, Mat& croppedImg, Mat& crossRef, cuda::GpuMat gCrossRef,
	const double& a, const double& b, double& nsr, bool& wiener, bool& threadwiener, double& Q,
	double& tauStab, double& framePart, Rect& roi);

void showServiceInfo(Mat& writerFrame, double Q, double nsr, bool wiener, bool threadwiener, bool stabPossible, vector <TransformParam> transforms, vector <TransformParam> movement,vector <TransformParam> movementKalman,
	double tauStab, double kSwitch, double framePart, int gP0_cols, int maxCorners,
	double seconds, double secondsPing, double secondsFullPing, int a, int b, vector <Point> textOrg, vector <Point> textOrgOrig, vector <Point> textOrgCrop, vector <Point> textOrgStab,
	int fontFace, double fontScale, Scalar color);

void showServiceInfoSmall(Mat& writerFrame, double Q, double nsr, bool wiener, bool threadwiener, bool stabPossible, vector <TransformParam> transforms, vector <TransformParam> movement, 
	double tauStab, double kSwitch, double framePart, int gP0_cols, int maxCorners,
	double seconds, double secondsPing, double secondsFullPing, int a, int b, vector <Point> textOrg, vector <Point> textOrgOrig, vector <Point> textOrgCrop, vector <Point> textOrgStab,
	int fontFace, double fontScale, Scalar color);

//#include <opencv2/opencv.hpp>

class KalmanFilterCV {
public:
	KalmanFilterCV(
		double dt,
		const cv::Mat& A,
		const cv::Mat& C,
		const cv::Mat& Q,
		const cv::Mat& R,
		const cv::Mat& P
	);

	KalmanFilterCV();

	void init();

	void init(double t0, const cv::Mat& x0);

	void update(const cv::Mat& y);

	void update(const cv::Mat& y, double dt, const cv::Mat& A);

	cv::Mat state() { return x_hat; };
	double time() { return t; };

private:
	// Matrices for computation
	cv::Mat A, C, Q, R, P, K, P0;

	// System dimensions
	int m, n;

	// Initial and current time
	double t0, t;

	// Discrete time step
	double dt;

	// Is the filter initialized?
	bool initialized;

	// n-size identity
	cv::Mat I;

	// Estimated states
	cv::Mat x_hat, x_hat_new;
};

// Implementation

KalmanFilterCV::KalmanFilterCV(
	double dt,
	const cv::Mat& A,
	const cv::Mat& C,
	const cv::Mat& Q,
	const cv::Mat& R,
	const cv::Mat& P
) : A(A.clone()), C(C.clone()), Q(Q.clone()), R(R.clone()), P(P.clone()),
dt(dt), initialized(false), t0(0), t(0) {

	m = C.rows;
	n = A.rows;
	I = cv::Mat::eye(n, n, CV_64F);
	x_hat = cv::Mat::zeros(n, 1, CV_64F);
	x_hat_new = cv::Mat::zeros(n, 1, CV_64F);
	P0 = P.clone();
}

KalmanFilterCV::KalmanFilterCV() : initialized(false), t0(0), t(0), dt(0), m(0), n(0) {}

void KalmanFilterCV::init() {
	x_hat.setTo(0);
	P = P0.clone();
	t = t0;
	initialized = true;
}

void KalmanFilterCV::init(double t0, const cv::Mat& x0) {
	this->t0 = t0;
	t = t0;
	x0.copyTo(x_hat);
	P = P0.clone();
	initialized = true;
}

void KalmanFilterCV::update(const cv::Mat& y) {
	if (!initialized) {
		throw std::runtime_error("Filter is not initialized!");
	}

	// Prediction step
	x_hat_new = A * x_hat;
	P = A * P * A.t() + Q;

	// Correction step
	K = P * C.t() * (C * P * C.t() + R).inv();
	x_hat_new += K * (y - C * x_hat_new);
	P = (I - K * C) * P;

	// Update state
	x_hat_new.copyTo(x_hat);
	t += dt;
}

void KalmanFilterCV::update(const cv::Mat& y, double dt, const cv::Mat& A) {
	this->dt = dt;
	A.copyTo(this->A);
	update(y);
}

// Функция для удаления пробелов в начале и конце строки
string trim(const string &str) {
    size_t start = str.find_first_not_of(" \t");
    if (start == string::npos) return "";
    size_t end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}