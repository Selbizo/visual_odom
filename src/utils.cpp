#include "utils.h"
#include "evaluate_odometry.h"
#include <fstream>
#include <vector>



// --------------------------------
// Loop closure correction for trajectory
// --------------------------------
static std::vector<int> corrected_frame_ids_;
static std::vector<cv::Mat> corrected_poses_;
static int loop_candidate_id_ = -1;
static int loop_current_id_ = -1;
static cv::Mat loop_R_delta_;
static cv::Mat loop_t_delta_;

static cv::Mat rotationMatrixFromEuler(const cv::Vec3f& euler)
{
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    
    float cx = cos(euler[0]);
    float sx = sin(euler[0]);
    float cy = cos(euler[1]);
    float sy = sin(euler[1]);
    float cz = cos(euler[2]);
    float sz = sin(euler[2]);
    
    R.at<double>(0, 0) = cy * cz;
    R.at<double>(0, 1) = sx * sy * cz - cx * sz;
    R.at<double>(0, 2) = cx * sy * cz + sx * sz;
    R.at<double>(1, 0) = cy * sz;
    R.at<double>(1, 1) = sx * sy * sz + cx * cz;
    R.at<double>(1, 2) = cx * sy * sz - sx * cz;
    R.at<double>(2, 0) = -sy;
    R.at<double>(2, 1) = sx * cy;
    R.at<double>(2, 2) = cx * cy;
    
    return R;
}

static void applyLoopClosureToTrajectory()
{
    if (loop_candidate_id_ < 0 || loop_current_id_ < 0 || corrected_poses_.empty()) {
        return;
    }
    
    if (loop_R_delta_.empty() || loop_t_delta_.empty()) {
        return;
    }
    
    int span = loop_current_id_ - loop_candidate_id_;
    if (span <= 0) {
        return;
    }
    
    cv::Mat rvec_delta;
    cv::Rodrigues(loop_R_delta_, rvec_delta);
    
    for (size_t i = 0; i < corrected_frame_ids_.size(); i++) {
        int frame_id = corrected_frame_ids_[i];
        
        if (frame_id > loop_candidate_id_ && frame_id < loop_current_id_) {
            double alpha = static_cast<double>(frame_id - loop_candidate_id_) / static_cast<double>(span);
            
            cv::Mat rvec_partial = rvec_delta * alpha;
            cv::Mat R_partial;
            cv::Rodrigues(rvec_partial, R_partial);
            cv::Mat t_partial = loop_t_delta_ * alpha;
            
            cv::Mat T_partial = cv::Mat::eye(4, 4, CV_64F);
            R_partial.copyTo(T_partial(cv::Rect(0, 0, 3, 3)));
            t_partial.copyTo(T_partial(cv::Rect(3, 0, 1, 3)));
            
            cv::Mat old_pose_mat = corrected_poses_[i];
            cv::Mat new_pose_mat = T_partial * old_pose_mat;
            new_pose_mat.copyTo(corrected_poses_[i]);
        } else if (frame_id == loop_current_id_) {
            cv::Mat T_full = cv::Mat::eye(4, 4, CV_64F);
            loop_R_delta_.copyTo(T_full(cv::Rect(0, 0, 3, 3)));
            loop_t_delta_.copyTo(T_full(cv::Rect(3, 0, 1, 3)));
            cv::Mat old_pose_mat = corrected_poses_[i];
            cv::Mat new_pose_mat = T_full * old_pose_mat;
            new_pose_mat.copyTo(corrected_poses_[i]);
        }
    }
    
    loop_candidate_id_ = -1;
    loop_current_id_ = -1;
    loop_R_delta_.release();
    loop_t_delta_.release();
}

// --------------------------------
// Loop closure correction API
// --------------------------------

void setLoopClosureCorrection(int candidate_frame_id, int current_frame_id, const cv::Mat& R_delta, const cv::Mat& t_delta)
{
    loop_candidate_id_ = candidate_frame_id;
    loop_current_id_ = current_frame_id;
    
    if (!R_delta.empty() && !t_delta.empty()) {
        R_delta.clone().copyTo(loop_R_delta_);
        t_delta.clone().copyTo(loop_t_delta_);
    }
}

// --------------------------------
// Visualization
// --------------------------------
void drawFeaturePoints(cv::Mat image, std::vector<cv::Point2f>& points)
{
    int radius = 2;
    
    for (int i = 0; i < points.size(); i++)
    {
        circle(image, cv::Point(points[i].x, points[i].y), radius, CV_RGB(255,255,255));
    }
}

void display(int frame_id, cv::Mat& trajectory, cv::Mat& trajectory_biased, cv::Mat& pose, float fps)
{
    static std::ofstream coord_file("/home/selbizo/CV/StabAndSLAM/visual_odom/trajectory_coordinates.txt");
    
    static int last_applied_frame_id = -1;
    
    if (last_applied_frame_id >= 0 && frame_id > last_applied_frame_id) {
        applyLoopClosureToTrajectory();
        last_applied_frame_id = -1;
    }
    
    if (coord_file.is_open() && frame_id%29 == 0) {
        coord_file << frame_id << " " 
                   << pose.at<double>(0) << " " 
                   << pose.at<double>(1) << " " 
                   << pose.at<double>(2) << std::endl;
    }
    
    cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_64F);
    pose_mat.at<double>(0, 3) = pose.at<double>(0);
    pose_mat.at<double>(1, 3) = pose.at<double>(1);
    pose_mat.at<double>(2, 3) = pose.at<double>(2);
    
    cv::Mat R_temp = pose_mat(cv::Rect(0, 0, 3, 3)).clone();
    cv::Vec3f euler = rotationMatrixToEulerAngles(R_temp);
    cv::Mat R = rotationMatrixFromEuler(euler);
    R.copyTo(pose_mat(cv::Rect(0, 0, 3, 3)));
    
    corrected_frame_ids_.push_back(frame_id);
    corrected_poses_.push_back(pose_mat.clone());
    
    // draw estimated trajectory 
    int x = trajectory.cols/2 + int(pose.at<double>(0));
    int y = trajectory.rows/2 - int(pose.at<double>(2));
    circle(trajectory, cv::Point(x, y) ,1, CV_RGB(130,180,230), 2);


    cv::Mat Bias = (cv::Mat_<double>(2, 3) <<
    1, 0, -pose.at<double>(0) - (trajectory.cols - trajectory_biased.cols)/2, 
    0, 1, pose.at<double>(2) - (trajectory.rows - trajectory_biased.rows)/2
    );
    cv::Mat temp_biased;
    cv::warpAffine(trajectory, temp_biased, Bias, trajectory_biased.size());
    temp_biased.copyTo(trajectory_biased);
    temp_biased.release();
    cv::imshow( "Trajectory my", trajectory_biased);
    cv::Mat temp_trajectory = trajectory * 0.98;
    temp_trajectory.copyTo(trajectory);
    temp_trajectory.release();
    cv::waitKey(1);
}



// --------------------------------
// Transformation
// --------------------------------


void integrateOdometryStereo(int frame_i, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, const cv::Mat& translation_stereo)
{

    // std::cout << "rotation" << rotation << std::endl;
    // std::cout << "translation_stereo" << translation_stereo << std::endl;

    
    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

    // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

    double scale = sqrt((translation_stereo.at<double>(0))*(translation_stereo.at<double>(0)) 
                        + (translation_stereo.at<double>(1))*(translation_stereo.at<double>(1))
                        + (translation_stereo.at<double>(2))*(translation_stereo.at<double>(2))) ;

    // frame_pose = frame_pose * rigid_body_transformation;
    // std::cout << "scale: " << scale << std::endl;

    rigid_body_transformation = rigid_body_transformation.inv();
    // if ((scale>0.1)&&(translation_stereo.at<double>(2) > translation_stereo.at<double>(0)) && (translation_stereo.at<double>(2) > translation_stereo.at<double>(1))) 
    if (scale > 0.002 && scale < 400000.0) 
    {
      // std::cout << "Rpose" << Rpose << std::endl;

      frame_pose = frame_pose * rigid_body_transformation;

    }
    else 
    {
        // scale is out of range, skip integration but don't print warning
    }
}

bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
     
    return  norm(I, shouldBeIdentity) < 1e-6;
     
}
 
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{
 
    assert(isRotationMatrix(R));
     
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
     
}

// --------------------------------
// I/O
// --------------------------------

void loadGyro(std::string filename, std::vector<std::vector<double>>& time_gyros)
// read time gyro txt file with format of timestamp, gx, gy, gz
{
    std::ifstream file(filename);

    std::string value;
    double timestamp, gx, gy, gz;

    while (file.good())
    {    

         std::vector<double> time_gyro;

         getline ( file, value, ' ' );
         timestamp = stod(value);
         time_gyro.push_back(timestamp);

         getline ( file, value, ' ' );
         gx = stod(value);
         time_gyro.push_back(gx);

         getline ( file, value, ' ' );
         gy = stod(value);
         time_gyro.push_back(gy);

         getline ( file, value);
         gz = stod(value);
         time_gyro.push_back(gz);

         // printf("t: %f, gx: %f, gy: %f, gz: %f\n" , timestamp, gx, gy, gz);    

         time_gyros.push_back(time_gyro);
    }
}

void loadImageLeft(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, std::string filepath){
    char file[200];
    sprintf(file, "image_0/%06d.png", frame_id);

    // sprintf(file, "image_0/%010d.png", frame_id);
    std::string filename = filepath + std::string(file);
    image_color.release();
    image_gary.release();
    image_color = cv::imread(filename, cv::IMREAD_COLOR);
    cvtColor(image_color, image_gary, cv::COLOR_BGR2GRAY);
}

void loadImageRight(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, std::string filepath){
    char file[200];
    sprintf(file, "image_1/%06d.png", frame_id);

    // sprintf(file, "image_0/%010d.png", frame_id);
    std::string filename = filepath + std::string(file);
    image_color.release();
    image_gary.release();
    image_color = cv::imread(filename, cv::IMREAD_COLOR);
    cvtColor(image_color, image_gary, cv::COLOR_BGR2GRAY);
}