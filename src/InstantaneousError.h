#ifndef INSTANTANEOUS_ERROR_H
#define INSTANTANEOUS_ERROR_H

#include "opencv2/core/core.hpp"

#include <vector>
#include <string>

// Instantaneous (per-frame) comparison of the estimated camera pose and the
// ground-truth camera pose. Both poses are 4x4 homogeneous matrices in the
// same convention (camera -> world), so their translation columns are directly
// comparable as positions in metres.
struct InstantaneousError {
    int   frame_id;
    double t_err_m;      // translation error |est - gt| [m]
    double r_err_deg;    // rotation error between the two orientations [deg]
    double est_x, est_y, est_z;   // estimated camera position [m]
    double gt_x,  gt_y,  gt_z;    // ground-truth camera position [m]
};

// Load KITTI-format ground-truth poses (each line = 12 numbers of a 3x4 matrix)
// into 4x4 homogeneous cv::Mat matrices. Returns an empty vector on failure.
std::vector<cv::Mat> loadGTPoses(const std::string& file_name);

// Compute the instantaneous error between one estimated pose and one ground-truth pose.
InstantaneousError computeInstantaneousError(int frame_id, const cv::Mat& est_pose, const cv::Mat& gt_pose);

#endif
