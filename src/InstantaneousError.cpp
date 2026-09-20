#include "InstantaneousError.h"

#include <cmath>

std::vector<cv::Mat> loadGTPoses(const std::string& file_name) {
    std::vector<cv::Mat> poses;
    FILE* fp = fopen(file_name.c_str(), "r");
    if (!fp) return poses;

    while (!feof(fp)) {
        double vals[12];
        int n = fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                       &vals[0],  &vals[1],  &vals[2],  &vals[3],
                       &vals[4],  &vals[5],  &vals[6],  &vals[7],
                       &vals[8],  &vals[9],  &vals[10], &vals[11]);
        if (n == 12) {
            cv::Mat M = cv::Mat::eye(4, 4, CV_64F);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    M.at<double>(r, c) = vals[r * 4 + c];
            poses.push_back(M.clone());
        }
    }
    fclose(fp);
    return poses;
}

InstantaneousError computeInstantaneousError(int frame_id, const cv::Mat& est_pose, const cv::Mat& gt_pose) {
    InstantaneousError res;
    res.frame_id = frame_id;

    cv::Vec3d est_t(est_pose.at<double>(0, 3), est_pose.at<double>(1, 3), est_pose.at<double>(2, 3));
    cv::Vec3d gt_t (gt_pose.at<double>(0, 3),  gt_pose.at<double>(1, 3),  gt_pose.at<double>(2, 3));

    res.est_x = est_t[0]; res.est_y = est_t[1]; res.est_z = est_t[2];
    res.gt_x  = gt_t[0];  res.gt_y  = gt_t[1];  res.gt_z  = gt_t[2];

    cv::Vec3d d = est_t - gt_t;
    res.t_err_m = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

    cv::Mat est_R = (cv::Mat_<double>(3, 3) <<
        est_pose.at<double>(0, 0), est_pose.at<double>(0, 1), est_pose.at<double>(0, 2),
        est_pose.at<double>(1, 0), est_pose.at<double>(1, 1), est_pose.at<double>(1, 2),
        est_pose.at<double>(2, 0), est_pose.at<double>(2, 1), est_pose.at<double>(2, 2));
    cv::Mat gt_R = (cv::Mat_<double>(3, 3) <<
        gt_pose.at<double>(0, 0), gt_pose.at<double>(0, 1), gt_pose.at<double>(0, 2),
        gt_pose.at<double>(1, 0), gt_pose.at<double>(1, 1), gt_pose.at<double>(1, 2),
        gt_pose.at<double>(2, 0), gt_pose.at<double>(2, 1), gt_pose.at<double>(2, 2));

    cv::Mat R_rel = est_R.t() * gt_R;                 // relative rotation between orientations
    double trace = R_rel.at<double>(0, 0) + R_rel.at<double>(1, 1) + R_rel.at<double>(2, 2);
    double cos_angle = std::min(1.0, std::max(-1.0, (trace - 1.0) * 0.5));
    res.r_err_deg = std::acos(cos_angle) * 180.0 / CV_PI;

    return res;
}
