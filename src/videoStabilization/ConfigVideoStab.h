// main.cpp
#pragma once
#include <fstream>
#include <iostream>

//#include <opencv2/cudaoptflow.hpp> 
//#include <opencv2/cudawarping.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
int videoSource = 0;

bool writeVideo = false;
bool stabPossible = false;

const int compression = 1; // //4k 1->26ms 2->20ms 3->20ms

//
int	srcType = CV_8UC1;
int maxCorners = 400 / compression; //100/n
double qualityLevel = 0.0001 / compression; //0.0001
double minDistance = 1.0; //8.0
int blockSize = 8; //45 80 
bool useHarrisDetector = true;
double harrisK = qualityLevel;

// 
bool useGray = true;
int winSize = blockSize;
int maxLevel = 3 + 4/compression;
int iters = 10;
