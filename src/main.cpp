#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;

void processNative(const Mat& input, Mat& output)
{
    GaussianBlur(input, output, Size(5, 5), 0);
}

void processOCL(const UMat& input, UMat& output)
{
    GaussianBlur(input, output, Size(5, 5), 0);
}

int main()
{
    // Подготовка видеокамеры
    //VideoCapture cap(0); // Используйте индекс камеры
    VideoCapture cap("http://192.168.0.100:4747/video?640x480");
    if (!cap.isOpened())
    {
        cerr << "Ошибка открытия веб-камеры." << endl;
        return -1;
    }

    // Матрицы для обработки изображений
    Mat frame;
    Mat resultNative;
    UMat uframe;
    UMat resultOCL;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;
        
        // frame.copyTo(resultNative); // Исходное изображение для сравнения
        frame.copyTo(uframe);       // Копия для OCL-обработки

        auto start = chrono::high_resolution_clock::now();
        // processNative(frame, resultNative);
        auto end = chrono::high_resolution_clock::now();
        // double timeNative = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        start = chrono::high_resolution_clock::now();
        processOCL(uframe, resultOCL);
        end = chrono::high_resolution_clock::now();
        double timeOCL = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        //cout << "Время обработки Native: " << timeNative << " ms" << endl;
        cout << "Время обработки OpenCL: " << timeOCL << " ms" << endl;

        //imshow("Native", resultNative);
        imshow("OpenCL", resultOCL.getMat(ACCESS_READ));

        char key = waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    
    destroyAllWindows();
    return 0;
}