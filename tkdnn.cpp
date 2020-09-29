/*
### Build
g++ -g tkdnn.cpp -o tkdnn -ldl -O3 `pkg-config --cflags --libs opencv4`


### Run
# argv: camera index or video file
LD_LIBRARY_PATH=. ./tkdnn argv
*/

#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string>
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define MAX_BOXES 128

struct box {
    int cl;
    int x, y, w, h;
    float prob;
    const char* name;
};


int main(int argc, char **argv) {

    void *handle;
    void (*init_so)(const char* tensor_path, const char* names_path);
    void (*dispose_so)();
    void (*inference_mat_so)(cv::Mat& frame, box detected[]);
    char *error;

    // Open shared library
    handle = dlopen ("libtkDNN.so", RTLD_LAZY);
    if (!handle) {
        fputs (dlerror(), stderr);
        exit(1);
    }

    // Get functions
    init_so =  (void (*)(const char*, const char*))dlsym(handle, "init_so");
    if ((error = dlerror()) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }

    dispose_so = (void (*)())dlsym(handle, "dispose_so");
    if ((error = dlerror()) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }

    inference_mat_so = (void (*)(cv::Mat&, box[]))dlsym(handle, "inference_mat_so");
    if ((error = dlerror()) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }

    int camera_index = -1;
    if(argv[1] != NULL && strlen(argv[1] ) == 1)
        camera_index = atoi(argv[1]);

    cout<<"camera_index:"<<camera_index<<endl;

    bool gRun = true;

    // Parameter1: TensorRT model file path
    // Parameter2: class names file path
    init_so("model.rt","model.names");

    // OpenCV configuration
    cv::VideoCapture cap;
    if(camera_index >= 0)
    {
        cap.open(camera_index,cv::CAP_V4L2);
    }
    else
    {
        cap.open(argv[1]);
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    //VideoWriter video("outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'),30, Size(1280,720)); 

    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";
    
    cv::Mat frame, frame_show;

    int count = 0;

    namedWindow("AI",WINDOW_NORMAL);
    resizeWindow("AI", 1280, 720);

    unsigned long begin = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    unsigned long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    while(gRun) {
        cap >> frame; 

        if(!frame.data) 
            break;
        count++;

        frame_show = frame;
    
        // Parameter1: OpenCV mat frame
        // Parameter2: box array
        box detected[MAX_BOXES]={0}; /*bounding boxes in output*/
        
        inference_mat_so(frame,detected);

        // Calculate fps
        now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if((now/1000) - (begin/1000) >= 3)
        {
            cout<<"fps:"<<count/3<<endl;
            count = 0;
            begin = now;
        }

        // Draw UI
        for(int i=0; i< MAX_BOXES; i++){
            if(detected[i].w == 0 || detected[i].h == 0)
                break;

            cv::putText(frame_show,detected[i].name,cv::Point(detected[i].x, detected[i].y-10),cv::FONT_HERSHEY_DUPLEX,2.0,CV_RGB(118, 185, 0),2);
            Rect rect(detected[i].x, detected[i].y, detected[i].w, detected[i].h);
            rectangle(frame_show, rect, cv::Scalar(0, 255, 0),3);
        }
 
    	imshow("AI",frame_show);
        //video.write(frame_show);

        waitKey(1);
    }

    cap.release();
    //video.release();

    dispose_so();

    // Close handle
    dlclose(handle);
    return 0;
}
