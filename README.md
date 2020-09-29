# tkDNNuse
Demonstrate inference on Nvidia Jetson Xavier NX using libtkDNN.so<br>  
* C++ (Ready)
* C# (ToDo)

### Youtube video of results
[![tkDNN YOLOv4](https://img.youtube.com/vi/rn3lYs3jkGM/0.jpg)](https://youtu.be/rn3lYs3jkGM)

#### Download FP16 TensorRT model
https://drive.google.com/file/d/1mp-4jz14Euj-9zlXcMzSYs-sd2UoPdf-/view?usp=sharing


#### Build
g++ -g tkdnn.cpp -o tkdnn -ldl -O3 `pkg-config --cflags --libs opencv4`

#### Run
argv: camera index or video file <br>
LD_LIBRARY_PATH=. ./tkdnn argv
