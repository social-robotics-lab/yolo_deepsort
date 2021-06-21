# yolo_deepsort
Sample program to track humans using pytorch, Yolo v3, and deepsort.
[original](https://github.com/GlassyWing/yolo_deepsort)


# Install

## NVIDIA Driver for Cuda on WSL

Download and install: https://developer.nvidia.com/cuda/wsl/download

## Download code
```
git clone https://github.com/social-robotics-lab/yolo_deepsort.git
```

## Download weights
```
cd yolo_deepsort
wget -P src/weights https://pjreddie.com/media/files/yolov3.weights
wget -P src/weights https://pjreddie.com/media/files/darknet53.conv.74
```
Download [ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
and put it to src/weights.

## Docker build
```
docker build -t yolo_deepsort .
```


# Run
```
docker run -it --name yolo_deepsort --mount type=bind,source="$(pwd)"/src,target=/tmp --rm yolo_deepsort /bin/bash
python sample.py
```
