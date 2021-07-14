# Use Image of CUDA11.3
FROM nvcr.io/nvidia/pytorch:21.05-py3

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

WORKDIR /workspace

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    cmake \
    g++ \
    gcc \
    git \
    gstreamer1.0-alsa \
    gstreamer1.0-doc \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-qt5 \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    libavcodec-dev \
    libavformat-dev \
    libgstreamer1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    libjpeg-dev  \
    libopenexr-dev \
    libpng-dev \
    libswscale-dev \
    libtiff-dev \
    libwebp-dev \
    ninja-build \
    pulseaudio

# OpenCV
RUN git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git

RUN cd opencv && mkdir build && cd build \
    && cmake .. -G "Ninja" \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DOPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules \
       -DWITH_PYTHON=ON \
       -DBUILD_opencv_python2=OFF \
       -DBUILD_opencv_python3=ON \
       -DPYTHON_DEFAULT_EXECUTABLE=/opt/conda/bin/python3 \
       -DPYTHON3_EXECUTABLE=/opt/conda/bin/python3 \
       -DPYTHON3_INCLUDE_DIR=/opt/conda/include/python3.8 \
       -DPYTHON3_LIBRARY=/opt/conda/lib/libpython3.8.so \
       -DPYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/lib/python3.8/site-packages/numpy/core/include \
       -DPYTHON3_PACKAGES_PATH=/opt/conda/lib/python3.8/site-packages \
       -DCMAKE_INSTALL_PREFIX="/usr" \
       -DOPENCV_ENABLE_NONFREE=OFF \
    && cmake --build . -j 8 --config RELEASE --target install

RUN rm -rf opencv opencv_contrib

RUN pip install imutils

ENV DISPLAY host.docker.internal:0.0

RUN useradd -m -d /home/sota -s /bin/bash sota
USER sota
WORKDIR /tmp