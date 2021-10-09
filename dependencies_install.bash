#!/bin/bash
DEPS_DIR="/tmp/vicon_calibration_dependencies"

# get release of Ubuntu
export UBUNTU_CODENAME=$(lsb_release -s -c)
case $UBUNTU_CODENAME in
  xenial)
    export ROS_DISTRO=kinetic;;
  bionic)
    export ROS_DISTRO=melodic;;  
  *)
    echo "Unsupported version of Ubuntu detected. Only xenial (16.04.*) and bionic (18.04.*) are supported. Exiting."
    exit  1
esac

# get number of processors for high-load installs
if [ $(nproc) -lt 2 ]; then
  echo "A minimum of 2 processors is required for installation. Exiting."
  exit  1
else
  export NUM_PROCESSORS=$(( $(nproc) / 2 ))
fi

make_with_progress()
{
    if [ -z "$CONTINUOUS_INTEGRATION" ]; then
        local awk_arg="-W interactive"
    fi
    # Run make, printing a character for every 10 lines
    make "$@" | awk ${awk_arg} 'NR%5==1 { printf ".", $0}'
    echo "done"
}


install_ceres()
{
    CERES_DIR="ceres-solver-1.14.0"
    BUILD_DIR="build"

    sudo apt-get -qq install libgoogle-glog-dev libatlas-base-dev > /dev/null
    # this install script is for local machines.
    if (find /usr/local/lib -name libceres.so | grep -q /usr/local/lib); then
        echo "Ceres is already installed."
    else
        echo "Installing Ceres 1.14.0 ..."
        mkdir -p "$DEPS_DIR"
        cd "$DEPS_DIR"

        if [ ! -d "$CERES_DIR" ]; then
          wget "http://ceres-solver.org/$CERES_DIR.tar.gz"
          tar zxf "$CERES_DIR.tar.gz"
          rm -rf "$CERES_DIR.tar.gz"
        fi

        cd $CERES_DIR
        if [ ! -d "$BUILD_DIR" ]; then
          mkdir -p $BUILD_DIR
          cd $BUILD_DIR
          cmake ..
          make -j$(nproc)
          make test
        fi

        cd $DEPS_DIR/$CERES_DIR/$BUILD_DIR
        sudo make -j$(nproc) install
    fi
}

install_pcl()
{
  PCL_VERSION="1.11.1"
  PCL_DIR="pcl"
  BUILD_DIR="build"

  cd $DEPS_DIR

  if [ -d 'pcl-pcl-1.8.0' ]; then
    echo "Removing old version of pcl (pcl-1.8.0) from deps"
    sudo rm -rf pcl-pcl-1.8.0
  fi

  if [ -d 'pcl-pcl-1.8.1' ]; then
    echo "Removing old version of pcl (pcl-1.8.1) from deps"
    sudo rm -rf pcl-pcl-1.8.1
  fi

  if [ ! -d "$PCL_DIR" ]; then
    echo "pcl not found... cloning"
    git clone https://github.com/PointCloudLibrary/pcl.git
    cd $PCL_DIR
    git checkout pcl-$PCL_VERSION
    cd ..
  fi
  
  cd $PCL_DIR
  if [ ! -d "$BUILD_DIR" ]; then
    echo "Existing build of PCL not found.. building from scratch"
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR

    PCL_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14"
    if [ -n "$CONTINUOUS_INTEGRATION" ]; then
              # Disable everything unneeded for a faster build
              echo "Installing light build for CI"
              PCL_CMAKE_ARGS="${PCL_CMAKE_ARGS} \
              -DWITH_CUDA=OFF -DWITH_DAVIDSDK=OFF -DWITH_DOCS=OFF \
              -DWITH_DSSDK=OFF -DWITH_ENSENSO=OFF -DWITH_FZAPI=OFF \
              -DWITH_LIBUSB=OFF -DWITH_OPENGL=OFF -DWITH_OPENNI=OFF \
              -DWITH_OPENNI2=OFF -DWITH_QT=OFF -DWITH_RSSDK=OFF \
              -DBUILD_CUDA=OFF -DBUILD_GPU=OFF \
              -DBUILD_tracking=OFF -DBUILD_people=OFF \
              -DBUILD_stereo=OFF -DBUILD_simulation=OFF -DBUILD_apps=OFF \
              -DBUILD_examples=OFF -DBUILD_tools=OFF -DBUILD_visualization=ON"
    fi

    cmake .. ${PCL_CMAKE_ARGS}
    make -j$NUM_PROCESSORS
  fi

  cd $DEPS_DIR/$PCL_DIR/$BUILD_DIR
  sudo make -j$NUM_PROCESSORS install
}

install_catch2()
{
  echo "Installing Catch2..."
  CATCH2_DIR="Catch2"
  BUILD_DIR="build"
  mkdir -p $DEPS_DIR
  cd $DEPS_DIR

  if [ ! -d "$DEPS_DIR/$CATCH2_DIR" ]; then
    git clone https://github.com/catchorg/Catch2.git --branch v2.13.2 $DEPS_DIR/$CATCH2_DIR
  fi

  cd $CATCH2_DIR
  if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    cmake -DCMAKE_CXX_STANDARD=11 ..
    make -j$(nproc)
  fi

  cd $DEPS_DIR/$CATCH2_DIR/$BUILD_DIR
  sudo make -j$(nproc) install
}

install_eigen3()
{
  EIGEN_DIR="eigen-3.3.7"
  BUILD_DIR="build"
  mkdir -p $DEPS_DIR
  cd $DEPS_DIR

  if [ ! -d "$EIGEN_DIR" ]; then
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2
    tar xjf eigen-3.3.7.tar.bz2
    rm -rf eigen-3.3.7.tar.bz2
  fi

  cd $EIGEN_DIR
  if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    cmake ..
    make
  fi

  cd $DEPS_DIR/$EIGEN_DIR/$BUILD_DIR
  sudo make -j$(nproc) install
}

install_gflags()
{
  sudo apt-get install libgflags-dev
}

install_gflags_from_source()
{
  GFLAGS_DIR="gflags"
  BUILD_DIR="build"
  mkdir -p $DEPS_DIR
  cd $DEPS_DIR

  if [ ! -d "$GFLAGS_DIR" ]; then
    git clone https://github.com/gflags/gflags.git
  fi

  cd $GFLAGS_DIR
  if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=true
    make
  fi

  cd $DEPS_DIR/$GFLAGS_DIR/$BUILD_DIR
  sudo make -j$(nproc) install

  # remove error inducing gtest and gmock (should just exist in /usr/include)
  GTEST_PATH="/usr/local/include/gtest"
  GMOCK_PATH="/usr/local/include/gmock"
  if test -f $GTEST_PATH; then
    sudo rm -r $GTEST_PATH
  fi

  if test -f $GMOCK_PATH; then
    sudo rm -r $GMOCK_PATH
  fi
}

install_json()
{
  JSON_DIR="json"
  BUILD_DIR="build"
  mkdir -p $DEPS_DIR
  cd $DEPS_DIR

  if [ ! -d "$JSON_DIR" ]; then
    git clone -b v3.6.1 https://github.com/nlohmann/json.git
  fi

  cd $JSON_DIR
  if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    cmake ..
    make -j$(nproc)
  fi

  cd $DEPS_DIR/$JSON_DIR/$BUILD_DIR
  sudo make -j$(nproc) install
}
