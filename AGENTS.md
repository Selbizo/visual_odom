# Visual Odometry System - Agent Customization

## Project Overview

This is a C++ OpenCV implementation of Stereo Visual Odometry that uses OpenCV's `calcOpticalFlowPyrLK` for feature tracking. It's designed to work with the KITTI odometry dataset and includes video stabilization capabilities.

## Architecture and Components

### Core Modules:
1. **visualOdometry** - Main visual odometry logic
2. **Frame** - Frame representation with pose information  
3. **feature** - Feature point handling
4. **bucket** - Feature bucketing system for efficient processing
5. **utils** - Utility functions for visualization and display
6. **camera_object** - Camera parameter management

### Video Stabilization Components:
1. **VideoStabilizationPipeline** - High-level stabilization pipeline (currently in refactoring)
2. **basicFunctions** - Basic image processing functions
3. **stabilizationFunctions** - Core stabilization algorithms  
4. **kalmanSplitter** - Kalman filter-based motion estimation
5. **wienerFilter** - Wiener filtering for noise reduction

### Build System (CMake):
- Uses CMake 3.5+ with C++17 standard
- Supports CUDA acceleration via `-DUSE_CUDA=on` flag  
- Modular structure with subdirectories for different components
- Links against OpenCV libraries

## Key Features:
- Stereo visual odometry using feature tracking
- GPU acceleration support (CUDA)
- Video stabilization capabilities 
- KITTI dataset compatibility
- Camera parameter calibration support

## Development Patterns:

1. **Modular Design**: Each component has its own source and header files
2. **Component-based Architecture**: Clear separation of concerns between features, frames, odometry, and stabilization  
3. **CUDA Support**: Optional GPU acceleration for performance-critical functions
4. **Configuration-driven**: Camera parameters loaded from YAML files
5. **Extensible Structure**: Easy to add new components or modify existing ones

## File Organization:
- `src/` - Main source code directory with core modules
- `videoStabilization/` - Video stabilization related components  
- `evaluate/` - Evaluation and logging functionality
- `calibration/` - Camera calibration parameters
- `build/` - Build artifacts and compiled binaries

## Key Functions:
1. **matchingFeatures** - Feature matching between frames
2. **matchingFeaturesStab** - Stabilized feature matching with CUDA support  
3. **euler2rot** - Conversion from Euler angles to rotation matrix
4. **checkValidMatch** - Validation of matched features

## Current Development Status

The system has an active refactoring effort focused on the video stabilization pipeline, as indicated by the TODO list:
- A new `VideoStabilizationPipeline` class is being created  
- The stabilization logic needs to be encapsulated in a single interface
- GPU resource initialization should move from main loop to constructor
- Process frame functionality needs to be simplified to 3-5 lines of code

## Important Conventions and Pitfalls:

1. **CUDA Dependency Issues**: 
   - The system requires CUDA-compatible hardware
   - Compilation with `-DUSE_CUDA=on` flag is needed for GPU acceleration  
   - Error handling when CUDA libraries aren't available

2. **Memory Management**:
   - Large feature sets can cause performance issues (threshold of 2000 features)
   - Memory leaks or inefficient memory usage in the bucketing system could occur
   - The code uses manual memory management with `reserve()` and `swap()`

3. **Feature Tracking Limitations**:
   - The circular matching validation might be too strict for some scenarios (threshold of 0 pixels)  
   - Feature point removal logic may remove valid features under certain conditions

4. **Thread Safety**: 
   - No explicit thread safety mechanisms, which could cause issues in multi-threaded environments
   - Shared state between frames needs careful handling

5. **Configuration Dependencies**:
   - Camera parameters must be correctly formatted in YAML files  
   - Dataset path requirements (KITTI format expected)
   - Calibration file paths are hardcoded in the build process

6. **Performance Considerations**:
   - The system is designed for real-time performance with CUDA acceleration
   - Memory allocation patterns need to be optimized for continuous processing
   - Frame rate handling and timing considerations

7. **Code Quality Issues**:
   - Some code has Russian comments, which might affect maintainability  
   - Mixed C++ standards (C++17 used but some older practices)
   - Inconsistent naming conventions in some parts of the codebase

8. **Build System Dependencies**:
   - Requires specific OpenCV version (3.0+)
   - Complex dependency chain between libraries
   - Need to ensure all subdirectories are properly included in CMakeLists.txt

## Build Instructions:

### Basic build without CUDA:
```bash
mkdir build
cd build
cmake ..
make -j4
```

### Build with CUDA acceleration:
```bash
mkdir build
cd build
cmake .. -DUSE_CUDA=on
make -j4
```

### Run the system:
```bash
./run /path/to/KITTI/dataset/sequences/00/ ../calibration/kitti00.yaml
```