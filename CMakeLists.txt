cmake_minimum_required(VERSION 3.10)
project(CUDAImageProcessing)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 启用CUDA支持
enable_language(CUDA)

# 针对RTX 3060设置CUDA架构 (计算能力8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 查找OpenCV库
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV VERSION: ${OpenCV_VERSION}")
#message(STATUS "OpenCV库: ${OpenCV_LIBS}")
#message(STATUS "OpenCV头文件: ${OpenCV_INCLUDE_DIRS}")

# 添加可执行文件
add_executable(test
    src/main.cu  # 你的CUDA源文件
)

# 设置CUDA编译选项
set_target_properties(test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 包含目录
target_include_directories(test PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(test PRIVATE
    ${OpenCV_LIBS}
)
