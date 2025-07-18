#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

// CUDA核函数：彩色转灰度
__global__ void rgbToGray(const uchar3* const input, uchar* const output, int width, int height) {
    // 计算全局线程索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 检查坐标是否在图像范围内
    if (x < width && y < height) {
        // 获取当前像素位置
        int idx = y * width + x;
        
        // 获取RGB值
        uchar3 rgb = input[idx];
        
        // 计算灰度值 (标准转换公式)
        output[idx] = static_cast<uchar>(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
    }
}

// 验证CUDA和OpenCV结果是否一致
bool verifyResults(const cv::Mat& cvResult, const cv::Mat& cudaResult) {
    if (cvResult.size() != cudaResult.size() || cvResult.type() != cudaResult.type()) {
        return false;
    }
    
    for (int i = 0; i < cvResult.rows; ++i) {
        for (int j = 0; j < cvResult.cols; ++j) {
            if (cvResult.at<uchar>(i, j) != cudaResult.at<uchar>(i, j)) {
                std::cerr << "验证失败: 像素差异在 (" << i << ", " << j << ")" << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

int main(int argc, char** argv) {
    // 读取图像
    std::string imagePath = "../images/input.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "fail to load: " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "image size: " << image.cols << "x" << image.rows << std::endl;
    
    // 获取图像尺寸
    int width = image.cols;
    int height = image.rows;
    
    // --------------- OpenCV直接灰度转换 ---------------
    cv::Mat grayOpenCV;
    auto startCV = std::chrono::high_resolution_clock::now();
    
    cv::cvtColor(image, grayOpenCV, cv::COLOR_BGR2GRAY);
    
    auto endCV = std::chrono::high_resolution_clock::now();
    auto durationCV = std::chrono::duration_cast<std::chrono::milliseconds>(endCV - startCV).count();
    std::cout << "OpenCV cost time: " << durationCV << " ms" << std::endl;
    
    // --------------- CUDA灰度转换 ---------------
    // 分配GPU内存
    uchar3* d_input;
    uchar* d_output;
    cudaMalloc(&d_input, width * height * sizeof(uchar3));
    cudaMalloc(&d_output, width * height * sizeof(uchar));
    
    // 将图像数据传输到GPU
    cudaMemcpy(d_input, image.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    
    // 定义线程块和网格尺寸
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // 预热运行 (排除编译和初始化开销)
    rgbToGray<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();  // 确保核函数执行完成
    
    // 性能计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 执行核函数
    rgbToGray<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    
    // 同步并测量时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA cost time: " << milliseconds << " ms" << std::endl;
    
    // 创建输出图像
    cv::Mat grayCUDA(height, width, CV_8UC1);
    
    // 将结果从GPU传回CPU
    cudaMemcpy(grayCUDA.ptr(), d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
    
    // 释放GPU内存
    cudaFree(d_input);
    cudaFree(d_output);
    
    // 保存结果
    cv::imwrite("output_opencv.jpg", grayOpenCV);
    cv::imwrite("output_cuda.jpg", grayCUDA);
    
    // 验证结果
    bool resultsMatch = verifyResults(grayOpenCV, grayCUDA);
    std::cout << "Val: " << (resultsMatch ? "Success" : "Fail") << std::endl;
    
    // 计算加速比
    if (milliseconds > 0) {
        float speedup = durationCV / milliseconds;
        std::cout << speedup << "x "  << " CUDA faster than Opencv "<< std::endl;
    }
    
    std::cout << "End!" << std::endl;
    
    return 0;
}
