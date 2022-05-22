#pragma once

typedef enum {
    PYTHORCH_OK,
    PYTHORCH_ERR
} pythorch_err_t;

pythorch_err_t conv2d_f32(float* dout,        // 输出数据
    float* din,         // 输入数据
    int din_hgt,        // 输入数据(矩阵)高度
    int din_wid,        // 输入数据(矩阵)宽度
    const float* weight,   // 卷积核
    const float* bias,  // 偏置
    const int* shape);   // 卷积核形状
// 全连接层运算
pythorch_err_t linear_f32(float* dout,// 输出数据
    float* din,         // 输入数据
    const float* weight,// 权重
    const float* bias,  // 偏置
    const int* shape);   // 权重矩阵形状

#define fc_f32 linear_f32

pythorch_err_t maxpool2d_f32(float* dout, // 输出数据
               float* din,  // 输入数据
               int din_hgt, // 输入数据(矩阵)高度
               int din_wid, // 输入数据(矩阵)宽度
               int num_c,   // 通道数
               int ksize);   // 窗口尺寸
pythorch_err_t relu_f32(float* dout, float* din, int size);
pythorch_err_t argmax_f32(int * dout, float* din, int bin);