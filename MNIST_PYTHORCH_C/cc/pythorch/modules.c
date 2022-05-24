#include <stdio.h>
#include <math.h>
#include "modules.h"

pythorch_err_t conv2d_f32(float* dout,        // 输出数据
    float* din,         // 输入数据
    int din_hgt,        // 输入数据(矩阵)高度
    int din_wid,        // 输入数据(矩阵)宽度
    const float* weight,   // 卷积核
    const float* bias,  // 偏置
    const int* shape)   // 卷积核形状
{
    // 卷积核尺寸
    int num_cout = shape[0], num_cin = shape[1], k_hgt = shape[2], k_wid = shape[3];
 
    // 输出数据尺寸
    int dout_hgt = (din_hgt - k_hgt + 1);
    int dout_wid = (din_wid - k_wid + 1);

    for (int cout = 0; cout < num_cout; cout++)
    {   // 加上偏置
        for (int n = 0; n < dout_hgt * dout_wid; n++)
            dout[cout * dout_hgt * dout_wid + n] = bias[cout];
        // 对每个输入通道计算2D卷积
        for (int cin = 0; cin < num_cin; cin++)
        {   // h和w是滑动窗位置
            for (int h = 0; h < dout_hgt; h++)
            {
                for (int w = 0; w < dout_wid; w++)
                {   // kh和kw是卷积核内的元素位置
                    for (int kh = 0; kh < k_hgt; kh++)
                    {
                        for (int kw = 0; kw < k_wid; kw++)
                            dout[cout * dout_hgt * dout_wid + h * dout_wid + w] +=                          // dout[cout][h][w]
                                din[cin * din_hgt * din_wid + (h + kh) * din_wid + (w + kw)] *              // din[cin][h+kh][w+kw]
                                weight[cout * num_cin * k_hgt * k_wid + cin * k_hgt * k_wid + kh * k_wid + kw];// ker[cout][cin][kh][kw]
                    }
                }
            }
        }
    }
    return PYTHORCH_OK;
}

// 全连接层运算
pythorch_err_t linear_f32(float* dout,// 输出数据
    float* din,         // 输入数据
    const float* weight,// 权重
    const float* bias,  // 偏置
    const int* shape)   // 权重矩阵形状
{
    // 数据尺寸
    int num_cout = shape[0], num_cin = shape[1];
    for (int cout = 0; cout < num_cout; cout++)
    {
        dout[cout] = bias[cout];
        for (int cin = 0; cin < num_cin; cin++)
            dout[cout] += weight[cout * num_cin + cin] * din[cin];
    }

    return PYTHORCH_OK;
}

// pythorch_err_t (*fc_f32)(float*, float*, const float*, const float*, const int*);

pythorch_err_t maxpool2d_f32(float* dout, // 输出数据
               float* din,  // 输入数据
               int din_hgt, // 输入数据(矩阵)高度
               int din_wid, // 输入数据(矩阵)宽度
               int num_c,   // 通道数
               int ksize)   // 窗口尺寸
{
    int dout_hgt = 1 + (din_hgt - ksize) / ksize;
    int dout_wid = 1 + (din_wid - ksize) / ksize;
    float m,v;
    float* din_sel;

    for (int c = 0; c < num_c; c++)
    {
        for (int h = 0; h < dout_hgt; h++)
        {
            for (int w = 0; w < dout_wid; w++)
            {
                din_sel = &din[c * din_hgt * din_wid + h * ksize * din_wid + w * ksize];
                m = din_sel[0];
                for (int y = 0; y < ksize; y++)
                {
                    for (int x = 0; x < ksize; x++)
                    {
                        v = din_sel[y * din_wid + x];
                        if (v > m) m = v;
                    }
                }
                dout[c * dout_hgt * dout_wid + h * dout_wid + w] = m;
            }
        }
    }
    return PYTHORCH_OK;
}

pythorch_err_t avgpool2d_f32(float* dout, // 输出数据
               float* din,  // 输入数据
               int din_hgt, // 输入数据(矩阵)高度
               int din_wid, // 输入数据(矩阵)宽度
               int num_c,   // 通道数
               int ksize)   // 窗口尺寸
{
    int dout_hgt = 1 + (din_hgt - ksize) / ksize;
    int dout_wid = 1 + (din_wid - ksize) / ksize;
    float m,v;
    float* din_sel;

    for (int c = 0; c < num_c; c++)
    {
        for (int h = 0; h < dout_hgt; h++)
        {
            for (int w = 0; w < dout_wid; w++)
            {
                din_sel = &din[c * din_hgt * din_wid + h * ksize * din_wid + w * ksize];
                m = 0;
                for (int y = 0; y < ksize; y++)
                {
                    for (int x = 0; x < ksize; x++)
                    {
                        m += din_sel[y * din_wid + x];
                    }
                }
                dout[c * dout_hgt * dout_wid + h * dout_wid + w] = m / (ksize * ksize);
            }
        }
    }
    return PYTHORCH_OK;
}


pythorch_err_t relu_f32(float* dout, float* din, int size)
{
    for (int n = 0; n < size; n++)
        dout[n] = din[n] > 0 ? din[n] : 0;
    
    return PYTHORCH_OK;
}

pythorch_err_t sigmoid_f32(float* dout, float* din, int size)
{
    for (int n = 0; n < size; n++)
        dout[n] = (1 / (1 + exp(-din[n])));
    
    return PYTHORCH_OK;
}

pythorch_err_t tanh_f32(float* dout, float* din, int size)
{
    for (int n = 0; n < size; n++)
        dout[n] = tanh(din[n]);
    
    return PYTHORCH_OK;
}

pythorch_err_t argmax_f32(int * dout, float* din, int bin) {
    float vmax = din[0];
    int res = 0;
    for (int n = 1; n < bin; n++)
        if (din[n] > vmax)
        {
            vmax = din[n];
            res = n;
        }
    *dout = res;
    
    return PYTHORCH_OK;
}
