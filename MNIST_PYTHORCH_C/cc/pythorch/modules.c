/**
 * @file modules.c
 * @author davidliyutong@sjtu.edu.cn
 * @brief
 * @version 0.1
 * @date 2022-05-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <memory.h>

#include "modules.h"
#include "conv.h"
#include "gemm.h"


#ifdef __AVX__
#include <immintrin.h>
#include <emmintrin.h>
#include "avx_mathfun.h"
#endif

 /**
  * @brief Conv2d算子
  *
  * @param dout 输出数据
  * @param din 输入数据
  * @param din_hgt 输入数据(矩阵)高度
  * @param din_wid 输入数据(矩阵)宽度
  * @param weight 卷积核
  * @param bias 偏置
  * @param shape 卷积核形状
  * @param buf 缓冲区
  * @return pythorch_err_t
  */
pythorch_err_t conv2d_f32(float* dout,
                          float* din,
                          int din_hgt,
                          int din_wid,
                          const float* weight,
                          const float* bias,
                          const int* shape,
                          float* buf) {
    // 卷积核尺寸
    int num_c_out = shape[0], num_c_in = shape[1], k_hgt = shape[2], k_wid = shape[3];

    // 输出数据尺寸
    int dout_hgt = (din_hgt - k_hgt + 1);
    int dout_wid = (din_wid - k_wid + 1);

    for (int c_out = 0; c_out < num_c_out; c_out++) {   // 加上偏置

        int n = 0;
#ifdef __AVX__
        __m256 ymm0 = _mm256_set1_ps(bias[c_out]);
        for (n = 0; n < ((dout_hgt * dout_wid) - 8); n += 8) {
            _mm256_store_ps(dout + (c_out * dout_hgt * dout_wid + n), ymm0);
        }
#endif
        for (; n < dout_hgt * dout_wid; n++)
            dout[c_out * dout_hgt * dout_wid + n] = bias[c_out];
    }

#ifdef OPTIMIZE_IM2COL
    // 对输入执行im2col，这样不需要转换卷积核。而且输出天然就是正常排列
    if (buf) {
        im2col(din, num_c_in, din_hgt, din_wid, k_hgt, k_wid, 1, 0, buf);
        gemm_f32(dout,
                 (float*)weight,
                 buf,
                 num_c_out,
                 num_c_in * k_hgt * k_wid,
                 num_c_in * k_hgt * k_wid,
                 dout_wid * dout_hgt);
    } else { // buf为空指针，在嵌入式上重新启用naive卷积算法节约内存
#endif
    for (int cout = 0; cout < num_c_out; cout++) {
        // 对每个输入通道计算2D卷积
        for (int cin = 0; cin < num_c_in; cin++) {   // h和w是滑动窗位置
            for (int h = 0; h < dout_hgt; h++) {
                for (int w = 0; w < dout_wid; w++) {   // kh和kw是卷积核内的元素位置
                    for (int kh = 0; kh < k_hgt; kh++) {
                        for (int kw = 0; kw < k_wid; kw++)
                            dout[cout * dout_hgt * dout_wid + h * dout_wid + w] +=                          // dout[cout][h][w]
                            din[cin * din_hgt * din_wid + (h + kh) * din_wid + (w + kw)] *              // din[cin][h+kh][w+kw]
                            weight[cout * num_c_in * k_hgt * k_wid + cin * k_hgt * k_wid + kh * k_wid + kw];// ker[cout][cin][kh][kw]
                    }
                }
            }
        }
    }
#ifdef OPTIMIZE_IM2COL
    }
#endif

    return PYTHORCH_OK;
}

/**
 * @brief Linear全连接层算子
 *
 * @param dout 输出数据
 * @param din 输入数据
 * @param weight 权重
 * @param bias 偏置
 * @param shape 权重矩阵形状
 * @return pythorch_err_t
 */
pythorch_err_t linear_f32(float* dout,
                          float* din,
                          const float* weight,
                          const float* bias,
                          const int* shape) {
    // 数据尺寸
    int num_c_out = shape[0], num_c_in = shape[1];

    int c_out = 0;
    for (c_out = 0; c_out < num_c_out; c_out++) {
        dout[c_out] = bias[c_out];
        int c_in = 0;

#ifdef __AVX__
        // AVX 加速，一次处理8个浮点数
        __m256 ymm0, ymm1;
        __m256 accmm = _mm256_set1_ps(0.0f);
        __attribute__((aligned(32))) float acc[8] = { 0 };

        for (c_in = 0; c_in < ((num_c_in)-8); c_in += 8) {
            ymm0 = _mm256_load_ps(weight + c_out * num_c_in + c_in);
            ymm1 = _mm256_load_ps(din + c_in);
            // 8个浮点数相成，然后求和
            accmm = _mm256_add_ps(accmm, _mm256_mul_ps(ymm0, ymm1));
        }
        int p = 0;
        _mm256_store_ps(&acc[0], accmm);
        // 八个位置求和，循环展开
        dout[c_out] = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
#endif
        for (; c_in < num_c_in; c_in++)
            dout[c_out] += weight[c_out * num_c_in + c_in] * din[c_in];
    }

    return PYTHORCH_OK;
}

/**
 * @brief MaxPool2d算子
 *
 * @param dout 输出数据
 * @param din 输入数据
 * @param din_hgt 输入数据(矩阵)高度
 * @param din_wid 输入数据(矩阵)宽度
 * @param num_c 通道数
 * @param ksize 窗口尺寸
 * @return pythorch_err_t
 */
pythorch_err_t maxpool2d_f32(float* dout,
                             float* din,
                             int din_hgt,
                             int din_wid,
                             int num_c,
                             int ksize) {
    int dout_hgt = 1 + (din_hgt - ksize) / ksize;
    int dout_wid = 1 + (din_wid - ksize) / ksize;
    float m, v;
    float* din_sel;
#ifdef OPTIMIZE_INDEX
    float* dout_sel = dout;
#endif

    for (int c = 0; c < num_c; c++) {
        for (int h = 0; h < dout_hgt; h++) {
            for (int w = 0; w < dout_wid; w++) {
#ifdef OPTIMIZE_INDEX
                din_sel = &din[(c * din_hgt + h * ksize) * din_wid + w * ksize];
#else
                din_sel = &din[c * din_hgt * din_wid + h * ksize * din_wid + w * ksize];
#endif
                m = din_sel[0];
                for (int y = 0; y < ksize; y++) {
                    for (int x = 0; x < ksize; x++) {
                        v = din_sel[y * din_wid + x];
                        if (v > m) m = v;
                    }
                }
#ifdef OPTIMIZE_INDEX
                * dout_sel++ = m;
#else
                dout[c * dout_hgt * dout_wid + h * dout_wid + w] = m;
#endif
            }
        }
    }
    return PYTHORCH_OK;
}

/**
 * @brief AvgPool2d算子
 *
 * @param dout 输出数据
 * @param din 输入数据
 * @param din_hgt 输入数据(矩阵)高度
 * @param din_wid 输入数据(矩阵)宽度
 * @param num_c 通道数
 * @param ksize 窗口尺寸
 * @return pythorch_err_t
 */
pythorch_err_t avgpool2d_f32(float* dout,
                             float* din,
                             int din_hgt,
                             int din_wid,
                             int num_c,
                             int ksize) {
    int dout_hgt = 1 + (din_hgt - ksize) / ksize;
    int dout_wid = 1 + (din_wid - ksize) / ksize;
    float m;
    float* din_sel;
#ifdef OPTIMIZE_INDEX
    float* dout_sel = dout;
#endif

    for (int c = 0; c < num_c; c++) {
        for (int h = 0; h < dout_hgt; h++) {
            for (int w = 0; w < dout_wid; w++) {
#ifdef OPTIMIZE_INDEX
                din_sel = &din[(c * din_hgt + h * ksize) * din_wid + w * ksize];
#else
                din_sel = &din[c * din_hgt * din_wid + h * ksize * din_wid + w * ksize];
#endif
                m = 0;
                for (int y = 0; y < ksize; y++) {
                    for (int x = 0; x < ksize; x++) {
                        m += din_sel[y * din_wid + x];
                    }
                }
#ifdef OPTIMIZE_INDEX
                * dout_sel++ = m / (ksize * ksize);
#else
                dout[c * dout_hgt * dout_wid + h * dout_wid + w] = m / (ksize * ksize);
#endif
            }
        }
    }
    return PYTHORCH_OK;
}

/**
 * @brief ReLu算子
 *
 * @param dout 输出数据
 * @param din 输入数据
 * @param size 输入/输出尺寸
 * @return pythorch_err_t
 */
pythorch_err_t relu_f32(float* dout, float* din, int size) {
    int i = 0;
#ifdef __AVX__
    // AVX 加速，一次处理16个浮点数
    const __m256 zero = _mm256_set1_ps(0.0f);

    __m256 ymm0, ymm1;

    for (i = 0; i <= ((size)-16); i += 16) {
        ymm0 = _mm256_load_ps(din + i);
        ymm1 = _mm256_load_ps(din + i + 8);
        ymm0 = _mm256_max_ps(zero, ymm0);
        ymm1 = _mm256_max_ps(zero, ymm1);
        _mm256_store_ps(dout + i, ymm0);
        _mm256_store_ps(dout + i + 8, ymm1);
    }
#endif
    for (; i < size; i++)
        dout[i] = din[i] > 0 ? din[i] : 0;

    return PYTHORCH_OK;
}

/**
 * @brief Sigmoid算子
 *
 * @param dout 输出数据
 * @param din 输入数据
 * @param size 输入/输出尺寸
 * @return pythorch_err_t
 */
pythorch_err_t sigmoid_f32(float* dout, float* din, int size) {
    int i = 0;
#ifdef __AVX__
    // AVX 加速，一次处理16个浮点数
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);

    __m256 ymm0, ymm1, ymm2, ymm3;

    for (i = 0; i <= ((size)-16); i += 16) {
        ymm0 = _mm256_load_ps(din + i);
        ymm1 = _mm256_load_ps(din + i + 8);
        ymm0 = _mm256_sub_ps(zero, ymm0);
        ymm1 = _mm256_sub_ps(zero, ymm1);
        ymm2 = _mm256_add_ps(one, exp256_ps(ymm0)); // avx_math.h提供了exp函数
        ymm3 = _mm256_add_ps(one, exp256_ps(ymm0));
        ymm2 = _mm256_div_ps(one, ymm2);
        ymm3 = _mm256_div_ps(one, ymm3);
        _mm256_store_ps(dout + i, ymm2);
        _mm256_store_ps(dout + i + 8, ymm3);
    }
#endif
    for (; i < size; i++)
        dout[i] = (1 / (1 + expf(-din[i])));

    return PYTHORCH_OK;
}

/**
 * @brief Tanh算子
 *
 * @param dout 输出数据
 * @param din 输入数据
 * @param size 输入/输出尺寸
 * @return pythorch_err_t
 */
pythorch_err_t tanh_f32(float* dout, float* din, int size) {
    int i = 0;
#ifdef __AVX__
    // AVX 加速，一次处理16个浮点数
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 neg_two = _mm256_set1_ps(-2.0f);

    __m256 ymm0, ymm1, ymm2, ymm3;

    for (i = 0; i <= ((size)-16); i += 16) {
        ymm0 = _mm256_load_ps(din + i);
        ymm1 = _mm256_load_ps(din + i + 8);
        ymm0 = _mm256_mul_ps(neg_two, ymm0);
        ymm1 = _mm256_mul_ps(neg_two, ymm1);
        ymm2 = _mm256_add_ps(one, exp256_ps(ymm0));
        ymm3 = _mm256_add_ps(one, exp256_ps(ymm0));
        ymm2 = _mm256_div_ps(two, ymm2);
        ymm3 = _mm256_div_ps(two, ymm3);
        ymm2 = _mm256_add_ps(neg_one, ymm2);
        ymm3 = _mm256_add_ps(neg_one, ymm3);
        _mm256_store_ps(dout + i, ymm2);
        _mm256_store_ps(dout + i + 8, ymm3);
    }
#endif
    for (; i < size; i++)
        dout[i] = tanhf(din[i]);

    return PYTHORCH_OK;
}

/**
 * @brief 求Argmax
 *
 * @param dout 输出，int*
 * @param din 输入数据
 * @param bin 输入数据的尺寸
 * @return pythorch_err_t
 */
pythorch_err_t argmax_f32(int* dout, float* din, int bin) {
    float vmax = din[0];
    int res = 0;
    for (int n = 1; n < bin; n++)
        if (din[n] > vmax) {
            vmax = din[n];
            res = n;
        }
    *dout = res;

    return PYTHORCH_OK;
}
