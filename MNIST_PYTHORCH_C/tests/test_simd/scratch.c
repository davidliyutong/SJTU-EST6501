#include<emmintrin.h>
#include<immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

void print128_num_epi32(__m128i var) {
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %i %i %i %i \n",
           val[0], val[1], val[2], val[3]);
}

int conv2d_f32(float* dout,        // 输出数据
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

    for (int cout = 0; cout < num_cout; cout++) {   // 加上偏置
        for (int n = 0; n < dout_hgt * dout_wid; n++)
            dout[cout * dout_hgt * dout_wid + n] = bias[cout];
        // 对每个输入通道计算2D卷积
        for (int cin = 0; cin < num_cin; cin++) {   // h和w是滑动窗位置
            for (int h = 0; h < dout_hgt; h++) {
                for (int w = 0; w < dout_wid; w++) {   // kh和kw是卷积核内的元素位置
                    for (int kh = 0; kh < k_hgt; kh++) {
                        for (int kw = 0; kw < k_wid; kw++)
                            dout[cout * dout_hgt * dout_wid + h * dout_wid + w] +=                          // dout[cout][h][w]
                            din[cin * din_hgt * din_wid + (h + kh) * din_wid + (w + kw)] *              // din[cin][h+kh][w+kw]
                            weight[cout * num_cin * k_hgt * k_wid + cin * k_hgt * k_wid + kh * k_wid + kw];// ker[cout][cin][kh][kw]
                    }
                }
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    // use SSE3 to implement integer calculation.
    int first_array[4] __attribute__((aligned(32))) = { 1, 2, 3, 4 };
    int second_array[4] __attribute__((aligned(32))) = { 1, 2, 3, 4 };

    __m128i first_values = _mm_set_epi32(first_array[0], first_array[1], first_array[2], first_array[3]);
    __m128i second_values = _mm_set_epi32(second_array[0], second_array[1], second_array[2], second_array[3]);
    __m128i rst;

    print128_num_epi32(first_values);
    print128_num_epi32(second_values);
    rst = _mm_add_epi32(first_values, second_values);

    print128_num_epi32(rst);
    return 0;
}