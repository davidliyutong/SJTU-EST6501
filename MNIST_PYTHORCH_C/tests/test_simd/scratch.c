#include <stdio.h>
#include <memory.h>
#include "pythorch/conv.h"
#include "pythorch/gemm.h"

const float test_x[2][784] = { 0 };
const int test_y[2] = { 0 };

// pythorch_err_t calc_fn(float* din, float* dout) {

//     pythorch_err_t err = PYTHORCH_OK;

//     conv2d_f32(var_0, din, 28, 28, conv2d_f32_0_weight, conv2d_f32_0_bias, conv2d_f32_0_shape);
//     relu_f32(var_1, var_0, 18432);
//     maxpool2d_f32(var_0, var_1, 24, 24, 32, 2);
//     conv2d_f32(var_1, var_0, 12, 12, conv2d_f32_3_weight, conv2d_f32_3_bias, conv2d_f32_3_shape);
//     relu_f32(var_0, var_1, 2048);
//     maxpool2d_f32(var_1, var_0, 8, 8, 32, 2);
//     /** Flatten **/
//     linear_f32(var_0, var_1, linear_f32_7_weight, linear_f32_7_bias, linear_f32_7_shape);
//     relu_f32(var_1, var_0, 1024);
//     /** Dropout **/
//     linear_f32(dout, var_1, linear_f32_10_weight, linear_f32_10_bias, linear_f32_10_shape);

//     return err;

// }

const float kernel[36] = {0,0,0,0,1,0,0,0,0,
                          0,0,0,0,0.5,0,0,0,0,
                          0,0,0,0,1,0,0,0,0,
                          0,0,0,0,1,0,0,0,0};

int run(float* din) {
    int res;
    float vmax;
    float dout[10];
    __attribute__((aligned(32))) float var_0[18432] = { 0 };
    __attribute__((aligned(32))) float var_1[18432] = { 0 };
    __attribute__((aligned(32))) float var_2[100] = { 0 };


    // calc_fn(din, dout);
    var_0[0] = 0;
    var_0[1] = 5;
    var_0[2] = -4;
    var_0[3] = 1;
    var_0[4] = 1;
    var_0[5] = 2;
    var_0[6] = -2;
    var_0[7] = 1;
    var_0[8] = 2;
    var_0[9] = -2;
    var_0[10] = -5;
    var_0[11] = 1;
    var_0[12] = 4;
    var_0[13] = 1;
    var_0[14] = 8;
    var_0[15] = 4;
    var_0[16] = 1;
    var_0[17] = 6;
    var_0[18] = -4;
    var_0[19] = -2;
    var_0[20] = 1;
    var_0[21] = -8;
    var_0[22] = 9;
    var_0[23] = 3;
    var_0[24] = 1;
    var_0[25 + 0] = 0;
    var_0[25 + 1] = 5;
    var_0[25 + 2] = -4;
    var_0[25 + 3] = 1;
    var_0[25 + 4] = 1;
    var_0[25 + 5] = 2;
    var_0[25 + 6] = -2;
    var_0[25 + 7] = 1;
    var_0[25 + 8] = 2;
    var_0[25 + 9] = -2;
    var_0[25 + 10] = -5;
    var_0[25 + 11] = 1;
    var_0[25 + 12] = 4;
    var_0[25 + 13] = 1;
    var_0[25 + 14] = 8;
    var_0[25 + 15] = 4;
    var_0[25 + 16] = 1;
    var_0[25 + 17] = 6;
    var_0[25 + 18] = -4;
    var_0[25 + 19] = -2;
    var_0[25 + 20] = 1;
    var_0[25 + 21] = -8;
    var_0[25 + 22] = 9;
    var_0[25 + 23] = 3;
    var_0[25 + 24] = 1;

    im2col(var_0, 2, 5, 5, 3, 3, 1, 0, var_1);
    gemm_f32(var_2, kernel, var_1, 2, 18, 2 * 3 * 3, 9);
    // col2im(var_0, 2, 5, 5, 3, 3, 1, 0, var_2);
    // int ret = memcmp(var_2, var_1, 25 * sizeof(float));

    // int shape[] = {5,5};
    // linear_f32(var_0, var_1, (const float*)var_1, (const float*)var_1, shape);
    // argmax_f32(&res, dout, 10);
    return res;
}

int main() {
    const float* din;
    int res, err;
    err = 0;
    din = &test_x[0][0];
    res = run((float*)din);
    err += res != test_y[0];
    return 0;
}
