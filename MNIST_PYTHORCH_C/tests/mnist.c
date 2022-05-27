#include <stdio.h>
#include "pythorch/modules.h"
#include "param.h"
#include "test_data.h"

float buf0[32 * 24 * 24];
float buf1[32 * 24 * 24];
float buf[51200];

int calc(float* din)
{
    int res;
    float vmax;

    // x = self.conv1(din)       # (1, 28, 28)->(32, 24, 24)
    conv2d_f32(buf0,                    // 输出数据
               din,                     // 输入数据
               28,28,                   // 输入数据(矩阵)高度/宽度
               seq_0_weight,            // 卷积核
               seq_0_bias,              // 偏置
               seq_0_weight_shape,
               buf);     // 卷积核形状
    // x = F.relu(x)
    relu_f32(buf0, buf0, 32 * 24 * 24);
    // x = F.max_pool2d(x, 2)  # (32, 24, 24)->(32, 12, 12)
    maxpool2d_f32(buf1, buf0, 24, 24, 32, 2);
    // x = self.conv2(x)       # (32, 12, 12)->(32, 8, 8)
    conv2d_f32(buf0,                // 输出数据
               buf1,                // 输入数据
               12, 12,              // 输入数据(矩阵)高度/宽度
               seq_3_weight,        // 卷积核
               seq_3_bias,          // 偏置
               seq_3_weight_shape,
               buf); // 卷积核形状
    // x = F.relu(x)
    relu_f32(buf0, buf0, 32 * 8 * 8);
    // x = F.max_pool2d(x, 2)  # (N, 32, 8, 8)->(N, 32, 4, 4)
    maxpool2d_f32(buf1, buf0, 8, 8, 32, 2);
    // x = torch.flatten(x, 1) # (N, 32, 4, 4)->(N, 512)
    // x = self.fc1(x)         # (N, 512)->(N, 1024)
    fc_f32(buf0, buf1, seq_7_weight, seq_7_bias, seq_7_weight_shape);
    // x = F.relu(x)
    relu_f32(buf0, buf0, 1024);
    // x = self.fc2(x)         # (N, 1024)->(N, 10)
    fc_f32(buf1, buf0, seq_10_weight, seq_10_bias, seq_10_weight_shape);
    // argmax
    argmax_f32(&res, buf1, 10);

    return res;
}

int main()
{
    const float* din;
    int res,err;
    err = 0;
    for (int n = 0; n < TEST_DATA_NUM; n++)
    {
        din = &test_x[n][0];
        res = calc((float*)din);
        err += res != test_y[n];
        printf("[INF] TEST: %d, OUT: %d, GT: %d %s\n", n, res, test_y[n],(res==test_y[n])?"":"******");
    }
    printf("[INF] #error: %d, ACC: %.2f%%\n", err, 100.0-(float)err / (float)TEST_DATA_NUM * 100.0);
	return 0;
}
