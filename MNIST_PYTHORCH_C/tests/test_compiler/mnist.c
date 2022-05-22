#include <stdio.h>
#include "pythorch/modules.h"
#include "calc_fn.h"
#include "test_data.h"

int run(float* din)
{
    int res;
    float vmax;
    float dout[10];
    calc_fn(din, dout);
    argmax_f32(&res, dout, 10);
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
        res = run((float*)din);
        err += res != test_y[n];
        printf("[INF] TEST: %d, OUT: %d, GT: %d %s\n", n, res, test_y[n],(res==test_y[n])?"":"******");
    }
    printf("[INF] #error: %d, ACC: %.2f%%\n", err, 100.0-(float)err / (float)TEST_DATA_NUM * 100.0);
	return 0;
}
