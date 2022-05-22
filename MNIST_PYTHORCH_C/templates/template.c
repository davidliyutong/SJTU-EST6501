typedef enum {
    PYTHORCH_OK,
    PYTHORCH_ERR
} pythorch_err_t;

#define PYTHORCH_MODEL_OUTPUT_DTYPE int


pythorch_err_t forward(float* din, PYTHORCH_MODEL_OUTPUT_DTYPE* dout) {
    pythorch_err_t err;

    /** Functions **/


}


#define PYTHORCH_CONV2D_OP conv2d_f32
#define PYTHORCH_CONV2D_ID_IN_HGT 24
#define PYTHORCH_CONV2D_ID_IN_WID 24
#define PYTHORCH_CONV2D_ID_KER 24
#define PYTHORCH_CONV2D_ID_BIAS 24
#define PYTHORCH_CONV2D_ID_KER_SHAPE 24
// ID to be replaced
pythorch_err_t conv2d_call(float* dout, float* din) {
    pythorch_err_t err;
    err = PYTHORCH_CONV2D_OP(dout,
                             din,
                             PYTHORCH_CONV2D_ID_IN_HGT, PYTHORCH_CONV2D_ID_IN_WID,
                             PYTHORCH_CONV2D_ID_KER,PYTHORCH_CONV2D_ID_BIAS,
                             PYTHORCH_CONV2D_ID_KER_SHAPE);
    return err;
}

#define PYTHORCH_LINEAR_OP linear_f32
#define PYTHORCH_LINEAR_ID_WEIGHT 24
#define PYTHORCH_LINEAR_ID_BIAS 24
#define PYTHORCH_LINEAR_ID_SHAPE 24
// ID to be replaced
pythorch_err_t linear_call(float* dout, float* din) {
    pythorch_err_t err;
    err = PYTHORCH_LINEAR_OP(dout,
                             din,
                             PYTHORCH_LINEAR_ID_WEIGHT, PYTHORCH_LINEAR_ID_BIAS, PYTHORCH_LINEAR_ID_SHAPE);
    return err;
}

#define PYTHORCH_MAXPOOL2D_OP maxpool2d_f32
#define PYTHORCH_MAXPOOL2D_ID_IN_HGT 24
#define PYTHORCH_MAXPOOL2D_ID_IN_WID 24
#define PYTHORCH_MAXPOOL2D_ID_NUM_C 24
#define PYTHORCH_MAXPOOL2D_ID_K_SIZE 24
// ID to be replaced
pythorch_err_t maxpool2d_call(float* dout, float* din) {
    pythorch_err_t err;
    err = PYTHORCH_MAXPOOL2D_OP(dout,
                                din,
                                PYTHORCH_MAXPOOL2D_ID_IN_HGT, PYTHORCH_MAXPOOL2D_ID_IN_WID, PYTHORCH_MAXPOOL2D_ID_NUM_C,
                                PYTHORCH_MAXPOOL2D_ID_K_SIZE);
    return err;
}

#define PYTHORCH_MAXPOOL2D_OP maxpool2d_f32
#define PYTHORCH_MAXPOOL2D_ID_IN_HGT 24
#define PYTHORCH_MAXPOOL2D_ID_IN_WID 24
#define PYTHORCH_MAXPOOL2D_ID_NUM_C 24
#define PYTHORCH_MAXPOOL2D_ID_K_SIZE 24
// ID to be replaced
pythorch_err_t maxpool2d_call(float* dout, float* din) {
    pythorch_err_t err;
    err = PYTHORCH_MAXPOOL2D_OP(dout,
                                din,
                                PYTHORCH_MAXPOOL2D_ID_IN_HGT, PYTHORCH_MAXPOOL2D_ID_IN_WID, PYTHORCH_MAXPOOL2D_ID_NUM_C,
                                PYTHORCH_MAXPOOL2D_ID_K_SIZE);
    return err;
}

#define PYTHORCH_RELU_OP relu_f32
#define PYTHORCH_RELU_SIZE 24
// ID to be replaced
pythorch_err_t relu_call(float* dout, float* din) {
    pythorch_err_t err;
    err = PYTHORCH_RELU_OP(dout,
                           din,
                           PYTHORCH_RELU_SIZE);
    return err;
}