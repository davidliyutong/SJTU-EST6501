/**
 * @file conv.c
 * @author davidliyutong@sjtu.edu.cn
 * @brief
 * @version 0.1
 * @date 2022-05-27
 *
 *
 * @ref https://github.com/pjreddie/darknet/blob/master/src/col2im.c
 * @ref https://github.com/pjreddie/darknet/blob/master/src/im2col.c
 */

#include "utils.h"
#include "conv.h"

float im2col_get_pixel(float* im,
                       int im_hgt,
                       int im_wid,
                       int im_num_c, // 没有用到
                       int row,
                       int col,
                       int channel,
                       int pad) {
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= im_hgt || col >= im_wid) return 0.;
    return im[(channel * im_hgt + row) * im_wid + col];
}

int im2col_get_buf_size(int data_num_c,
                        int data_hgt,
                        int data_wid,
                        int k_hgt,
                        int k_wid,
                        int stride,
                        int pad) {
    int height_col = (data_hgt + 2 * pad - k_hgt) / stride + 1;
    int width_col = (data_wid + 2 * pad - k_wid) / stride + 1;
    int channels_col = data_num_c * k_hgt * k_wid;
    return height_col * width_col * channels_col;
}

pythorch_err_t im2col(float* data_im,
                      int data_num_c,
                      int data_hgt,
                      int data_wid,
                      int k_hgt,
                      int k_wid,
                      int stride,
                      int pad,
                      float* data_col) {
    int c, h, w;
    int height_col = (data_hgt + 2 * pad - k_hgt) / stride + 1;
    int width_col = (data_wid + 2 * pad - k_wid) / stride + 1;
    int channels_col = data_num_c * k_hgt * k_wid;

    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % k_wid;           // 卷积核的位置
        int h_offset = (c / k_wid) % k_hgt; // 卷积核的位置
        int channel = c / k_wid / k_hgt;       // 输入通道

        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int row = h_offset + h * stride; // im矩阵的位置
                int col = w_offset + w * stride; // im矩阵的位置
                int col_index = (c * height_col + h) * width_col + w; // im矩阵的下标
                data_col[col_index] = data_im[(channel * data_hgt + (row - pad)) * data_wid + (col - pad)];
            }
        }
    }

    return PYTHORCH_OK;
}

void col2im_add_pixel(float* im,
                      int im_hgt,
                      int im_wid,
                      int channels,
                      int row,
                      int col,
                      int channel,
                      int pad,
                      float val) {
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= im_hgt || col >= im_wid) return;

    im[(channel * im_hgt + row) * im_wid + col] += val;
}

pythorch_err_t col2im(float* data_col,
                      int data_num_c,
                      int data_hgt,
                      int data_wid,
                      int k_hgt,
                      int k_wid,
                      int stride,
                      int pad,
                      float* data_im) {
    int c, h, w;
    int height_col = (data_hgt + 2 * pad - k_hgt) / stride + 1;
    int width_col = (data_wid + 2 * pad - k_wid) / stride + 1;
    int channels_col = data_num_c * k_hgt * k_wid;

    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % k_wid;
        int h_offset = (c / k_wid) % k_hgt;
        int channel = c / k_wid / k_hgt;

        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int row = h_offset + h * stride;
                int col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                data_im[(channel * data_hgt + (row - pad)) * data_wid + (col - pad)] += val;
            }
        }
    }

    return PYTHORCH_OK;
}
