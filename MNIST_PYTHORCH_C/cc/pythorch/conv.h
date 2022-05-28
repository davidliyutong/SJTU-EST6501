/**
 * @file conv.h
 * @author davidliyutong@sjtu.edu.cn
 * @brief
 * @version 0.1
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include "utils.h"

pythorch_err_t im2col(float* data_im,
                      int data_num_c,
                      int data_hgt,
                      int data_wid,
                      int k_hgt,
                      int k_wid,
                      int stride,
                      int pad,
                      float* data_col);

int im2col_get_buf_size(int data_num_c,
                        int data_hgt,
                        int data_wid,
                        int k_hgt,
                        int k_wid,
                        int stride,
                        int pad);

pythorch_err_t col2im(float* data_col,
                      int data_num_c,
                      int data_hgt,
                      int data_wid,
                      int k_hgt,
                      int k_wid,
                      int stride,
                      int pad,
                      float* data_im);