/**
 * @file gemm.h
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

pythorch_err_t gemm_f32(float* dout,
                        float* m,
                        float* n,
                        int m_hgt,
                        int m_wid,
                        int n_hgt,
                        int n_wid);