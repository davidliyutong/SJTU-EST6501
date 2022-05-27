/**
 * @file gemm.c
 * @author davidliyutong@sjtu.edu.cn
 * @brief 
 * @version 0.1
 * @date 2022-05-27
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <memory.h>
#include "pythorch.h"
#include "gemm.h"

pythorch_err_t gemm_f32(float* dout,
                        float* m,
                        float* n,
                        int m_hgt,
                        int m_wid,
                        int n_hgt,
                        int n_wid) {

    if (m_wid != n_hgt) return PYTHORCH_ERR;
    int k = m_wid;

    memset(dout, 0, sizeof(float) * m_hgt * n_wid);
    for (int i = 0; i < m_hgt; i++)
        for (int j = 0; j < n_wid; j++)
            for (int k = 0; k < n_hgt; k++)
                dout[i * n_wid + j] += m[i * m_wid + k] * n[k * n_wid + j];
    
    return PYTHORCH_OK;

}