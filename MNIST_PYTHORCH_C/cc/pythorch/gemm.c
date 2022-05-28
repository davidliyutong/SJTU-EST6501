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

#ifdef __SSE__
#include <immintrin.h>
#include <emmintrin.h>
#endif

 /**
  * @brief 通用矩阵乘法dout = m * n
  *
  * @param dout
  * @param m
  * @param n
  * @param m_hgt M
  * @param m_wid K
  * @param n_hgt K
  * @param n_wid N
  * @return pythorch_err_t
  */
pythorch_err_t gemm_f32(float* dout,
                        float* m,
                        float* n,
                        int m_hgt,
                        int m_wid,
                        int n_hgt,
                        int n_wid) {

    if (m_wid != n_hgt) return PYTHORCH_ERR;

    memset(dout, 0, sizeof(float) * m_hgt * n_wid);

#if !defined(OPTIMIZE_GEMM)
    for (int i = 0; i < m_hgt; i++)
        for (int k = 0; k < n_hgt; k++)
            for (int j = 0; j < n_wid; j++)
                dout[i * n_wid + j] += m[i * m_wid + k] * n[k * n_wid + j];
#else
#if OPTIMIZE_GEMM == 1

    for (int i = 0; i < m_hgt; i++) {
        for (int k = 0; k < n_hgt; k++) {
            int j = 0;
#ifdef __AVX__
            __m256 ymm0 = _mm256_set1_ps(m[i * m_wid + k]);
            for (j = 0; j < ((n_wid)-8); j += 8) {
                __m256 ymm1 = _mm256_load_ps(n + (k * n_wid + j));
                __m256 ymm2 = _mm256_load_ps(dout + (i * n_wid + j));
                _mm256_store_ps(dout + (i * n_wid + j), _mm256_add_ps(ymm2, _mm256_mul_ps(ymm0, ymm1)));
            }
#endif
            for (; j < n_wid; j++) {
                dout[i * n_wid + j] += m[i * m_wid + k] * n[k * n_wid + j];
            }
        }
    }
#elif OPTIMIZE_GEMM == 2
#if defined(__AVX__)

    int i = 0, k = 0, j = 0;
    for (i = 0; i < m_hgt - 8; i += 8) {
        for (k = 0; k < m_wid - 8; k += 8) {
            __m256 dout0v, dout1v, dout2v, dout3v, dout4v, dout5v, dout6v, dout7v, n0b, n1v, n2v, n3v, n4v, n5v, n6v, n7v;
            for (j = 0; j < n_wid - 8; j += 8) {
                dout0v = _mm256_load_ps(&dout[(i + 0) * n_wid + j]);
                dout1v = _mm256_load_ps(&dout[(i + 1) * n_wid + j]);
                dout2v = _mm256_load_ps(&dout[(i + 2) * n_wid + j]);
                dout3v = _mm256_load_ps(&dout[(i + 3) * n_wid + j]);
                dout4v = _mm256_load_ps(&dout[(i + 4) * n_wid + j]);
                dout5v = _mm256_load_ps(&dout[(i + 5) * n_wid + j]);
                dout6v = _mm256_load_ps(&dout[(i + 6) * n_wid + j]);
                dout7v = _mm256_load_ps(&dout[(i + 7) * n_wid + j]);

                n0b = _mm256_load_ps(&n[(k + 0) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 0] * n0b;
                dout1v += m[(i + 1) * m_wid + k + 0] * n0b;
                dout2v += m[(i + 2) * m_wid + k + 0] * n0b;
                dout3v += m[(i + 3) * m_wid + k + 0] * n0b;
                dout4v += m[(i + 4) * m_wid + k + 0] * n0b;
                dout5v += m[(i + 5) * m_wid + k + 0] * n0b;
                dout6v += m[(i + 6) * m_wid + k + 0] * n0b;
                dout7v += m[(i + 7) * m_wid + k + 0] * n0b;

                n1v = _mm256_load_ps(&n[(k + 1) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 1] * n1v;
                dout1v += m[(i + 1) * m_wid + k + 1] * n1v;
                dout2v += m[(i + 2) * m_wid + k + 1] * n1v;
                dout3v += m[(i + 3) * m_wid + k + 1] * n1v;
                dout4v += m[(i + 4) * m_wid + k + 1] * n1v;
                dout5v += m[(i + 5) * m_wid + k + 1] * n1v;
                dout6v += m[(i + 6) * m_wid + k + 1] * n1v;
                dout7v += m[(i + 7) * m_wid + k + 1] * n1v;

                n2v = _mm256_load_ps(&n[(k + 2) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 2] * n2v;
                dout1v += m[(i + 1) * m_wid + k + 2] * n2v;
                dout2v += m[(i + 2) * m_wid + k + 2] * n2v;
                dout3v += m[(i + 3) * m_wid + k + 2] * n2v;
                dout4v += m[(i + 4) * m_wid + k + 2] * n2v;
                dout5v += m[(i + 5) * m_wid + k + 2] * n2v;
                dout6v += m[(i + 6) * m_wid + k + 2] * n2v;
                dout7v += m[(i + 7) * m_wid + k + 2] * n2v;

                n3v = _mm256_load_ps(&n[(k + 3) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 3] * n3v;
                dout1v += m[(i + 1) * m_wid + k + 3] * n3v;
                dout2v += m[(i + 2) * m_wid + k + 3] * n3v;
                dout3v += m[(i + 3) * m_wid + k + 3] * n3v;
                dout4v += m[(i + 4) * m_wid + k + 3] * n3v;
                dout5v += m[(i + 5) * m_wid + k + 3] * n3v;
                dout6v += m[(i + 6) * m_wid + k + 3] * n3v;
                dout7v += m[(i + 7) * m_wid + k + 3] * n3v;

                n4v = _mm256_load_ps(&n[(k + 4) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 4] * n4v;
                dout1v += m[(i + 1) * m_wid + k + 4] * n4v;
                dout2v += m[(i + 2) * m_wid + k + 4] * n4v;
                dout3v += m[(i + 3) * m_wid + k + 4] * n4v;
                dout4v += m[(i + 4) * m_wid + k + 4] * n4v;
                dout5v += m[(i + 5) * m_wid + k + 4] * n4v;
                dout6v += m[(i + 6) * m_wid + k + 4] * n4v;
                dout7v += m[(i + 7) * m_wid + k + 4] * n4v;

                n5v = _mm256_load_ps(&n[(k + 5) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 5] * n5v;
                dout1v += m[(i + 1) * m_wid + k + 5] * n5v;
                dout2v += m[(i + 2) * m_wid + k + 5] * n5v;
                dout3v += m[(i + 3) * m_wid + k + 5] * n5v;
                dout4v += m[(i + 4) * m_wid + k + 5] * n5v;
                dout5v += m[(i + 5) * m_wid + k + 5] * n5v;
                dout6v += m[(i + 6) * m_wid + k + 5] * n5v;
                dout7v += m[(i + 7) * m_wid + k + 5] * n5v;

                n6v = _mm256_load_ps(&n[(k + 6) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 6] * n6v;
                dout1v += m[(i + 1) * m_wid + k + 6] * n6v;
                dout2v += m[(i + 2) * m_wid + k + 6] * n6v;
                dout3v += m[(i + 3) * m_wid + k + 6] * n6v;
                dout4v += m[(i + 4) * m_wid + k + 6] * n6v;
                dout5v += m[(i + 5) * m_wid + k + 6] * n6v;
                dout6v += m[(i + 6) * m_wid + k + 6] * n6v;
                dout7v += m[(i + 7) * m_wid + k + 6] * n6v;

                n7v = _mm256_load_ps(&n[(k + 7) * n_wid + j]);
                dout0v += m[(i + 0) * m_wid + k + 7] * n7v;
                dout1v += m[(i + 1) * m_wid + k + 7] * n7v;
                dout2v += m[(i + 2) * m_wid + k + 7] * n7v;
                dout3v += m[(i + 3) * m_wid + k + 7] * n7v;
                dout4v += m[(i + 4) * m_wid + k + 7] * n7v;
                dout5v += m[(i + 5) * m_wid + k + 7] * n7v;
                dout6v += m[(i + 6) * m_wid + k + 7] * n7v;
                dout7v += m[(i + 7) * m_wid + k + 7] * n7v;

                _mm256_store_ps(&dout[(i + 0) * n_wid + j], dout0v);
                _mm256_store_ps(&dout[(i + 1) * n_wid + j], dout1v);
                _mm256_store_ps(&dout[(i + 2) * n_wid + j], dout2v);
                _mm256_store_ps(&dout[(i + 3) * n_wid + j], dout3v);
                _mm256_store_ps(&dout[(i + 4) * n_wid + j], dout4v);
                _mm256_store_ps(&dout[(i + 5) * n_wid + j], dout5v);
                _mm256_store_ps(&dout[(i + 6) * n_wid + j], dout6v);
                _mm256_store_ps(&dout[(i + 7) * n_wid + j], dout7v);
            }
            for (; j < n_wid; j++) {
                dout[i * n_wid + j] += m[i * m_wid + k + 0] * n[k + 0 * n_wid + j];
                dout[i * n_wid + j] += m[i * m_wid + k + 1] * n[k + 1 * n_wid + j];
                dout[i * n_wid + j] += m[i * m_wid + k + 2] * n[k + 2 * n_wid + j];
                dout[i * n_wid + j] += m[i * m_wid + k + 3] * n[k + 3 * n_wid + j];
                dout[i * n_wid + j] += m[i * m_wid + k + 4] * n[k + 4 * n_wid + j];
                dout[i * n_wid + j] += m[i * m_wid + k + 5] * n[k + 5 * n_wid + j];
                dout[i * n_wid + j] += m[i * m_wid + k + 6] * n[k + 6 * n_wid + j];
                dout[i * n_wid + j] += m[i * m_wid + k + 7] * n[k + 7 * n_wid + j];
            }
        }
        for (; k < m_wid; k++) {
            __m256 ymm0 = _mm256_set1_ps(m[i * m_wid + k]);
            for (j = 0; j < ((n_wid)-16); j += 16) {
                __m256 ymm1 = _mm256_load_ps(n + (k * n_wid + j));
                __m256 ymm2 = _mm256_load_ps(n + (k * n_wid + j + 8));
                __m256 ymm3 = _mm256_load_ps(dout + (i * n_wid + j));
                __m256 ymm4 = _mm256_load_ps(dout + (i * n_wid + j + 8));
                _mm256_store_ps(dout + (i * n_wid + j), _mm256_add_ps(ymm3, _mm256_mul_ps(ymm0, ymm1)));
                _mm256_store_ps(dout + (i * n_wid + j), _mm256_add_ps(ymm4, _mm256_mul_ps(ymm0, ymm2)));
            }
            for (; j < n_wid; j++) {
                dout[i * n_wid + j] += m[i * m_wid + k] * n[k * n_wid + j];
            }
        }
    } for (; i < m_hgt; i++) {
        for (int k = 0; k < n_hgt; k++) {
            int j = 0;
            __m256 ymm0 = _mm256_set1_ps(m[i * m_wid + k]);
            for (j = 0; j < ((n_wid)-16); j += 16) {
                __m256 ymm1 = _mm256_load_ps(n + (k * n_wid + j));
                __m256 ymm2 = _mm256_load_ps(n + (k * n_wid + j + 8));
                __m256 ymm3 = _mm256_load_ps(dout + (i * n_wid + j));
                __m256 ymm4 = _mm256_load_ps(dout + (i * n_wid + j + 8));
                _mm256_store_ps(dout + (i * n_wid + j), _mm256_add_ps(ymm3, _mm256_mul_ps(ymm0, ymm1)));
                _mm256_store_ps(dout + (i * n_wid + j), _mm256_add_ps(ymm4, _mm256_mul_ps(ymm0, ymm2)));
            }
            for (; j < n_wid; j++) {
                dout[i * n_wid + j] += m[i * m_wid + k] * n[k * n_wid + j];
            }
        }
    }
#else
#error("AVX not enabled")
#endif
#endif
#endif

    return PYTHORCH_OK;

}

