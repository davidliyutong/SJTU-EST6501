/**
 * @file utils.h
 * @author davidliyutong@sjtu.edu.cn
 * @brief
 * @version 0.1
 * @date 2022-05-24
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

typedef enum {
    PYTHORCH_OK,
    PYTHORCH_ERR
} pythorch_err_t;

float calc_error(float* p1, float* p2, int size);