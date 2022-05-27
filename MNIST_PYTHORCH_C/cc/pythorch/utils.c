/**
 * @file utils.c
 * @author davidliyutong@sjtu.edu.cn
 * @brief 
 * @version 0.1
 * @date 2022-05-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "utils.h"

float calc_error(float* p1, float* p2, int size)
{
    float e = 0, v;
    for (int n = 0; n < size; n++)
    {
        v = p1[n] - p2[n];
        if (v < 0) v = -v;
        if (v > e) e=v;
    }
    return e;
}
