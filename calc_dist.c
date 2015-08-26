/*
 * @Author: Conrad Shiao, Grace Kim
 * CS61C Sp14 Project 3 Part 1: YOUR CODE HERE
 *
 * You MUST implement the calc_min_dist() function in this file.
 *
 * You do not need to implement/use the swap(), flip_horizontal(), transpose(), or rotate_ccw_90()
 * functions, but you may find them useful. Feel free to define additional helper functions.
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "digit_rec.h"
#include "utils.h"
#include <omp.h>
#include <emmintrin.h>


void swap(float *x, float *y);
void flip_vertical(float *arr, int width);
void flip_horizontal(float *arr, int width);
void transpose(float *arr, int width);
void rotate_cw_90(float *arr, int width);
void rotate_ccw_90(float *arr, int width);
float euclidean_dist(float *image, int i_width, int i_height, float *template, int t_width);
float min(float x, float y);

/* Swaps the values pointed to by the pointers X and Y. */
void swap(float *x, float *y) {
    float temp = *x;
    *x = *y;
    *y = temp;
}

/* Flips the elements of a square array ARR of length WIDTH across the x-axis. */
void flip_vertical(float *arr, int width) {
    for (int j = 0; j < width / 2; j++) {
        for (int k = 0; k < width / 8 * 8; k += 8) {
            __m128 up = _mm_loadu_ps(arr + j * width + k);
            __m128 down = _mm_loadu_ps(arr + width * (width - j - 1) + k);
            _mm_storeu_ps(arr + j * width + k, down);
            _mm_storeu_ps(arr + width * (width - j - 1) + k, up);
            __m128 up1 = _mm_loadu_ps(arr + j * width + k + 4);
            __m128 down1 = _mm_loadu_ps(arr + width * (width - j - 1) + k + 4);
            _mm_storeu_ps(arr + j * width + k + 4, down1);
            _mm_storeu_ps(arr + width * (width - j - 1) + k + 4, up1);
        }
        for (int k = width / 8 * 8; k < width; k++) {
            swap(arr + j * width + k, arr + width * (width - j - 1) + k);
        }
    }
}

/* Flips the elements of a square array ARR across the y-axis. */
void flip_horizontal(float *arr, int width) {
	 int j, k;
	for (j = 0; j < width; j++) {
		for (k = 0; k < (width / 2) / 8 * 8; k += 8) {
            __m128 a = _mm_loadu_ps(arr + j * width + k);
            __m128 b = _mm_loadu_ps(arr + (j + 1) * width - k - 4);
			__m128 c = _mm_loadu_ps(arr + j * width + k + 4);
			__m128 d = _mm_loadu_ps(arr + (j + 1) * width - k - 8);

			_mm_storeu_ps(arr + j * width + k, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3)));
			_mm_storeu_ps(arr + (j + 1) * width - k - 4, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0,1,2,3)));
			_mm_storeu_ps(arr + j * width + k + 4, _mm_shuffle_ps(d, d, _MM_SHUFFLE(0, 1, 2, 3)));
			_mm_storeu_ps(arr + (j + 1) * width - k - 8, _mm_shuffle_ps(c, c, _MM_SHUFFLE(0, 1, 2, 3)));
           /* _mm_storeu_ps(arr + j * width + k + 8, _mm_shuffle_ps(f, f, _MM_SHUFFLE(0, 1, 2, 3)));
            _mm_storeu_ps(arr + (j + 1) * width - k - 12, _mm_shuffle_ps(e, e, _MM_SHUFFLE(0,1,2,3)));
            _mm_storeu_ps(arr + j * width + k + 12, _mm_shuffle_ps(h, h, _MM_SHUFFLE(0, 1, 2, 3)));
            _mm_storeu_ps(arr + (j + 1) * width - k - 16, _mm_shuffle_ps(g, g, _MM_SHUFFLE(0, 1, 2, 3))); */
        }
        for (k = (width / 2) / 8 * 8; k < width / 2; k++) {
            swap(arr + j * width + k, arr + (j + 1) * width - 1 - k);
        }
    }
}

/* Transposes the square array ARR. */
void transpose(float *arr, int width) {
	int r, c;
    for (r = 0; r < width; r++) {
        for (c = 0; c < r / 8 * 8; c += 8) { // analyzing lower trianglular matrix of ARR
            swap(arr + r * width + c, arr + c * width + r);
            swap(arr + r * width + (c + 1), arr + (c + 1) * width + r);
            swap(arr + r * width + (c + 2), arr + (c + 2) * width + r);
            swap(arr + r * width + (c + 3), arr + (c + 3) * width + r);
            swap(arr + r * width + (c + 4), arr + (c + 4) * width + r);
            swap(arr + r * width + (c + 5), arr + (c + 5) * width + r);
            swap(arr + r * width + (c + 6), arr + (c + 6) * width + r);
            swap(arr + r * width + (c + 7), arr + (c + 7) * width + r);
        }
        for (c = r / 8 * 8; c < r; c++) {
            swap(arr + r * width + c, arr + c * width + r);
        }
    }
}

/* Rotates the square array ARR of length WIDTH by 90 degrees counterclockwise. */
void rotate_ccw_90(float *arr, int width) {
    flip_horizontal(arr, width);
    transpose(arr, width);
}
/* Rotates the square array ARR of length WIDTH by 90 degrees clockwise. */
void rotate_cw_90(float *arr, int width) {
    flip_vertical(arr, width);
    transpose(arr, width);
}

/* Calculates the minimum euclidean distance found for a particular 
orientation of IMAGE, which has dimension I_WIDTH * I_HEIGHT,
on the TEMPLATE, of square dimension T_WIDTH. */
float euclidean_dist(float *image, int i_width, int i_height, float *template, int t_width) {
    float min_dist = FLT_MAX;
    for (int y = 0; y <= i_height - t_width; y++) {
        for (int x = 0; x <= i_width - t_width; x++) {
            float tail = 0.0;
            __m128 curr_dist = _mm_setzero_ps();
            float sum[4] = {0.0, 0.0, 0.0, 0.0};
            for (int row = y; row < (y + t_width); row++) {
                for (int col = x; col < (x + t_width) / 8 * 8; col += 8) {
                    __m128 template_temp = _mm_loadu_ps(template + (col - x) + (row - y) * t_width);
                    __m128 image_temp = _mm_loadu_ps(image + i_width * row + col);
                    __m128 template_temp2 = _mm_loadu_ps(template + (col + 4 - x) + (row - y) * t_width);
                    __m128 image_temp2 = _mm_loadu_ps(image + i_width * row + col + 4);
                    __m128 subtracting = _mm_sub_ps(template_temp, image_temp);
                    __m128 subtracting2 = _mm_sub_ps(template_temp2, image_temp2);
                    curr_dist = _mm_add_ps(curr_dist, _mm_mul_ps(subtracting, subtracting));
                    curr_dist = _mm_add_ps(curr_dist, _mm_mul_ps(subtracting2, subtracting2));
                }
                for (int col = (x + t_width) / 8 * 8; col < x + t_width; col++) {
                    float template_temp = template[(col - x) + (row - y) * t_width];
                    float image_temp = image[i_width * row + col];
                    tail += (template_temp - image_temp) * (template_temp - image_temp);
                }
            }
            _mm_storeu_ps(sum, curr_dist);
            min_dist = min(min_dist, tail + sum[0] + sum[1] + sum[2] + sum[3]);
        }
    }
    return min_dist;
}

/* Returns the minimum float value of {a, b}. */
float min(float a, float b) {
    return (a < b) ? a : b;
}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist(float *image, int i_width, int i_height, float *template, int t_width) {
    float storing[8];

    float *template_norm = malloc(sizeof(float) * t_width * t_width);
    float *template_f = malloc(sizeof(float) * t_width * t_width);
    float *template_r = malloc(sizeof(float) * t_width * t_width);
    float *template_fr = malloc(sizeof(float) * t_width * t_width);
    float *template_rr = malloc(sizeof(float) * t_width * t_width);
    float *template_frr = malloc(sizeof(float) * t_width * t_width);
    float *template_rrr = malloc(sizeof(float) * t_width * t_width);
    float *template_frrr = malloc(sizeof(float) * t_width * t_width);

    # pragma omp parallel for collapse(2)
    for (int r = 0; r < t_width; r++) {
        for (int c = 0; c < t_width; c++) {
            float temp = template[r * t_width + c];
            template_norm[r * t_width + c] = template_f[r * t_width + c] = temp;
            template_r[r * t_width + c] = template_fr[r * t_width + c] = temp;
            template_rr[r * t_width + c] = template_frr[r * t_width + c] = temp;
            template_rrr[r * t_width + c] = template_frrr[r * t_width + c] = temp;
        }
    }
    int i;
    # pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        switch(i) {
            case 0:
                {
                    storing[i] = euclidean_dist(image, i_width, i_height, template_norm, t_width);
                    break;
                }
            case 1:
                {
                    flip_horizontal(template_f, t_width);
                    storing[i] = euclidean_dist(image, i_width, i_height, template_f, t_width);
                    break;
                }
            case 2: 
                {
                    rotate_ccw_90(template_r, t_width);
                    storing[i] = euclidean_dist(image, i_width, i_height, template_r, t_width);
                    break;
                }
            case 3:
                {
                    rotate_ccw_90(template_rr, t_width);
                    rotate_ccw_90(template_rr, t_width);
                    storing[i] = euclidean_dist(image, i_width, i_height, template_rr, t_width);
                    break;
                }
            case 4:
                {
                    rotate_cw_90(template_rrr, t_width);
                    storing[i] = euclidean_dist(image, i_width, i_height, template_rrr, t_width);
                    break;
                }
            case 5:
                {
                    rotate_ccw_90(template_fr, t_width);
                    flip_horizontal(template_fr, t_width);
                    storing[i] = euclidean_dist(image, i_width, i_height, template_fr, t_width);
                    break;
                }
            case 6:
                {
                    rotate_ccw_90(template_frr, t_width);
                    rotate_ccw_90(template_frr, t_width);
                    flip_horizontal(template_frr, t_width);
                    storing[i] = euclidean_dist(image, i_width, i_height, template_frr, t_width);
                    break;
                }
            case 7:
                {
                    rotate_cw_90(template_frrr, t_width);
                    flip_horizontal(template_frrr, t_width);
                    storing[i] = euclidean_dist(image, i_width, i_height, template_frrr, t_width);
                    break;
                }
        }
    }
    float min_dist = storing[0];
    for (i = 1; i < 8; i++) {
        min_dist = min(min_dist, storing[i]);
    }
    free(template_norm);
    free(template_f);
    free(template_r);
    free(template_rr);
    free(template_rrr);
    free(template_fr);
    free(template_frr);
    free(template_frrr); 

    return min_dist;
}
