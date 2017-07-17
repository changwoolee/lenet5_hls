/*
 * image_pool_sw.h
 *
 *  Created on: 2017. 5. 21.
 *      Author: woobes
 */

#ifndef SRC_LENET5_SW_LAYERS_IMAGE_POOL_SW_H_
#define SRC_LENET5_SW_LAYERS_IMAGE_POOL_SW_H_
#include "../common.h"

void POOLING_LAYER_1_SW(float* src, float* pool_kernel, float* pool_bias, float* dst,int scale_factor=2)
{
	int row, col, row_sub, col_sub, batch_cnt, depth;
	float value;
	for (batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++)
	{
		for (depth = 0; depth < POOL_1_TYPE; depth++)
		{
			for (row = 0; row < POOL_1_OUTPUT_WH; row++)
			{
				for (col = 0; col < POOL_1_OUTPUT_WH; col++)
				{
					// Computation of Pooling
					value = src[(depth + POOL_1_TYPE * batch_cnt)*POOL_1_INPUT_SIZE + (row * 2) * POOL_1_INPUT_WH + (col * 2)]// * pool_kernel[depth*POOL_1_SIZE+0]
						+ src[(depth + POOL_1_TYPE * batch_cnt)*POOL_1_INPUT_SIZE + (row * 2) * POOL_1_INPUT_WH + (col * 2 + 1)]// * pool_kernel[depth*POOL_1_SIZE+1]
						+ src[(depth + POOL_1_TYPE * batch_cnt)*POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2)]// * pool_kernel[depth*POOL_1_SIZE+2]
						+ src[(depth + POOL_1_TYPE * batch_cnt)*POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2 + 1)];// * pool_kernel[depth*POOL_1_SIZE+3];

					float weight = pool_kernel[depth]*0.25;
					value *= weight;
					value += pool_bias[depth];
					// Activation function
					dst[(batch_cnt * POOL_1_TYPE + depth)*POOL_1_OUTPUT_SIZE + row * POOL_1_OUTPUT_WH + col] = tanhf(value);
				}
			}
		}
	}
}

void POOLING_LAYER_2_SW(float* src, float* pool_kernel, float* pool_bias, float* dst,int scale_factor=2)
{
	int row, col, row_sub, col_sub, batch_cnt, depth;
	float value;
	for (batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++)
	{
		for (depth = 0; depth < POOL_2_TYPE; depth++)
		{
			for (row = 0; row < POOL_2_OUTPUT_WH; row++)
			{
				for (col = 0; col < POOL_2_OUTPUT_WH; col++)
				{
					// Computation of Pooling
					value = src[(depth + POOL_2_TYPE * batch_cnt)*POOL_2_INPUT_SIZE + (row * 2) * POOL_2_INPUT_WH + (col * 2)]// * pool_kernel[depth*POOL_2_SIZE+0]
						+ src[(depth + POOL_2_TYPE * batch_cnt)*POOL_2_INPUT_SIZE + (row * 2) * POOL_2_INPUT_WH + (col * 2 + 1)]// * pool_kernel[depth*POOL_2_SIZE+1]
						+ src[(depth + POOL_2_TYPE * batch_cnt)*POOL_2_INPUT_SIZE + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2)]// * pool_kernel[depth*POOL_2_SIZE+2]
						+ src[(depth + POOL_2_TYPE * batch_cnt)*POOL_2_INPUT_SIZE + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2 + 1)];// * pool_kernel[depth*POOL_2_SIZE+3];

					float weight = pool_kernel[depth]*0.25;
					value *= weight;

					// Activation function
					dst[(batch_cnt * POOL_2_TYPE + depth)*POOL_2_OUTPUT_SIZE + row * POOL_2_OUTPUT_WH + col] = tanhf(value + pool_bias[depth]);
				}
			}
		}
	}
}


void MAXPOOL_1_SW(float* src, float* dst){
	for(int batch=0;batch<image_Batch;batch++){
		for(int depth=0;depth<POOL_1_TYPE;depth++){
			for(int row=0;row<POOL_1_OUTPUT_WH;row++){
				for(int col=0;col<POOL_1_OUTPUT_WH;col++){
					float max=-FLT_MAX;
					for(int row_w=0;row_w<2;row_w++){
						for(int col_w=0;col_w<2;col_w++){
							if(src[batch*POOL_1_TYPE*POOL_1_INPUT_SIZE + depth*POOL_1_INPUT_SIZE +
								(2*row+row_w)*POOL_1_INPUT_WH + col*2+col_w]>max){
								max = src[batch*POOL_1_TYPE*POOL_1_INPUT_SIZE + depth*POOL_1_INPUT_SIZE +
											(2*row+row_w)*POOL_1_INPUT_WH + col*2+col_w];
							}
						}
					}
					dst[batch*POOL_1_TYPE*POOL_1_OUTPUT_SIZE + depth*POOL_1_OUTPUT_SIZE + row*POOL_1_OUTPUT_WH + col] = max;
				}
			}
		}
	}
}
void MAXPOOL_2_SW(float* src, float* dst){
	for(int batch=0;batch<image_Batch;batch++){
		for(int depth=0;depth<POOL_2_TYPE;depth++){
			for(int row=0;row<POOL_2_OUTPUT_WH;row++){
				for(int col=0;col<POOL_2_OUTPUT_WH;col++){
					float max=-FLT_MAX;
					for(int row_w=0;row_w<2;row_w++){
						for(int col_w=0;col_w<2;col_w++){
							if(src[batch*POOL_2_TYPE*POOL_2_INPUT_SIZE + depth*POOL_2_INPUT_SIZE +
								(2*row+row_w)*POOL_2_INPUT_WH + col*2+col_w]>max){
								max = src[batch*POOL_2_TYPE*POOL_2_INPUT_SIZE + depth*POOL_2_INPUT_SIZE +
											(2*row+row_w)*POOL_2_INPUT_WH + col*2+col_w];
							}
						}
					}
					dst[batch*POOL_2_TYPE*POOL_2_OUTPUT_SIZE + depth*POOL_2_OUTPUT_SIZE + row*POOL_2_OUTPUT_WH + col] = max;
				}
			}
		}
	}
}
#endif /* SRC_LENET5_SW_LAYERS_IMAGE_POOL_SW_H_ */
