
//#include <lenet5/hw_layers/image_pool.h>
#include "../common.h"
#include "ap_int.h"
#define CONV_2_SIZE 25

float _tanh(float x);
float relu(float x);
/*
void copy_input(float* DRAM, float buffer[1][32][32]);
void copy_weights(float Wconv1[6*25], float bconv1[6],
			float Wconv2[6*16*25], float bconv2[16],
			float Wconv3[16*120*25], float bconv3[120],
			float W1BRAM[6][5][5], float b1BRAM[6],
			float W2BRAM[6][16][5][5], float b2BRAM[16],
			float W3BRAM[16][120][5][5], float b3BRAM[120]
			);
void store_output(float buffer[1][120], float* b3BRAM, float* DRAM);
*/
#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature,conv_kernel,conv_bias,output_feature)
void CONVOLUTION_LAYER_1(float input[image_Batch*32*32],
		float weights[6*5*5],
		float bias[6],
		float output[image_Batch*6*28*28], ap_uint<1> init
		);

#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature,conv_kernel,conv_bias,output_feature)
//#pragma SDS data copy(conv_kernel[0:kernel_size],conv_bias[0:bias_size])
void CONVOLUTION_LAYER_2(float input[image_Batch*6*14*14],
		float weights[6*16*5*5],
		float bias[CONV_2_TYPE],
		float output[image_Batch*16*10*10], ap_uint<1> init
		);

#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature,conv_kernel,conv_bias,output_feature)
//#pragma SDS data copy(conv_kernel[0:kernel_size],conv_bias[0:bias_size])
void CONVOLUTION_LAYER_3(float input[image_Batch*16*5*5],
		float weights[16*120*5*5],
		float bias[120],
		float output[image_Batch*120], ap_uint<1> init
		);


