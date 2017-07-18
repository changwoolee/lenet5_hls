
//#include <lenet5/hw_layers/image_pool.h>
#include "../common.h"
#include "ap_int.h"
#define CONV_2_SIZE 25

float _tanh(float x);
float relu(float x);

#pragma SDS data access_pattern(input:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output:SEQUENTIAL)
//#pragma SDS data zero_copy(input,weights,bias,output)
void CONVOLUTION_LAYER_1(float input[image_Batch*32*32],
		float weights[6*5*5],
		float bias[6],
		float output[image_Batch*6*28*28]//, int init
		);

#pragma SDS data access_pattern(input:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output:SEQUENTIAL)
//#pragma SDS data zero_copy(input,weights,bias,output)
//#pragma SDS data copy(weights[0:kernel_size],bias[0:bias_size])
void CONVOLUTION_LAYER_2(float input[image_Batch*6*14*14],
		float weights[6*16*5*5],
		float bias[CONV_2_TYPE],
		float output[image_Batch*16*10*10]//, int init
		);

#pragma SDS data access_pattern(input:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output:SEQUENTIAL)
//#pragma SDS data zero_copy(input,weights,bias,output)
//#pragma SDS data copy(weights[0:kernel_size],bias[0:bias_size])
void CONVOLUTION_LAYER_3(float input[image_Batch*16*5*5],
		float weights[16*120*5*5],
		float bias[120],
		float output[image_Batch*120]//, int init
		);


