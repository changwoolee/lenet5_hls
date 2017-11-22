
//#include <lenet5/hw_layers/image_pool.h>
#include "../common.h"
#include "ap_int.h"
#define CONV_2_SIZE 25

float _tanh(float x);
float relu(float x);

#pragma SDS data mem_attribute(input_feature:PHYSICAL_CONTIGUOUS,weights:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS,output_feature:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input_feature:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output_feature:SEQUENTIAL)
#pragma SDS data zero_copy(input_feature,weights,bias,output_feature)
void CONVOLUTION_LAYER_1(const float input_feature[image_Batch*CONV_1_INPUT_WH*CONV_1_INPUT_WH],
		const float weights[6*5*5],
		const float bias[6],
		float output_feature[image_Batch*6*CONV_1_OUTPUT_WH*CONV_1_OUTPUT_WH], int init
		);
#pragma SDS data mem_attribute(input_feature:PHYSICAL_CONTIGUOUS,weights:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS,output_feature:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input_feature:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output_feature:SEQUENTIAL)
#pragma SDS data zero_copy(input_feature,weights,bias,output_feature)
////#pragma SDS data copy(weights[0:kernel_size],bias[0:bias_size])
void CONVOLUTION_LAYER_2(const float input_feature[image_Batch*6*14*14],
		const float weights[6*16*5*5],
		const float bias[CONV_2_TYPE],
		float output_feature[image_Batch*16*10*10], int init
		);
#pragma SDS data mem_attribute(input_feature:PHYSICAL_CONTIGUOUS,weights:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS,output_feature:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input_feature:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output_feature:SEQUENTIAL)
#pragma SDS data zero_copy(input_feature,weights,bias,output_feature)
////#pragma SDS data copy(weights[0:kernel_size],bias[0:bias_size])
void CONVOLUTION_LAYER_3(const float input_feature[image_Batch*16*5*5],
		const float weights[16*120*5*5],
		const float bias[120],
		float output_feature[image_Batch*120], int init
		);


