#include "../common.h"

#define CONV_2_SIZE 25



#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature,conv_kernel,conv_bias,output_feature)
void CONVOLUTION_LAYER_1(float input_feature[image_Batch*INPUT_WH *INPUT_WH],
		float conv_kernel[CONV_1_TYPE*25],
		float conv_bias[CONV_1_TYPE],
		float output_feature[image_Batch*CONV_1_TYPE*CONV_1_OUTPUT_SIZE]
		);

#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature,conv_kernel,conv_bias,output_feature)
//#pragma SDS data copy(conv_kernel[0:kernel_size],conv_bias[0:bias_size])
void CONVOLUTION_LAYER_2(float input_feature[CONV_1_TYPE * image_Batch*CONV_2_INPUT_WH *CONV_2_INPUT_WH],
	float conv_kernel[CONV_2_TYPE*CONV_1_TYPE*CONV_2_WH * CONV_2_WH],
	float conv_bias[CONV_2_TYPE],
	float output_feature[CONV_2_TYPE * image_Batch*CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH]
	);

#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature,conv_kernel,conv_bias,output_feature)
//#pragma SDS data copy(conv_kernel[0:kernel_size],conv_bias[0:bias_size])
void CONVOLUTION_LAYER_3(float input_feature[CONV_2_TYPE*image_Batch*CONV_3_INPUT_WH *CONV_3_INPUT_WH],
						float conv_kernel[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH],
						//float conv_kernel2[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/2],
						float conv_bias[CONV_3_TYPE],
						 float output_feature[image_Batch*CONV_3_TYPE]
						 );
//void CONV3_PE(float input[image_Batch][16][5][5], float kernel[16][120][5][5], float bias[120], float output[image_Batch][120]);
float _tanh(float x);
float relu(float x);
