#include "../common.h"

#define CONV_2_SIZE 25

/*
void kernel_load(char *fileName, float * target)

{
	int read_i = 0;
	FILE* read_ptr;

	if (read_ptr = fopen(fileName, "rb"))
	{
		while (EOF != fscanf(read_ptr, "%f", &target[read_i]))
		{
			read_i++;
		}
	}
	fclose(read_ptr);

}
*/


//#pragma SDS data mem_attribute(input_feature:PHYSICAL_CONTIGUOUS,conv_kernel:PHYSICAL_CONTIGUOUS,conv_bias:PHYSICAL_CONTIGUOUS,output_feature:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
void CONVOLUTION_LAYER_1(float input_feature[image_Batch*INPUT_WH *INPUT_WH],
		float conv_kernel[CONV_1_TYPE*CONV_1_WH * CONV_1_WH],
		float conv_bias[CONV_1_TYPE],
		float output_feature[CONV_1_TYPE * image_Batch*CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH]);

//#pragma SDS data mem_attribute(input_feature:PHYSICAL_CONTIGUOUS,conv_kernel:PHYSICAL_CONTIGUOUS,conv_bias:PHYSICAL_CONTIGUOUS,output_feature:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input_feature:RANDOM,conv_kernel:SEQUENTIAL,conv_bias:SEQUENTIAL,output_feature:RANDOM)
void CONVOLUTION_LAYER_2(float input_feature[CONV_1_TYPE * image_Batch*CONV_2_INPUT_WH *CONV_2_INPUT_WH],
	float conv_kernel[CONV_2_TYPE*CONV_1_TYPE*CONV_2_WH * CONV_2_WH],
	float conv_bias[CONV_2_TYPE],
	float output_feature[CONV_2_TYPE * image_Batch*CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH]);

//#pragma SDS data mem_attribute(input_feature:PHYSICAL_CONTIGUOUS,conv_kernel:PHYSICAL_CONTIGUOUS,conv_bias:PHYSICAL_CONTIGUOUS,output_feature:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input_feature:SEQUENTIAL,conv_kernel:RANDOM,conv_bias:SEQUENTIAL,output_feature:SEQUENTIAL)
void CONVOLUTION_LAYER_3(float input_feature[CONV_2_TYPE*image_Batch*CONV_3_INPUT_WH *CONV_3_INPUT_WH],

		 float conv_kernel1[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3],
		 float conv_kernel2[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3],
		 float conv_kernel3[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3],
						 float conv_bias[CONV_3_TYPE],
						 float output_feature[image_Batch * CONV_3_TYPE]);

float _tanh(float x);
