#include <lenet5/hw_layers/image_convolution.h>




void CONVOLUTION_LAYER_1(float input_feature[image_Batch*INPUT_WH *INPUT_WH],
		float conv_kernel[CONV_1_TYPE*CONV_1_WH * CONV_1_WH],
		float conv_bias[CONV_1_TYPE],
		float output_feature[CONV_1_TYPE * image_Batch*CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH])
{

	int col, row, col_f, row_f;
	int depth_out, batch_cnt;

	float input[image_Batch][INPUT_WH][INPUT_WH];
#pragma HLS array_partition variable=input cyclic factor=2 dim=2
#pragma HLS array_partition variable=input cyclic factor=2 dim=3
	float kernel[CONV_1_TYPE][CONV_1_WH*CONV_1_WH];
#pragma HLS array_partition variable=kernel cyclic factor=25 dim=2

	float bias[CONV_1_TYPE];
#pragma HLS array_partition variable=bias complete dim=0

	float output_buffer[image_Batch*CONV_1_TYPE*CONV_1_OUTPUT_SIZE];



	copy_kernel_1:
	for(int i=0;i<CONV_1_TYPE;i++){
		copy_kernel_2:
		for(int j=0;j<CONV_1_SIZE;j++){
#pragma HLS unroll factor=5
			kernel[i][j] = conv_kernel[i*CONV_1_SIZE+j];
			/*
			copy_kernel_3:
			for(int k=0;k<CONV_1_WH;k++){
#pragma HLS unroll
#pragma pipeline II=1
				kernel[i][j][k] = conv_kernel[i*CONV_1_SIZE+j*CONV_1_WH + k];
			}*/
		}
	}
	copy_input_1:
	for(int batch_cnt=0;batch_cnt<image_Batch;batch_cnt++){
		copy_input_2 :
		for(int i=0;i<INPUT_WH;i++){
			copy_input_3 :
			for(int j=0;j<INPUT_WH;j++){
#pragma HLS unroll factor=4
				input[batch_cnt][i][j] = input_feature[batch_cnt*INPUT_WH*INPUT_WH+i*INPUT_WH + j];
			}
		}
	}


	copy_bias:
	for(int i=0;i<CONV_1_TYPE;i++){
#pragma HLS pipeline
		bias[i] = conv_bias[i];
	}


	BATCH :
	for(batch_cnt=0; batch_cnt<image_Batch; batch_cnt++) {

		DEPTH_OUT :
		for (depth_out = 0; depth_out < CONV_1_TYPE; depth_out++) {
			ROW :
			for (row = 0; row < CONV_1_OUTPUT_WH; row++) {
//#pragma HLS unroll factor=2
				COL :
				for (col = 0; col < CONV_1_OUTPUT_WH; col++) {
//#pragma HLS unroll factor=2
#pragma HLS pipeline II=5
					float mult[CONV_1_SIZE];
#pragma HLS array_partition variable=mult complete dim=0
					float acc=0;

					// Multiplication
					for(int i=0;i<CONV_1_WH;i++){
#pragma HLS unroll
						for(int j=0;j<CONV_1_WH;j++){
#pragma HLS unroll
							mult[i*5+j] = input[batch_cnt][row+i][col+j]*kernel[depth_out][i*5+j];
						}
					}
					
					// Accumulation
					Accumulate:
					for(int i=0;i<CONV_1_SIZE;i++)
#pragma HLS unroll
						acc+=mult[i];

					output_buffer[(batch_cnt*CONV_1_TYPE + depth_out)*CONV_1_OUTPUT_SIZE +
								  CONV_1_OUTPUT_WH * row + col] = (acc+bias[depth_out]);

				}
			}
		}
	}
	
	copy_output:
	for(int i=0;i<image_Batch*CONV_1_TYPE*CONV_1_OUTPUT_SIZE;i++){
#pragma HLS pipeline
		output_feature[i] = _tanh(output_buffer[i]);
	}
}

float _tanh(float x){
#pragma HLS INLINE
	float exp2x = 2*exp(2*x)+1;
	return (exp2x-2)/(exp2x);
	//return sinhf(x)/coshf(x);
}


// Convolution Layer 2
// Function by Batch_size(10)
// Input_feature_map[6][14x14],  Conv_kernel[16][6][25], Bias[16], Output_feature_map[16][10x10]
void CONVOLUTION_LAYER_2(float input_feature[CONV_1_TYPE * image_Batch*CONV_2_INPUT_WH *CONV_2_INPUT_WH],
	float conv_kernel[CONV_2_TYPE*CONV_1_TYPE*CONV_2_WH * CONV_2_WH],
	float conv_bias[CONV_2_TYPE],
	float output_feature[CONV_2_TYPE * image_Batch*CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH])
{
	// Connection Table for Dummy Operation
/*
	3 Input feature map (6)
	----------------------------------------
	{ 1, 2, 3, 0, 0, 0 }, // 1,2 + 3 --> 2,3
	{ 0, 2, 3, 4, 0, 0 }, // 2,3 + 4 --> 3,4
	{ 0, 0, 3, 4, 5, 0 }, // 3,4 + 5 --> 4,5 V
	{ 0, 0, 0, 4, 5, 6 }, // 4,5 + 6 --> 5,6 V
	{ 1, 0, 0, 0, 5, 6 }, // 5,6 + 1 --> 6,1
	{ 1, 2, 0, 0, 0, 6 }, // 6,1 + 2

	4 Input feature map (9)
	----------------------------------------
	{ 1, 2, 3, 4, 0, 0 }, // 1,2,3 + 4
	{ 0, 2, 3, 4, 5, 0 }, // 2,3,4 + 5
	{ 0, 0, 3, 4, 5, 6 }, // 3,4,5 + 6
	{ 1, 0, 0, 4, 5, 6 }, // 4,5,6 + 1
	{ 1, 2, 0, 0, 5, 6 }, // 5,6,1 + 2
	{ 1, 2, 3, 0, 0, 6 }, // 6,1,2 + 3
	{ 1, 2, 0, 4, 5, 0 }, // 1,4 + 2,5
	{ 0, 2, 3, 0, 5, 6 }, // 2,5 + 3,6
	{ 1, 0, 3, 4, 0, 6 }, // 3,6 + 4,1

	6 Input feature map (1)
	----------------------------------------
	{ 1, 2, 3, 4, 5, 6 }  // 4,1 + 5,2

*/

	float input[image_Batch][CONV_1_TYPE][CONV_2_INPUT_WH][CONV_2_INPUT_WH];
	float kernel[CONV_2_TYPE][CONV_1_TYPE][CONV_2_WH][CONV_2_WH];
	float bias[CONV_2_TYPE];
	float output_buffer[image_Batch*CONV_2_TYPE*CONV_2_OUTPUT_SIZE];
#pragma HLS array_partition variable=input block factor=14 dim=4
#pragma HLS array_partition variable=kernel cyclic factor=5 dim=3
#pragma HLS array_partition variable=kernel cyclic factor=5 dim=4
#pragma HLS array_partition variable=bias complete dim=0

	int col, row;
	int col_f, row_f;
	int depth_in, depth_out;
	float temp = 0;
	int batch_idx;

	copy_input_1:
	for(int batch=0;batch<image_Batch;batch++){
		copy_input_2:
		for(int j=0;j<CONV_1_TYPE;j++){
			copy_input_3:
			for(int k=0;k<CONV_2_INPUT_WH;k++){
				copy_input_4:
				for(int l=0;l<CONV_2_INPUT_WH;l++){
#pragma HLS unroll factor=14
					input[batch][j][k][l] = input_feature[batch*CONV_1_TYPE*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + j*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + k*CONV_2_INPUT_WH
													  +l];
				}
			}
		}
	}


	copy_kernel_1 :
	for (int i = 0; i < CONV_2_TYPE; i++) {
		copy_kernel_2 :
		for(int j=0;j<CONV_1_TYPE;j++){
			copy_kernel_3 :
			for(int k=0;k<CONV_2_WH;k++){
				copy_kernel_4 :
				for(int l=0;l<CONV_2_WH;l++){
#pragma HLS unroll
					kernel[i][j][k][l] = conv_kernel[i*CONV_1_TYPE*CONV_2_WH*CONV_2_WH
													 + j*CONV_2_WH*CONV_2_WH
													 + k*CONV_2_WH
													 + l];
				}
			}
		}
	}
	copy_bias:
	for(int i=0;i<CONV_2_TYPE;i++){
#pragma HLS pipeline II=1
		bias[i] = conv_bias[i];
	}

	BATCH :
	for (batch_idx = 0; batch_idx < image_Batch; batch_idx++) {
		ROW :
		for (row = 0; row < CONV_2_OUTPUT_WH; row++) {
			COL :
			for (col = 0; col < CONV_2_OUTPUT_WH; col++) {
				float acc_din=0;
				float acc[CONV_2_TYPE];
				#pragma HLS array_partition variable=acc complete dim=0

				D_IN :
				for (depth_in = 0; depth_in < CONV_1_TYPE; depth_in++) {
					#pragma HLS pipeline II=4
					float mult[CONV_2_SIZE]; // multiplication
#pragma HLS array_partition variable=mult complete dim=0

					acc[depth_in]=0;
					// Multiplication
					for(int i=0;i<CONV_2_WH;i++){
					#pragma HLS unroll
						for(int j=0;j<CONV_2_WH;j++){
					#pragma HLS unroll
							mult[i*5+j] = input[batch_idx][depth_in][row+i][col+j]*kernel[depth_out][depth_in][i][j];
						}
					}
					Accumulate:
					for(int i=0;i<CONV_2_SIZE;i++){
					#pragma HLS unroll
						acc[depth_in] += mult[i];
					}
				}
				for(int i=0;i<CONV_2_TYPE;i++){
				#pragma HLS unroll
					acc_din += acc[i];
				}
				output_buffer[batch_idx*CONV_2_TYPE*CONV_2_OUTPUT_SIZE + depth_out*CONV_2_OUTPUT_SIZE + row*CONV_2_OUTPUT_WH + col]
							  = _tanh(acc_din+bias[depth_out]);
			}
		}
	}

	copy_output:
	for(int i=0;i<image_Batch*CONV_2_TYPE*CONV_2_OUTPUT_SIZE;i++){
		output_feature[i] = output_buffer[i];
	}
}



// Convolution Layer 3 (FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5],  Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]
void CONVOLUTION_LAYER_3(float input_feature[CONV_2_TYPE*image_Batch*CONV_3_INPUT_WH *CONV_3_INPUT_WH],
		 float conv_kernel1[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3],
		 float conv_kernel2[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3],
		 float conv_kernel3[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3],
		 float conv_bias[CONV_3_TYPE],
		 float output_feature[image_Batch * CONV_3_TYPE])
{

	float kernel[CONV_2_TYPE][CONV_3_WH][CONV_3_WH];
//#pragma HLS array_partition variable=kernel cyclic factor=5 dim=2
	float input[CONV_2_TYPE][CONV_3_INPUT_WH][CONV_3_INPUT_WH];
//#pragma HLS array_partition variable = input cyclic factor=5 dim=3



	int col, row, col_f, row_f;
	int depth_in, batch_cnt, depth_out;




	float temp;
	float __temp[5], _temp[CONV_2_TYPE];

	for (batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++) {
		for (depth_out = 0; depth_out < CONV_3_TYPE; depth_out++) {
			for(int i=0;i<CONV_2_TYPE;i++){
				for(int j=0;j<CONV_3_WH;j++){
////#pragma HLS pipeline II=1
					for(int k=0;k<CONV_3_WH;k++){
						if(depth_out<40)
							kernel[i][j][k] = conv_kernel1[depth_out*CONV_2_TYPE*CONV_3_WH*CONV_3_WH + i*CONV_3_WH*CONV_3_WH+j*CONV_3_WH + k];
						else if(depth_out>=40 && depth_out<80)
							kernel[i][j][k] = conv_kernel2[depth_out*CONV_2_TYPE*CONV_3_WH*CONV_3_WH + i*CONV_3_WH*CONV_3_WH+j*CONV_3_WH + k];
						else
							kernel[i][j][k] = conv_kernel3[depth_out*CONV_2_TYPE*CONV_3_WH*CONV_3_WH + i*CONV_3_WH*CONV_3_WH+j*CONV_3_WH + k];
					}
				}
			}
			for(int i=0;i<CONV_2_TYPE;i++){
				for(int j=0;j<CONV_3_INPUT_WH;j++){
//#pragma HLS pipeline II=1
					for(int k=0;k<CONV_3_INPUT_WH;k++){
						input[i][j][k] = input_feature[batch_cnt*CONV_2_TYPE*CONV_3_INPUT_SIZE + i*CONV_3_INPUT_SIZE + j*CONV_3_INPUT_WH + k];
					}
				}
			}


			// Init
			temp=0;
			for(int i=0;i<5;i++)
				__temp[i] = 0;
			for(int i=0;i<CONV_2_TYPE-1;i++)
				_temp[i] = 0;
			// Multiplication by Convolution and Input feature maps
			for (depth_in = 0; depth_in < POOL_2_TYPE; depth_in++) {
//#pragma HLS pipeline II=4
				for (row_f = 0; row_f < CONV_3_WH; row_f++) {

					__temp[row_f] = 	input[depth_in][row_f][0] * kernel[depth_in][row_f][0] +
									input[depth_in][row_f][1] * kernel[depth_in][row_f][1] +
									input[depth_in][row_f][2] * kernel[depth_in][row_f][2] +
									input[depth_in][row_f][3] * kernel[depth_in][row_f][3] +
									input[depth_in][row_f][4] * kernel[depth_in][row_f][4];
					if(row_f==CONV_3_WH-1){
						_temp[depth_in] = __temp[0]+__temp[1]+__temp[2]+__temp[3]+__temp[4];
						if(depth_in == CONV_2_TYPE-1){
							temp = _temp[0]+_temp[1]+_temp[2]+_temp[3]+_temp[4];
							// Result of Convolution
							output_feature[batch_cnt * CONV_3_TYPE + depth_out] = (temp+ conv_bias[depth_out]);
						}
					}
				}
			}
		}
	}
}

