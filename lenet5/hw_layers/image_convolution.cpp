#include <lenet5/hw_layers/image_convolution.h>




void CONVOLUTION_LAYER_1(float input_feature[image_Batch*INPUT_WH *INPUT_WH],
		float conv_kernel[CONV_1_TYPE*CONV_1_WH * CONV_1_WH],
		float conv_bias[CONV_1_TYPE],
		float output_feature[CONV_1_TYPE * image_Batch*CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH])
{

	int col, row, col_f, row_f;
	int depth_out, batch_cnt;
//#pragma HLS INTERFACE bram port=input_feature
//#pragma HLS INTERFACE bram port=output_feature
	float input[image_Batch][INPUT_WH][INPUT_WH];
#pragma HLS array_partition variable=input cyclic factor=2 dim=2
//#pragma HLS array_partition variable=input cyclic factor=2 dim=3
	float kernel[CONV_1_TYPE][CONV_1_WH*CONV_1_WH];
#pragma HLS array_partition variable=kernel complete dim=2

	float bias[CONV_1_TYPE];
#pragma HLS array_partition variable=bias complete dim=0

	float output_buffer[image_Batch*CONV_1_TYPE*CONV_1_OUTPUT_SIZE];



	copy_kernel_1:
	for(int i=0;i<CONV_1_TYPE;i++){
		copy_kernel_2:
		for(int j=0;j<CONV_1_SIZE;j++){
#pragma HLS PIPELINE II=1
			kernel[i][j] = conv_kernel[i*CONV_1_SIZE+j];
		}
	}

	copy_input_1:
	for(int batch_cnt=0;batch_cnt<image_Batch;batch_cnt++){
		copy_input_2 :
		for(int i=0;i<INPUT_WH;i++){
			copy_input_3 :
			for(int j=0;j<INPUT_WH;j++){
#pragma HLS pipeline
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
		ROW :
		for (row = 0; row < CONV_1_OUTPUT_WH; row++) {
			COL :
			for (col = 0; col < CONV_1_OUTPUT_WH; col++) {

				DEPTH_OUT :
				for (depth_out = 0; depth_out < CONV_1_TYPE; depth_out++) {
//#pragma HLS unroll factor=2
#pragma HLS pipeline II=7
					float mult[CONV_1_SIZE];
#pragma HLS array_partition variable=mult complete dim=0
					float acc=0;

					// Multiplication
					for(int i=0;i<CONV_1_WH;i++){
#pragma HLS unroll
						for(int j=0;j<CONV_1_WH;j++){
#pragma HLS unroll
							mult[i*5+j] = input[batch_cnt][row+i][col+j]*kernel[depth_out][i*5+j];
							//mult[i] = input[i]*kernel[depth_out][i];
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

float relu(float x){
#pragma HLS inline
	return x>0 ? x : 0;
}

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
//#pragma HLS INTERFACE ap_memory port=input_feature
//#pragma HLS INTERFACE ap_memory port=output_feature
	static const int C2_N_PE = 2;
//#pragma HLS INTERFACE bram port=input_feature
//#pragma HLS INTERFACE bram port=output_feature
//	float input[image_Batch][CONV_1_TYPE][CONV_2_INPUT_WH][CONV_2_INPUT_WH];
	float kernel[CONV_2_TYPE][CONV_1_TYPE][CONV_2_WH][CONV_2_WH];
	float bias[CONV_2_TYPE];
	//float output_buffer[image_Batch][CONV_2_TYPE][CONV_2_OUTPUT_SIZE];
//#pragma HLS array_partition variable=input cyclic factor=2 dim=2
//#pragma HLS array_partition variable=input cyclic factor=2 dim=3
//#pragma HLS array_partition variable=input cyclic factor=2 dim=4
//#pragma HLS array_partition variable=kernel cyclic factor=2 dim=1
//#pragma HLS array_partition variable=kernel cyclic factor=5 dim=3
//#pragma HLS array_partition variable=kernel cyclic factor=5 dim=4
//#pragma HLS array_partition variable=bias complete dim=0
//#pragma HLS array_partition variable=output_buffer cyclic factor=2 dim=2

/*
	copy_input_1:
	for(int batch=0;batch<image_Batch;batch++){
		copy_input_2:
		for(int j=0;j<CONV_1_TYPE;j++){
			copy_input_3:
			for(int k=0;k<CONV_2_INPUT_WH;k++){
				copy_input_4:
				for(int l=0;l<CONV_2_INPUT_WH;l++){
#pragma HLS unroll
					input[batch][j][k][l] = input_feature[batch*CONV_1_TYPE*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + j*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + k*CONV_2_INPUT_WH
													  +l];
				}
			}
		}
	}
*/

	copy_kernel_1 :
	for (int i = 0; i < CONV_2_TYPE; i++) {
		copy_kernel_2 :
		for(int j=0;j<CONV_1_TYPE;j++){
			copy_kernel_3 :
			for(int k=0;k<CONV_2_WH;k++){
				copy_kernel_4 :
				for(int l=0;l<CONV_2_WH;l++){
#pragma HLS unroll
					kernel[i][j][k][l] = conv_kernel[i*CONV_1_TYPE*CONV_2_SIZE
													 + j*CONV_2_SIZE
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
	for (int batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++) {
		ROW :
		for (int row = 0; row < CONV_2_OUTPUT_WH; row++) {
			COL	 :
			for (int col = 0; col < CONV_2_OUTPUT_WH; col++) {
				float output_buf[CONV_2_TYPE];
				#pragma HLS array_partition variable=output_buf cyclic factor=C2_N_PE
				float input[CONV_1_TYPE][CONV_2_SIZE];
				#pragma HLS array_partition variable=input cyclic dim=1 factor=C2_N_PE
				fetch_input:
				for(int k=0;k<CONV_1_TYPE;k++){
					for(int i=0;i<5;i++){
					#pragma HLS unroll
						for(int j=0;j<5;j++){
						#pragma HLS unroll
							input[k][i*5+j] = input_feature[batch_cnt*1176+k*196+(row+i)*14+col+j];
						}
					}
				}
				DEPTH_IN:
				for(int depth_in = 0; depth_in < CONV_1_TYPE; depth_in++){

					DEPTH_OUT :
					for (int depth_out = 0; depth_out < CONV_2_TYPE; depth_out++) {
					#pragma HLS unroll factor=C2_N_PE
					#pragma HLS pipeline II=8
						float mult[CONV_2_SIZE]; // multiplication
						#pragma HLS array_partition variable=mult complete dim=0

						float acc=0.0f;
						// Multiplication
						Mult:
						for(int i=0;i<CONV_2_WH;i++){
#pragma HLS unroll
							for(int j=0;j<CONV_2_WH;j++){
#pragma HLS unroll
								//mult[i*5+j] = input[batch_cnt][depth_in][row+i][col+j]*kernel[depth_out][depth_in][i][j];
								mult[i*5+j] = input[depth_in][i*5+j]*kernel[depth_out][depth_in][i][j];
							}
						}
						Accumulate:
						for(int i=0;i<CONV_2_SIZE;i++){
						#pragma HLS unroll
							acc += mult[i];
						}
						output_buf[depth_out] += acc;
					}

				}
				for(int i=0;i<CONV_2_TYPE;i++){
#pragma HLS pipeline
					output_feature[batch_cnt*CONV_2_TYPE*CONV_2_OUTPUT_SIZE + i*CONV_2_OUTPUT_SIZE + row*CONV_2_OUTPUT_WH + col]=
							_tanh(output_buf[i]+bias[i]);
					//output_buffer[batch_cnt][i][row*CONV_2_OUTPUT_WH + col] = output_buf[i] + bias[i];
				}
			}
		}
	}
/*
	copy_output:
	for(int i=0;i<image_Batch;i++){
		for(int j=0;j<CONV_2_TYPE;j++){
			for(int k=0;k<CONV_2_OUTPUT_SIZE;k++){
#pragma HLS pipeline
				output_feature[i*CONV_2_TYPE*CONV_2_OUTPUT_SIZE + j*CONV_2_OUTPUT_SIZE + k] = _tanh(output_buffer[i][j][k]);
			}
		}

	}*/
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
	static const int C3_N_PE = 1;

//	float input[image_Batch][CONV_2_TYPE][CONV_3_INPUT_WH][CONV_3_INPUT_WH];
//#pragma HLS array_partition variable=input cyclic factor=5 dim=3
//#pragma HLS array_partition variable=input cyclic factor=5 dim=4
//	float kernel[CONV_3_TYPE][CONV_2_TYPE][CONV_3_WH][CONV_3_WH];
//#pragma HLS array_partition variable=kernel cyclic factor=5 dim=3
//#pragma HLS array_partition variable=kernel cyclic factor=5 dim=4

	float bias[CONV_3_TYPE];
//	#pragma HLS array_partition variable=bias complete dim=0
	
	//float output_buffer[image_Batch][CONV_3_TYPE];
//#pragma HLS array_partition variable=output_buffer cyclic factor=2 dim=2

/*
	copy_input_1:
	for(int batch=0; batch<image_Batch; batch++){
		copy_input_2:
		for(int i=0; i<CONV_2_TYPE; i++){
			copy_input_3:
			for(int j=0;j<CONV_3_INPUT_WH;j++){
				for(int k=0;k<CONV_3_INPUT_WH;k++){
#pragma HLS unroll
					input[batch][i][j][k] = input_feature[batch*CONV_2_TYPE*CONV_3_INPUT_SIZE + i*CONV_3_INPUT_SIZE + j*CONV_3_INPUT_WH+k];
				}

			}
		}
	}

	copy_kernel_1:
	for(int i=0;i<CONV_3_TYPE;i++){
		copy_kernel_2:
		for(int j=0;j<CONV_2_TYPE; j++){
			copy_kernel_3:
			for(int k=0;k<CONV_3_WH; k++){
#pragma HLS unroll
				for(int l=0;l<CONV_3_WH;l++){
#pragma HLS unroll
					kernel[i][j][k][l] = conv_kernel[i*CONV_2_TYPE*CONV_3_SIZE + j*CONV_3_SIZE + k*5+l];
				}

			}
		}
	}
	*/
	copy_bias:
	for(int i=0;i<CONV_3_TYPE; i++){
#pragma HLS pipeline II=1
		bias[i] = conv_bias[i];
	}


	BATCH:
	for (int batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++) {
		float output_buf[CONV_3_TYPE];
#pragma HLS array_partition variable=output_buf cyclic factor=C3_N_PE
		float input[CONV_3_TYPE][25];
#pragma HLS array_partition variable=input cyclic factor=25 dim=2
		for(int k=0;k<CONV_2_TYPE;k++){
			for(int i=0;i<5;i++){
				for(int j=0;j<5;j++){
					input[k][i*5+j] = input_feature[batch_cnt*CONV_2_TYPE*CONV_3_INPUT_SIZE + k*CONV_3_INPUT_SIZE + i*5+j];
				}
			}
		}
		DEPTH_IN:
		for(int depth_in = 0; depth_in < CONV_2_TYPE; depth_in++){
			float kernel[CONV_3_TYPE][CONV_3_SIZE];
#pragma HLS array_partition variable=kernel cyclic factor=C3_N_PE dim=1
			for(int i=0;i<40;i++){
				for(int j=0;j<CONV_3_SIZE;j++){
				#pragma HLS unroll
					kernel[i][j] = conv_kernel1[depth_in*40*25+i*25+j];
				}
			}
			for(int i=0;i<40;i++){
				for(int j=0;j<CONV_3_SIZE;j++){
					#pragma HLS unroll
					kernel[i+40][j] = conv_kernel2[depth_in*40*25+i*25+j];
				}
			}
			for(int i=0;i<40;i++){
				for(int j=0;j<CONV_3_SIZE;j++){
					#pragma HLS unroll
					kernel[i+80][j] = conv_kernel3[depth_in*40*25+i*25+j];
				}
			}
			DEPTH_OUT :
			for (int depth_out = 0; depth_out < CONV_3_TYPE; depth_out++) {
			#pragma HLS unroll factor=C3_N_PE
			#pragma HLS pipeline II=5
				float mult[CONV_3_SIZE]; // multiplication
				#pragma HLS array_partition variable=mult complete dim=0

				float acc=0.0f;
				// Multiplication
				for(int i=0;i<CONV_3_WH;i++){
				#pragma HLS unroll
					for(int j=0;j<CONV_3_WH;j++){
					#pragma HLS unroll
						//mult[i*5+j] = input[batch_cnt][depth_in][i][j]*kernel[depth_out][depth_in][i][j];
						mult[i*5+j] = input[depth_in][i*5+j]*kernel[depth_out][i*5+j];//kernel[depth_out][depth_in][i][j];
					}
				}
				Accumulate:
				for(int i=0;i<CONV_3_SIZE;i++){
				#pragma HLS unroll
					acc += mult[i];
				}

				output_buf[depth_out] += acc;
			}

		}
		for(int i=0;i<120;i++){
		#pragma HLS pipeline
			output_feature[batch_cnt*CONV_3_TYPE + i] = _tanh(output_buf[i]+bias[i]);
			//output_buffer[batch_cnt][i] = output_buf[i] + bias[i];
		}

	}
	/*
	for(int i=0;i<image_Batch;i++){
		for(int j=0;j<CONV_3_TYPE;j++)
		output_feature[i*120+j] = output_buffer[i][j];
	}*/
}
		

		
