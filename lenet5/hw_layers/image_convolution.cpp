#include <lenet5/hw_layers/image_convolution.h>
//#include "hls_math.h"

float _tanh(float x){
#pragma HLS INLINE
//#pragma HLS pipeline
	float exp2x = expf(2*x)+1;
	return (exp2x-2)/(exp2x);
}

float relu(float x){
#pragma HLS inline
	return x>0 ? x : 0;
}


void conv_top(float input[32*32],float Wconv1[6*25], float bconv1[6],
			float Wconv2[6*16*25], float bconv2[16],
			float Wconv3[16*120*25], float bconv3[120],
			float output[120]){
//#pragma HLS DATAFLOW
	float pool1[6*14*14];
	float pool2[16*5*5];

	CONVOLUTION_LAYER_1(input,Wconv1,bconv1,pool1);
	CONVOLUTION_LAYER_2(pool1,Wconv2,bconv2,pool2);
	CONVOLUTION_LAYER_3(pool2,Wconv3,bconv3,output);

}

void CONVOLUTION_LAYER_1(float input_feature[image_Batch*INPUT_WH *INPUT_WH],
		float conv_kernel[CONV_1_TYPE*25],
		float conv_bias[CONV_1_TYPE],
		float output_feature[image_Batch*CONV_1_TYPE*CONV_1_OUTPUT_SIZE/4]
		)
{
	float input[image_Batch][INPUT_WH][INPUT_WH];
	float kernel[CONV_1_TYPE][5][5];
	float bias[CONV_1_TYPE];
	float output[image_Batch][CONV_1_TYPE][CONV_1_OUTPUT_SIZE];
#pragma HLS array_partition variable=kernel complete dim=1
#pragma HLS array_partition variable=bias complete dim=0
#pragma HLS array_partition variable=output complete dim=2

	copy_kernel_1:
	for(int i=0;i<CONV_1_TYPE;i++){
		copy_kernel_2:
		for(int j=0;j<5;j++){
			for(int k=0;k<5;k++){
			#pragma HLS PIPELINE II=1
				kernel[i][j][k] = conv_kernel[i*CONV_1_SIZE+j*5+k];
			}
		}
	}

	copy_input_1:
	for(int batch_cnt=0;batch_cnt<image_Batch;batch_cnt++){
		copy_input_2 :
		for(int i=0;i<INPUT_WH;i++){
			copy_input_3 :
			for(int j=0;j<INPUT_WH;j++){
#pragma HLS PIPELINE II=1
				input[batch_cnt][i][j] = input_feature[batch_cnt*INPUT_WH*INPUT_WH+i*INPUT_WH + j];
			}
		}
	}


	copy_bias:
	for(int i=0;i<CONV_1_TYPE;i++){
#pragma HLS PIPELINE II=1
		bias[i] = conv_bias[i];
	}


	BATCH :
	for(int batch_cnt=0; batch_cnt<image_Batch; batch_cnt++) {
		ROW :
		for (int row = 0; row < CONV_1_OUTPUT_WH; row++) {
			COL :
			for (int col = 0; col < CONV_1_OUTPUT_WH; col++) {
				ROW_K:
				for(int row_k=0;row_k<5;row_k++){
					COL_K:
					for(int col_k=0;col_k<5;col_k++){
					#pragma HLS pipeline
						D_OUT:
						for(int co=0;co<6;co++){
#pragma HLS DEPENDENCE variable=input inter false
#pragma HLS DEPENDENCE variable=kernel inter false
#pragma HLS DEPENDENCE variable=output inter false
						#pragma HLS unroll
							float mult = input[batch_cnt][row+row_k][col+col_k]*kernel[co][row_k][col_k];
							if(row_k==0)
								output[batch_cnt][co][row*28+col]=mult;
							else
								output[batch_cnt][co][row*28+col]+=mult;
						}
					}
				}
			}
		}
	}

	copy_output:
	for(int i=0;i<image_Batch;i++){
		for(int j=0;j<6;j++){
			for(int k=0;k<784;k++){
#pragma HLS PIPELINE II=1
				//output_feature[i*6*28*28+j*784+k] = _tanh(output[i][j][k]+bias[j]);
				output[i][j][k]=output[i][j][k]+bias[j];
			}
		}

	}
	
	for(int batch=0;batch<image_Batch;batch++){
			for(int depth=0;depth<POOL_1_TYPE;depth++){
				for(int row=0;row<POOL_1_OUTPUT_WH;row++){
					for(int col=0;col<POOL_1_OUTPUT_WH;col++){
						#pragma HLS pipeline
						float max1, max2, max;
						float a00, a01, a10, a11;
						int rr = row<<1;
						int cc = col<<1;
						a00 = output[batch][depth][rr*28+cc];
						a01 = output[batch][depth][rr*28+cc+1];
						a10 = output[batch][depth][(rr+1)*28+cc];
						a11 = output[batch][depth][(rr+1)*28+cc+1];
						max1 = a00 > a01 ? a00 : a01;
						max2 = a10 > a11 ? a10 : a11;
						max  = max1 > max2 ? max1 : max2;

						/*
						for(int row_w=0;row_w<2;row_w++){
							for(int col_w=0;col_w<2;col_w++){
								if(src[batch*POOL_1_TYPE*POOL_1_INPUT_SIZE + depth*POOL_1_INPUT_SIZE +
									(2*row+row_w)*POOL_1_INPUT_WH + col*2+col_w]>max){
									max = src[batch*POOL_1_TYPE*POOL_1_INPUT_SIZE + depth*POOL_1_INPUT_SIZE +
												(2*row+row_w)*POOL_1_INPUT_WH + col*2+col_w];
								}
							}
						}*/
						output_feature[batch*POOL_1_TYPE*POOL_1_OUTPUT_SIZE + depth*POOL_1_OUTPUT_SIZE + row*POOL_1_OUTPUT_WH + col] = _tanh(max);
					}
				}
			}
		}

	
}

void CONVOLUTION_LAYER_2(float input_feature[CONV_1_TYPE * image_Batch*CONV_2_INPUT_WH *CONV_2_INPUT_WH],
		float conv_kernel[CONV_2_TYPE*CONV_1_TYPE*CONV_2_WH * CONV_2_WH],
		float conv_bias[CONV_2_TYPE],
		float output_feature[CONV_2_TYPE * image_Batch*CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH/4]
		)

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
	static const int C2_N_PE = 2;
	float input[image_Batch][CONV_1_TYPE][CONV_2_INPUT_WH][CONV_2_INPUT_WH];
	float kernel[CONV_1_TYPE][CONV_2_TYPE][CONV_2_WH][CONV_2_WH];
	float bias[CONV_2_TYPE];
	float output[image_Batch][CONV_2_TYPE][CONV_2_OUTPUT_SIZE];
#pragma HLS array_partition variable=input complete dim=2
#pragma HLS array_partition variable=kernel complete dim=1
#pragma HLS array_partition variable=kernel cyclic factor=C2_N_PE dim=2
#pragma HLS array_partition variable=bias complete dim=0
#pragma HLS array_partition variable=output cyclic factor=C2_N_PE dim=2

//#pragma HLS DATAFLOW


	copy_input_1:
	for(int batch=0;batch<image_Batch;batch++){
		copy_input_2:
		for(int j=0;j<CONV_1_TYPE;j++){
			copy_input_3:
			for(int k=0;k<CONV_2_INPUT_WH;k++){
				copy_input_4:
				for(int l=0;l<CONV_2_INPUT_WH;l++){
#pragma HLS pipeline II=1
					input[batch][j][k][l] = input_feature[batch*CONV_1_TYPE*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + j*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + k*CONV_2_INPUT_WH
													  +l];
				}
			}
		}
	}


	copy_kernel_1 :
	for (int i = 0; i < CONV_1_TYPE; i++) {
		copy_kernel_2 :
		for(int j=0;j<CONV_2_TYPE;j++){
			copy_kernel_3 :
			for(int k=0;k<CONV_2_WH;k++){
				copy_kernel_4 :
				for(int l=0;l<CONV_2_WH;l++){
#pragma HLS pipeline II=1
					kernel[i][j][k][l] = conv_kernel[i*CONV_2_TYPE*CONV_2_SIZE
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
				ROW_K:
				for(int row_k = 0;row_k<5;row_k++){
					COL_K:
					for(int col_k=0;col_k<5;col_k++){
						DEPTH_OUT:
						for(int depth_out = 0; depth_out < CONV_2_TYPE; depth_out++){
//#pragma HLS DEPENDENCE variable=input inter false
//#pragma HLS DEPENDENCE variable=kernel inter false
#pragma HLS DEPENDENCE variable=output inter false
						#pragma HLS unroll factor=C2_N_PE
						#pragma HLS pipeline II=1
							float mult[CONV_1_TYPE]; // multiplication
							#pragma HLS array_partition variable=mult complete dim=0
							float acc=0;
							DEPTH_IN:
							for (int depth_in = 0; depth_in < CONV_1_TYPE; depth_in++) {
							#pragma HLS unroll
								mult[depth_in] = input[batch_cnt][depth_in][row+row_k][col+col_k] *
										kernel[depth_in][depth_out][row_k][col_k];
							}
							acc = (mult[0]+mult[1])+(mult[2]+mult[3])+(mult[4]+mult[5]);
							if(row_k==0)
								output[batch_cnt][depth_out][row*10 + col] = acc;
							else
								output[batch_cnt][depth_out][row*10 + col] += acc;
						}
					}
				}
			}
		}
	}

	copy_output:
	for(int i=0;i<image_Batch;i++){
		for(int j=0;j<CONV_2_TYPE;j++){
			int depth_offset = j*100;
			for(int k=0;k<CONV_2_OUTPUT_SIZE;k++){
#pragma HLS pipeline II=1
				//output_feature[i*1600 + depth_offset + k] = _tanh(output[i][j][k]+bias[j]);
				output[i][j][k]=output[i][j][k]+bias[j];
			}
		}

	}
	for(int batch=0;batch<image_Batch;batch++){
			for(int depth=0;depth<POOL_2_TYPE;depth++){
				for(int row=0;row<POOL_2_OUTPUT_WH;row++){
					for(int col=0;col<POOL_2_OUTPUT_WH;col++){
						#pragma HLS pipeline

						float max1, max2, max;
						float a00, a01, a10, a11;
						int rr = row<<1;
						int cc = col<<1;
						a00 = output[batch][depth][rr*10+cc];
						a01 = output[batch][depth][rr*10+cc+1];
						a10 = output[batch][depth][(rr+1)*10+cc];
						a11 = output[batch][depth][(rr+1)*10+cc+1];
						max1 = a00 > a01 ? a00 : a01;
						max2 = a10 > a11 ? a10 : a11;
						max  = max1 > max2 ? max1 : max2;

						/*
						for(int row_w=0;row_w<2;row_w++){
							for(int col_w=0;col_w<2;col_w++){
								if(src[batch*POOL_1_TYPE*POOL_1_INPUT_SIZE + depth*POOL_1_INPUT_SIZE +
									(2*row+row_w)*POOL_1_INPUT_WH + col*2+col_w]>max){
									max = src[batch*POOL_1_TYPE*POOL_1_INPUT_SIZE + depth*POOL_1_INPUT_SIZE +
												(2*row+row_w)*POOL_1_INPUT_WH + col*2+col_w];
								}
							}
						}*/
						output_feature[batch*POOL_2_TYPE*POOL_2_OUTPUT_SIZE + depth*POOL_2_OUTPUT_SIZE + row*POOL_2_OUTPUT_WH + col] = _tanh(max);
						/*float max=-FLT_MAX;
						for(int row_w=0;row_w<2;row_w++){
							for(int col_w=0;col_w<2;col_w++){
								if(src[batch*POOL_2_TYPE*POOL_2_INPUT_SIZE + depth*POOL_2_INPUT_SIZE +
									(2*row+row_w)*POOL_2_INPUT_WH + col*2+col_w]>max){
									max = src[batch*POOL_2_TYPE*POOL_2_INPUT_SIZE + depth*POOL_2_INPUT_SIZE +
												(2*row+row_w)*POOL_2_INPUT_WH + col*2+col_w];
								}
							}
						}
						dst[batch*POOL_2_TYPE*POOL_2_OUTPUT_SIZE + depth*POOL_2_OUTPUT_SIZE + row*POOL_2_OUTPUT_WH + col] = max;*/
					}
				}
			}
		}

}


// TODO : CONV3 READ&WRITE FUNCTION, PE Inline OFF



// Convolution Layer 3 (FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5],  Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]
void CONVOLUTION_LAYER_3(float input_feature[CONV_2_TYPE*image_Batch*CONV_3_INPUT_WH *CONV_3_INPUT_WH],
						float conv_kernel[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH],
						float conv_bias[CONV_3_TYPE],
						 float output_feature[image_Batch*CONV_3_TYPE]
						 )
{
	float input[image_Batch][CONV_2_TYPE][CONV_3_INPUT_WH][CONV_3_INPUT_WH];
#pragma HLS array_partition variable=input complete dim=2
//#pragma HLS array_partition variable=input complete dim=4
	float kernel[CONV_2_TYPE][120][CONV_3_WH][CONV_3_WH];
#pragma HLS array_partition variable=kernel complete dim=1
//#pragma HLS array_partition variable=kernel complete dim=4

	float bias[CONV_3_TYPE];
	float output[image_Batch][CONV_3_TYPE];


//#pragma HLS DATAFLOW
	copy_input_1:
	for(int batch=0; batch<image_Batch; batch++){
		copy_input_2:
		for(int i=0; i<CONV_2_TYPE; i++){
			copy_input_3:
			for(int j=0;j<CONV_3_INPUT_WH;j++){
				for(int k=0;k<CONV_3_INPUT_WH;k++){
#pragma HLS pipeline
					input[batch][i][j][k] = input_feature[batch*CONV_2_TYPE*CONV_3_INPUT_SIZE + i*CONV_3_INPUT_SIZE + j*CONV_3_INPUT_WH+k];
				}

			}
		}
	}

	copy_kernel_1:
	for(int i=0;i<CONV_2_TYPE;i++){
		for(int j=0;j<120; j++){
			for(int k=0;k<CONV_3_WH; k++){
				for(int l=0;l<CONV_3_WH;l++){
#pragma HLS pipeline
					kernel[i][j][k][l] = conv_kernel[i*120*CONV_3_SIZE + j*25 + k*5+l];
				}
			}
		}
	}

	copy_bias:
	for(int i=0;i<120; i++){
#pragma HLS pipeline II=1
		bias[i] = conv_bias[i];
	}

	BATCH:
	for (int batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++) {
		ROW_K:
		for(int row_k=0;row_k<5;row_k++){
			for(int col_k=0;col_k<5;col_k++){
				D_OUT:
				for(int co=0;co<120;co++){
				#pragma HLS pipeline II=1
					float mult[16];
					#pragma HLS array_partition variable=mult complete dim=0
					float acc[4];
					#pragma HLS array_partition variable=acc complete dim=0
					float result=0;
						D_IN:
					for(int ci=0;ci<16;ci++){
					#pragma HLS unroll
						mult[ci] = input[batch_cnt][ci][row_k][col_k]*kernel[ci][co][row_k][col_k];
					}
					for(int i=0,ii=0;i<4;i++,ii+=4){
					#pragma HLS unroll
						acc[i] = (mult[ii]+mult[ii+1])+(mult[ii+2]+mult[ii+3]);
					}
					result = (acc[0]+acc[1])+(acc[2]+acc[3]);
					if(row_k==0)
						output[batch_cnt][co]=result;
					else
						output[batch_cnt][co]+=result;
				}
			}
		}
	}

	for(int i=0;i<image_Batch;i++){
		for(int j=0;j<120;j++)
#pragma HLS pipeline II=1
		output_feature[i*120+j] = _tanh(output[i][j]+bias[j]);
	}
}
