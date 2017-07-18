#include <lenet5/hw_layers/image_convolution.h>
//#include "hls_math.h"

float _tanh(float x){
#pragma HLS INLINE
	float exp2x = expf(2*x)+1;
	return (exp2x-2)/(exp2x);
}

float relu(float x){
#pragma HLS inline
	return x>0 ? x : 0;
}




void CONVOLUTION_LAYER_1(float input[image_Batch*32*32],
		float weights[6*5*5],
		float bias[6],
		float output[image_Batch*6*28*28]//, int init
		)
{

	float IBRAM[image_Batch][INPUT_WH][INPUT_WH];
	float WBRAM[CONV_1_TYPE][5][5];
	float biasBRAM[CONV_1_TYPE];
	float OBRAM[image_Batch][CONV_1_TYPE][CONV_1_OUTPUT_SIZE];
#pragma HLS array_partition variable=WBRAM complete dim=1
#pragma HLS array_partition variable=biasBRAM complete dim=0
#pragma HLS array_partition variable=OBRAM complete dim=2

	copy_input_1:
	for(int batch_cnt=0;batch_cnt<image_Batch;batch_cnt++){
		copy_input_2 :
		for(int i=0;i<INPUT_WH;i++){
			copy_input_3 :
			for(int j=0;j<INPUT_WH;j++){
			#pragma HLS PIPELINE II=1
				IBRAM[batch_cnt][i][j] = input[batch_cnt*INPUT_WH*INPUT_WH+i*INPUT_WH + j];
			}
		}
	}

	// load weights & bias at first iteration only
	//if(init){
		copy_kernel_1:
		for(int i=0;i<CONV_1_TYPE;i++){
			copy_kernel_2:
			for(int j=0;j<5;j++){
				for(int k=0;k<5;k++){
				#pragma HLS PIPELINE II=1
					WBRAM[i][j][k] = weights[i*CONV_1_SIZE+j*5+k];
				}
			}
		}


		copy_bias:
		for(int i=0;i<CONV_1_TYPE;i++){
	#pragma HLS PIPELINE II=1
			biasBRAM[i] = bias[i];
		}
	//}

	//////////////////////////////////////////////////////////////////////
	//						   Convolution								//
	//////////////////////////////////////////////////////////////////////
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
#pragma HLS DEPENDENCE variable=IBRAM inter false
#pragma HLS DEPENDENCE variable=WBRAM inter false
#pragma HLS DEPENDENCE variable=OBRAM inter false
						#pragma HLS unroll
							float mult = IBRAM[batch_cnt][row+row_k][col+col_k]*WBRAM[co][row_k][col_k];
							if(col_k==0&&row_k==0)
								OBRAM[batch_cnt][co][row*28+col]=mult;
							else
								OBRAM[batch_cnt][co][row*28+col]+=mult;
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
				output[i*6*784+j*784+k]=_tanh(OBRAM[i][j][k]+biasBRAM[j]);
			}
		}

	}
}

void CONVOLUTION_LAYER_2(float input[image_Batch*6*14*14],
		float weights[6*16*5*5],
		float bias[CONV_2_TYPE],
		float output[image_Batch*16*10*10]//, int init
		)
{


	static const int C2_N_PE = 2;


	float IBRAM[image_Batch][CONV_1_TYPE][CONV_2_INPUT_WH][CONV_2_INPUT_WH];
	float WBRAM[CONV_2_TYPE][CONV_1_TYPE][CONV_2_WH][CONV_2_WH];
	float biasBRAM[CONV_2_TYPE];
	float OBRAM[image_Batch][CONV_2_TYPE][CONV_2_OUTPUT_SIZE];
#pragma HLS array_partition variable=IBRAM complete dim=2
#pragma HLS array_partition variable=WBRAM complete dim=2
#pragma HLS array_partition variable=WBRAM cyclic factor=C2_N_PE dim=1
#pragma HLS array_partition variable=biasBRAM complete dim=0
#pragma HLS array_partition variable=OBRAM cyclic factor=C2_N_PE dim=2

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
					IBRAM[batch][j][k][l] = input[batch*CONV_1_TYPE*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + j*CONV_2_INPUT_WH*CONV_2_INPUT_WH
													  + k*CONV_2_INPUT_WH
													  +l];
				}
			}
		}
	}

//	if(init){
	copy_kernel_1 :
	for (int i = 0; i < CONV_2_TYPE; i++) {
		copy_kernel_2 :
		for(int j=0;j<CONV_1_TYPE;j++){
			copy_kernel_3 :
			for(int k=0;k<CONV_2_WH;k++){
				copy_kernel_4 :
				for(int l=0;l<CONV_2_WH;l++){
#pragma HLS pipeline II=1
					WBRAM[i][j][k][l] = weights[i*CONV_1_TYPE*CONV_2_SIZE
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
		biasBRAM[i] = bias[i];
	}
	//}
	//////////////////////////////////////////////////////////////////////
	//						   Convolution								//
	//////////////////////////////////////////////////////////////////////
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
//#pragma HLS DEPENDENCE variable=IBRAM inter false
//#pragma HLS DEPENDENCE variable=WBRAM inter false
#pragma HLS DEPENDENCE variable=OBRAM inter false
						#pragma HLS unroll factor=C2_N_PE
						#pragma HLS pipeline II=1
							float mult[CONV_1_TYPE]; // multiplication
							#pragma HLS array_partition variable=mult complete dim=0
							float acc=0;
							DEPTH_IN:
							for (int depth_in = 0; depth_in < CONV_1_TYPE; depth_in++) {
							#pragma HLS unroll
								mult[depth_in] = IBRAM[batch_cnt][depth_in][row+row_k][col+col_k] *
										WBRAM[depth_out][depth_in][row_k][col_k];
							}
							acc = (mult[0]+mult[1])+(mult[2]+mult[3])+(mult[4]+mult[5]);
							if(col_k==0&&row_k==0)
								OBRAM[batch_cnt][depth_out][row*10 + col] = acc;
							else
								OBRAM[batch_cnt][depth_out][row*10 + col] += acc;
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
				output[i*1600+depth_offset+k]=_tanh(OBRAM[i][j][k]+biasBRAM[j]);
			}
		}
	}
}


// Convolution Layer 3 (FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5],  Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]
void CONVOLUTION_LAYER_3(float input[image_Batch*16*5*5],
		float weights[16*120*5*5],
		float bias[120],
		float output[image_Batch*120]//, int init
		)
{
	
	
	float IBRAM[image_Batch][CONV_2_TYPE][CONV_3_INPUT_WH][CONV_3_INPUT_WH];
	float WBRAM[CONV_3_TYPE][CONV_2_TYPE][CONV_3_WH][CONV_3_WH];
#pragma HLS array_partition variable=IBRAM complete dim=2
#pragma HLS array_partition variable=WBRAM complete dim=2


	float biasBRAM[CONV_3_TYPE];
	float OBRAM[image_Batch][CONV_3_TYPE];


//#pragma HLS DATAFLOW
	copy_input_1:
	for(int batch=0; batch<image_Batch; batch++){
		copy_input_2:
		for(int i=0; i<CONV_2_TYPE; i++){
			copy_input_3:
			for(int j=0;j<CONV_3_INPUT_WH;j++){
				for(int k=0;k<CONV_3_INPUT_WH;k++){
#pragma HLS pipeline
					IBRAM[batch][i][j][k] = input[batch*CONV_2_TYPE*CONV_3_INPUT_SIZE + i*CONV_3_INPUT_SIZE + j*CONV_3_INPUT_WH+k];
				}

			}
		}
	}
	//if(init){
		copy_kernel_1:
		for(int i=0;i<CONV_3_TYPE;i++){
			for(int j=0;j<CONV_2_TYPE; j++){
				for(int k=0;k<CONV_3_WH; k++){
					for(int l=0;l<CONV_3_WH;l++){
					#pragma HLS pipeline
						WBRAM[i][j][k][l] = weights[i*CONV_2_TYPE*CONV_3_SIZE + j*25 + k*5+l];
					}
				}
			}
		}

		copy_bias:
		for(int i=0;i<120; i++){
		#pragma HLS pipeline II=1
			biasBRAM[i] = bias[i];
		}
	//}
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
						mult[ci] = IBRAM[batch_cnt][ci][row_k][col_k]*WBRAM[co][ci][row_k][col_k];
					}
					for(int i=0,ii=0;i<4;i++,ii+=4){
					#pragma HLS unroll
						acc[i] = (mult[ii]+mult[ii+1])+(mult[ii+2]+mult[ii+3]);
					}
					result = (acc[0]+acc[1])+(acc[2]+acc[3]);
					if(col_k==0&&row_k==0)
						OBRAM[batch_cnt][co]=result;
					else
						OBRAM[batch_cnt][co]+=result;
				}
			}
		}
	}

	for(int i=0;i<image_Batch;i++){
		for(int j=0;j<120;j++)
#pragma HLS pipeline II=1
		output[i*120+j] = _tanh(OBRAM[i][j]+biasBRAM[j]);
	}
}


/*
void copy_input(float* DRAM, float buffer[1][32][32]){
	copy_input_1:
	for(int batch_cnt=0;batch_cnt<image_Batch;batch_cnt++){
		copy_input_2 :
		for(int i=0;i<INPUT_WH;i++){
			copy_input_3 :
			for(int j=0;j<INPUT_WH;j++){
#pragma HLS PIPELINE II=1
				buffer[batch_cnt][i][j] = DRAM[batch_cnt*INPUT_WH*INPUT_WH+i*INPUT_WH + j];
			}
		}
	}
}
void copy_weights(float Wconv1[6*25], float bconv1[6],
			float Wconv2[6*16*25], float bconv2[16],
			float Wconv3[16*120*25], float bconv3[120],
			float W1BRAM[6][5][5], float b1BRAM[6],
			float W2BRAM[6][16][5][5], float b2BRAM[16],
			float W3BRAM[16][120][5][5], float b3BRAM[120]
			)
{
	copy_conv1_weight:
	for(int i=0;i<CONV_1_TYPE;i++){
		for(int j=0;j<5;j++){
			for(int k=0;k<5;k++){
			#pragma HLS PIPELINE II=1
				W1BRAM[i][j][k] = Wconv1[i*CONV_1_SIZE+j*5+k];
			}
		}
	}
	copy_conv2_weight:
	for (int i = 0; i < CONV_1_TYPE; i++) {
		for(int j=0;j<CONV_2_TYPE;j++){
			for(int k=0;k<CONV_2_WH;k++){
				for(int l=0;l<CONV_2_WH;l++){
				#pragma HLS pipeline II=1
					W2BRAM[i][j][k][l] = Wconv2[i*CONV_2_TYPE*CONV_2_SIZE
													 + j*CONV_2_SIZE
													 + k*CONV_2_WH
													 + l];
				}
			}
		}
	}
	
	copy_conv3_weight:
	for(int i=0;i<CONV_2_TYPE;i++){
		for(int j=0;j<120; j++){
			for(int k=0;k<CONV_3_WH; k++){
				for(int l=0;l<CONV_3_WH;l++){
#pragma HLS pipeline
					W3BRAM[i][j][k][l] = Wconv3[i*120*CONV_3_SIZE + j*25 + k*5+l];
				}
			}
		}
	}
	copy_conv1_bias:
	for(int i=0;i<6;i++){
	#pragma HLS pipeline
		b1BRAM[i] = bconv1[i];
	}
	for(int i=0;i<16;i++){
	#pragma HLS pipeline
		b2BRAM[i] = bconv2[i];
	}
	for(int i=0;i<120;i++){
	#pragma HLS pipeline
		b3BRAM[i] = bconv3[i];
	}
	
}

void store_output(float buffer[1][120], float* b3BRAM, float* DRAM){
	for(int i=0;i<image_Batch;i++){
		for(int j=0;j<120;j++){
		#pragma HLS pipeline
			DRAM[i*120 + j] = _tanh(buffer[i][j]+b3BRAM[j]);
		}
	}
}
*/
