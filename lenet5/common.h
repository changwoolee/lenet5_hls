#ifndef SRC_LENET5_COMMON_H_
#define SRC_LENET5_COMMON_H_
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<float.h>

// Environment Option
// ===============================================

//////////////////////// Options ///////////////////////
//#define SW_TEST			// SW version
#define HW_TEST		// HW version

//#define LOG			// print layer result logs


/////////////////////// Layer config ///////////////////
#define image_Move 10000
#define image_Batch 1

#define label_type int
// 100(1310),
#define MNIST_SIZE 784
#define MNIST_PAD_SIZE 1024
#define MNIST_WH 28
#define MNIST_LABEL_SIZE 10
#define INPUT_SIZE 1024
#define INPUT_WH 32
#define INPUT_DEPTH 1

#define CONV_1_INPUT_SIZE 1024
#define CONV_1_INPUT_WH 32
#define CONV_1_OUTPUT_SIZE 784
#define CONV_1_OUTPUT_WH 28
#define CONV_1_TYPE 6
#define CONV_1_SIZE 25
#define CONV_1_WH 5

#define POOL_1_INPUT_WH 28
#define POOL_1_INPUT_SIZE 784
#define POOL_1_OUTPUT_WH 14
#define POOL_1_OUTPUT_SIZE 196
#define POOL_1_TYPE 6
#define POOL_1_SIZE 4


#define CONV_2_OUTPUT_WH 10
#define CONV_2_OUTPUT_SIZE 100
#define CONV_2_INPUT_SIZE 196
#define CONV_2_INPUT_WH 14
#define CONV_2_SIZE 25
#define CONV_2_TYPE 16
#define CONV_2_WH 5
#define O 1
#define X 0
static const int tbl[] = {
			O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
			O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
			O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
			X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
			X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
			X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
		};
#define POOL_2_INPUT_WH 10
#define POOL_2_OUTPUT_WH 5
#define POOL_2_TYPE 16
#define POOL_2_SIZE 4
#define POOL_2_OUTPUT_SIZE 25
#define POOL_2_INPUT_SIZE 100

#define CONV_3_OUTPUT_WH 1
#define CONV_3_INPUT_WH 5
#define CONV_3_TYPE 120
#define CONV_3_WH 5
#define CONV_3_SIZE 25
#define CONV_3_INPUT_SIZE 25
#define CONV_3_OUTPUT_SIZE 1

#define NN_INPUT_N 120

#define INPUT_NN_1_SIZE 120
#define FILTER_NN_1_SIZE 120 * 84
#define BIAS_NN_1_SIZE 84
#define OUTPUT_NN_1_SIZE 84

#define INPUT_NN_2_SIZE 84
#define FILTER_NN_2_SIZE 84 * 10
#define OUTPUT_NN_2_SIZE 10
#define BIAS_NN_2_SIZE 10

#define IMAGE_FILE "./MNIST_DATA/t10k-images.idx3-ubyte"//"./train/image.txt"
#define LABEL_FILE "./MNIST_DATA/t10k-labels.idx1-ubyte"//"./train/label.txt"
//#define LOG
//int total_cnt;
/*
#define LOG_PRINT

FILE *MNIST_ORG;
FILE *MNIST_PAD;

FILE * RESULT_CONV_1;
FILE * RESULT_POOL_1;
FILE * RESULT_CONV_2;
FILE * RESULT_POOL_2;
FILE * RESULT_CONV_3;
FILE * RESULT_FC_1;
FILE * RESULT_FC_2;

time_t start_time;
time_t end_time;

*/
#endif
