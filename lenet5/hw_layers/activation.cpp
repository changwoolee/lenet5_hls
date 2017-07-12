#include <lenet5/hw_layers/activation.h>


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