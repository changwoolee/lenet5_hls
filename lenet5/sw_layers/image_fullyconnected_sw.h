/*
 * image_fullyconnected.h
 *
 *  Created on: 2017. 5. 21.
 *      Author: woobes
 */

#ifndef SRC_LENET5_SW_LAYERS_IMAGE_FULLYCONNECTED_SW_H_
#define SRC_LENET5_SW_LAYERS_IMAGE_FULLYCONNECTED_SW_H_

void FULLY_CONNECTED_LAYER_1_SW(float* input_feature, float* weights, float* bias, float* output_feature){
	for (int batch = 0; batch < image_Batch; batch++) {
			for (int i = 0; i < OUTPUT_NN_1_SIZE; i++) {
				float temp = 0;
				for (int j = 0; j < INPUT_NN_1_SIZE; j++) {
					float in_val = input_feature[j];
					float w_val = weights[j*84+i];
					temp += in_val*w_val;
				}
				output_feature[batch*84 + i] = tanhf(temp + bias[i]);
			}
		}
}
void FULLY_CONNECTED_LAYER_2_SW(float* input_feature, float* weights, float* bias, float* output_feature){
	for (int batch = 0; batch < image_Batch; batch++) {
		for (int i = 0; i < OUTPUT_NN_2_SIZE; i++) {
			float temp = 0;
			for (int j = 0; j < INPUT_NN_2_SIZE; j++) {
				float in_val = input_feature[j];
				float w_val = weights[j*10+i];
				temp += in_val*w_val;//input_feature[batch*84 + j] * weights[j*10 + i];
			}
			output_feature[batch*10 + i] = tanhf(temp + bias[i]);
		}
	}
}


#endif /* SRC_LENET5_SW_LAYERS_IMAGE_FULLYCONNECTED_SW_H_ */
