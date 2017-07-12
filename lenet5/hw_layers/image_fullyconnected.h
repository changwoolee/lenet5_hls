/*
 * image_fullyconnected.h
 *
 *  Created on: 2017. 5. 21.
 *      Author: woobes
 */

#ifndef SRC_HW_LAYERS_IMAGE_FULLYCONNECTED_H_
#define SRC_HW_LAYERS_IMAGE_FULLYCONNECTED_H_

void FULLY_CONNECTED_LAYER_1(float* input_feature, float* weights, float* bias, float* output_feature){
	for (int batch = 0; batch < image_Batch; batch++) {
			for (int j = 0; j < OUTPUT_NN_1_SIZE; j++) {
				float temp = 0;
				for (int i = 0; i < INPUT_NN_1_SIZE; i++) {
					temp += input_feature[batch*120 + i] * weights[i*84 + j];
				}
				output_feature[batch*84 + j] = tanh(temp + bias[j]);
			}
		}
}
void FULLY_CONNECTED_LAYER_2(float* input_feature, float* weights, float* bias, float* output_feature){
	for (int batch = 0; batch < image_Batch; batch++) {
		for (int j = 0; j < OUTPUT_NN_2_SIZE; j++) {
			float temp = 0;
			for (int i = 0; i < INPUT_NN_2_SIZE; i++) {
				temp += input_feature[batch*120 + i] * weights[i*84 + j];
			}
			output_feature[batch*84 + j] = tanh(temp + bias[j]);
		}
	}
}


#endif /* SRC_SW_LAYERS_IMAGE_FULLYCONNECTED_H_ */
