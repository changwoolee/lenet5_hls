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
			for (int j = 0; j < OUTPUT_NN_1_SIZE; j++) {
				float temp = 0;
				for (int i = 0; i < INPUT_NN_1_SIZE; i++) {
					temp += input_feature[batch*120 + i] * weights[i*84 + j];
				}
				output_feature[batch*84 + j] = tanhf(temp + bias[j]);
			}
		}
}
void FULLY_CONNECTED_LAYER_2_SW(float* input_feature, float* weights, float* bias, float* output_feature){
	for (int batch = 0; batch < image_Batch; batch++) {
		for (int j = 0; j < OUTPUT_NN_2_SIZE; j++) {
			float temp = 0;
			for (int i = 0; i < INPUT_NN_2_SIZE; i++) {
				temp += input_feature[batch*120 + i] * weights[i*84 + j];
			}
			output_feature[batch*84 + j] = tanhf(temp + bias[j]);
		}
	}
}


#endif /* SRC_LENET5_SW_LAYERS_IMAGE_FULLYCONNECTED_SW_H_ */
