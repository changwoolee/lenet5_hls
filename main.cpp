/*
 * main.cpp
 *
 *  Created on: 2017. 4. 11.
 *      Author: woobes
 */




//#include "image_convolution.h"
#include <vector>
#include <numeric>

#include "lenet5/lenet5.h"
#include "sdx_test.h"
#include "./MNIST_DATA/MNIST_DATA.h"
#include "LOG.h"

// load weights & biases
void load_model(string filename, float* weight, int size) {

	ifstream file(filename.c_str(), ios::in);
	if (file.is_open()) {
		for (int i = 0; i < size; i++) {
			float temp = 0.0;
			file >> temp;
			weight[i] = temp;
		}
	}else{
		cout<<"Loading model is failed : "<<filename<<endl;
	}
}
using namespace std;
int main(void){
	cout<<"-------------------------------------------------------"<<endl;
	cout<<"LeNet-5(Max pool) HW accelerator test"<<endl;
	cout<<"version 0.1.0"<<endl;
	cout<<"Original source : Acclerationg Lenet-5 (Base Version) for Default,"<<endl;
	cout<<"implemented by Constant Park, HYU, ESoCLab[Version 1.0]"<<endl;
	cout<<"HW implementated by CW Lee & JH Woo"<<endl;
	cout<<"batch : "<<image_Batch<<" test img num : "<<image_Move<<endl;
	cout<<"-------------------------------------------------------"<<endl;


	float* MNIST_IMG = (float*) malloc(image_Move*MNIST_PAD_SIZE*sizeof(float)); // MNIST TEST IMG
	int* MNIST_LABEL = (int*) malloc(image_Move*sizeof(int)); // MNIST TEST LABEL
	if(!MNIST_IMG || !MNIST_LABEL){
		cout<< "Memory allocation error(0)"<<endl;
		return 1;
	}
	// read MNIST data & label
	READ_MNIST_DATA("/mnt/LeNet5/MNIST_DATA/t10k-images.idx3-ubyte",MNIST_IMG,image_Move);
	READ_MNIST_LABEL("/mnt/LeNet5/MNIST_DATA/t10k-labels.idx1-ubyte",MNIST_LABEL,image_Move,false);
	const int conv_weight_size = CONV_1_TYPE*CONV_1_SIZE + CONV_1_TYPE + CONV_2_TYPE*CONV_1_TYPE*CONV_2_SIZE + CONV_2_TYPE + CONV_3_TYPE*CONV_2_TYPE*CONV_3_SIZE+CONV_3_TYPE;

	//float Wconv1[CONV_1_TYPE*CONV_1_SIZE];
	float* Wconv1= (float*) sds_alloc(CONV_1_TYPE*CONV_1_SIZE*sizeof(float));
	float* bconv1=(float*)sds_alloc(CONV_1_TYPE*sizeof(float));
	float* Wconv2=(float*)sds_alloc(CONV_2_TYPE*CONV_1_TYPE*CONV_2_SIZE*sizeof(float));
	float* bconv2=(float*)sds_alloc(CONV_2_TYPE*sizeof(float));
	float* Wconv3=(float*)sds_alloc(CONV_3_TYPE*CONV_2_TYPE*CONV_3_SIZE*sizeof(float));
	float* bconv3=(float*)sds_alloc(CONV_3_TYPE*sizeof(float));
	//float* Wconv3_1 = (float*) sds_alloc(16*120*25/2*sizeof(float));
	//float* Wconv3_2 = (float*) sds_alloc(16*120*25/2*sizeof(float));
	if(!Wconv1||!Wconv2||!Wconv3||!bconv1||!bconv2||!bconv3){
		cout<<"mem alloc error(1)"<<endl;
		return 1;
	}
	/*
	float* weights = (float*) sds_alloc( conv_weight_size * sizeof(float));
	float* Wconv1 = weights;
	float* bconv1 = weights + CONV_1_TYPE*CONV_1_SIZE;
	float* Wconv2 = bconv1 + CONV_1_TYPE;
	float* bconv2 = Wconv2 + CONV_2_TYPE*CONV_1_TYPE*CONV_2_SIZE;
	float* Wconv3 = bconv2 + CONV_2_TYPE;
	float* bconv3 = Wconv3 + CONV_3_TYPE*CONV_2_TYPE*CONV_3_SIZE;*/
	//float Wpool1[POOL_1_TYPE*4];
	//float bpool1[POOL_1_TYPE];
	//float Wpool2[POOL_2_TYPE*4];
	//float bpool2[POOL_2_TYPE];

	float Wfc1[FILTER_NN_1_SIZE];
	float bfc1[BIAS_NN_1_SIZE];
	float Wfc2[FILTER_NN_2_SIZE];
	float bfc2[BIAS_NN_2_SIZE];

	cout<<"Load models"<<endl;
	load_model("/mnt/LeNet5/filter/Wconv1.mdl",Wconv1,CONV_1_TYPE*CONV_1_SIZE);
	load_model("/mnt/LeNet5/filter/Wconv3.mdl",Wconv2,CONV_2_TYPE*CONV_1_TYPE*CONV_2_SIZE);
	load_model("/mnt/LeNet5/filter/Wconv5.mdl",Wconv3,CONV_3_TYPE*CONV_2_TYPE*CONV_3_SIZE);

	load_model("/mnt/LeNet5/filter/bconv1.mdl",bconv1,CONV_1_TYPE);
	load_model("/mnt/LeNet5/filter/bconv3.mdl",bconv2,CONV_2_TYPE);
	load_model("/mnt/LeNet5/filter/bconv5.mdl",bconv3,CONV_3_TYPE);

	//load_model("/mnt/LeNet5/filter/LeNet-weights_Pool_1.txt",Wpool1,POOL_1_TYPE*4);
	//load_model("/mnt/LeNet5/filter/LeNet-weights_Pool_2.txt",Wpool2,POOL_2_TYPE*4);

	//load_model("/mnt/LeNet5/filter/LeNet-weights_Pool_1_Bias.txt",bpool1,POOL_1_TYPE);
	//load_model("/mnt/LeNet5/filter/LeNet-weights_Pool_2_Bias.txt",bpool2,POOL_2_TYPE);

	load_model("/mnt/LeNet5/filter/Wfc1.mdl",Wfc1,FILTER_NN_1_SIZE);
	load_model("/mnt/LeNet5/filter/Wfc2.mdl",Wfc2,FILTER_NN_2_SIZE);

	load_model("/mnt/LeNet5/filter/bfc1.mdl",bfc1,BIAS_NN_1_SIZE);
	load_model("/mnt/LeNet5/filter/bfc2.mdl",bfc2,BIAS_NN_2_SIZE);


	cout<<"LeNet-5(HW) test start"<<endl;
	// Memory allocation
	float* input_layer	= (float*) sds_alloc(image_Batch *INPUT_WH * INPUT_WH*sizeof(float));
	float* hconv1 		= (float*) sds_alloc(image_Batch * CONV_1_TYPE * CONV_1_OUTPUT_SIZE*sizeof(float));
	float* pool1 		= (float*) sds_alloc(image_Batch * CONV_1_TYPE * POOL_1_OUTPUT_SIZE*sizeof(float));
	float* hconv2 		= (float*) sds_alloc(image_Batch * CONV_2_TYPE * CONV_2_OUTPUT_SIZE*sizeof(float));
	float* pool2 		= (float*) sds_alloc(image_Batch * CONV_2_TYPE * POOL_2_OUTPUT_SIZE*sizeof(float));
	float* hconv3 		= (float*) sds_alloc(image_Batch * CONV_3_TYPE*sizeof(float));
	float* hfc1 		= (float*) sds_alloc(image_Batch * OUTPUT_NN_1_SIZE*sizeof(float));
	float* output 		= (float*) sds_alloc(image_Batch * OUTPUT_NN_2_SIZE*sizeof(float));
	if(!input_layer || !hconv1 || !pool1 || !hconv2 || !pool2 || !hconv3 || !hfc1 || !output){
		cout<<"Memory allocation error(2)"<<endl;
		return 1;
	}

	///////////////////////////////// TEST /////////////////////////////////////////

	// save accuracies
	vector<double> result_hw, result_sw;

	double accuracy_hw, accuracy_sw;

	// cycle counters
	perf_counter hw_ctr_tot, hw_ctr_conv1, hw_ctr_conv2, hw_ctr_conv3, hw_ctr_pool1, hw_ctr_pool2, hw_ctr_fc1, hw_ctr_fc2;
	perf_counter sw_ctr_tot, sw_ctr_conv1, sw_ctr_conv2, sw_ctr_conv3, sw_ctr_pool1, sw_ctr_pool2, sw_ctr_fc1, sw_ctr_fc2;

	// test number
	int test_num = image_Move/image_Batch;

	 //HW test start
	cout<<"HW test start"<<endl;
	for(int i=0;i<test_num;i++){
		for(int batch=0;batch<image_Batch*INPUT_WH*INPUT_WH;batch++)
			input_layer[batch] = MNIST_IMG[i*MNIST_PAD_SIZE + batch];
		for(int i=0;i<32;i++){
			for(int j=0;j<32;j++){
				printf("%1.1f ",input_layer[i*32+j]);
			}
			cout<<"\n";
		}
		cout<<"\n";
		hw_ctr_tot.start();// counter for total test

		// C1 start
		hw_ctr_conv1.start(); // counter for C1 layer
		//CONVOLUTION_LAYER_1(input_layer,Wconv1,bconv1,hconv1,6*25,6);
		CONVOLUTION_LAYER_1(input_layer,Wconv1,bconv1,hconv1);
		hw_ctr_conv1.stop();
		for(int i=0;i<6;i++){
			for(int j=0;j<28;j++){
				for(int k=0;k<28;k++){
					printf("%1.1f ",hconv1[i*28*28+j*28+k]);
				}
				cout<<"\n";
			}
			cout<<"\n";
		}

		// S1 start
		hw_ctr_pool1.start();
		//POOLING_LAYER_1_SW(hconv1,Wpool1,bpool1,pool1);
		MAXPOOL_1_SW(hconv1,pool1);
		hw_ctr_pool1.stop();
		for(int i=0;i<6;i++){
			for(int j=0;j<14;j++){
				for(int k=0;k<14;k++){
					//if(pool1[i*14*14+j*14+k]!=0)
					//	cout<<"*";
					//else
					//	cout<<" ";
					printf("%1.1f ",pool1[i*14*14+j*14+k]);
				}
				cout<<"\n";
			}
			cout<<"\n";
		}
		//C2 start
		hw_ctr_conv2.start();
		//CONVOLUTION_LAYER_2(pool1,Wconv2,bconv2,hconv2,6*16*25,16);
		CONVOLUTION_LAYER_2(pool1,Wconv2,bconv2,hconv2);
		hw_ctr_conv2.stop();
		for(int i=0;i<16;i++){
					for(int j=0;j<10;j++){
						for(int k=0;k<10;k++){
							printf("%1.1f ",hconv2[i*100+j*10+k]);
						}
						cout<<"\n";
					}
					cout<<"\n";
				}
		hw_ctr_pool2.start();
		//POOLING_LAYER_2_SW(hconv2,Wpool2,bpool2,pool2);
		MAXPOOL_2_SW(hconv2,pool2);
		hw_ctr_pool2.stop();
		for(int i=0;i<16;i++){
					for(int j=0;j<5;j++){
						for(int k=0;k<5;k++){
							printf("%1.1f ",pool2[i*25+j*5+k]);
						}
						cout<<"\n";
					}
					cout<<"\n";
				}
		hw_ctr_conv3.start();
		//CONVOLUTION_LAYER_3(pool2,Wconv3,bconv3,hconv3,16*120*25,120);
		CONVOLUTION_LAYER_3(pool2,Wconv3,bconv3,hconv3);

		hw_ctr_conv3.stop();
		for(int i=0;i<120;i++){
			printf("%1.1f ",hconv3[i]);
		}
		cout<<"\n";

		hw_ctr_fc1.start();
		FULLY_CONNECTED_LAYER_1_SW(hconv3,Wfc1,bfc1,hfc1);
		hw_ctr_fc1.stop();
		for(int i=0;i<84;i++){
					printf("%1.1f ",hfc1[i]);
				}cout<<"\n";

		hw_ctr_fc2.start();
		FULLY_CONNECTED_LAYER_2_SW(hfc1,Wfc2,bfc2,output);
		hw_ctr_fc2.stop();

		hw_ctr_tot.stop();
		for(int i=0;i<10;i++){
					printf("%f ",output[i]);
				}cout<<"\n";

		result_hw.push_back(equal(MNIST_LABEL[i],argmax(output)));
	}
	// accuracy estimation
	accuracy_hw = 1.0*accumulate(result_hw.begin(),result_hw.end(),0.0)/result_hw.size();
	cout<<"HW test completed"<<endl;
	cout<<"accuracy : "<<accuracy_hw<<endl;
#ifndef HW_TEST
	// SW test
	cout<< "SW test start"<<endl;
	for(int i=0;i<test_num;i++){
		for(int batch=0;batch<image_Batch*INPUT_WH*INPUT_WH;batch++)
			input_layer[batch] = MNIST_IMG[i*MNIST_PAD_SIZE + batch];
		for(int i=0;i<32;i++){
				for(int j=0;j<32;j++){
					printf("%1.1f ",input_layer[i*32+j]);
				}
				cout<<"\n";
			}

		sw_ctr_tot.start();// counter for total test

		// C1 start
		sw_ctr_conv1.start(); // counter for C1 layer
		CONVOLUTION_LAYER_1_SW(input_layer,Wconv1,bconv1,hconv1);
		sw_ctr_conv1.stop();
		for(int i=0;i<6;i++){
					for(int j=0;j<28;j++){
						for(int k=0;k<28;k++){
							printf("%1.1f ",hconv1[i*28*28+j*28+k]);
						}
						cout<<"\n";
					}
					cout<<"\n";
				}
		// S1 start
		sw_ctr_pool1.start();
		//POOLING_LAYER_1_SW(hconv1,Wpool1,bpool1,pool1);
		MAXPOOL_1_SW(hconv1,pool1);
		sw_ctr_pool1.stop();
		for(int i=0;i<6;i++){
			for(int j=0;j<14;j++){
				for(int k=0;k<14;k++){
					//if(pool1[i*14*14+j*14+k]!=0)
					//	cout<<"*";
					//else
					//	cout<<" ";
					printf("%1.1f ",pool1[i*14*14+j*14+k]);
				}
				cout<<"\n";
			}
			cout<<"\n";
		}
		//C2 start
		sw_ctr_conv2.start();
		CONVOLUTION_LAYER_2_SW(pool1,Wconv2,bconv2,hconv2);
		sw_ctr_conv2.stop();
		for(int i=0;i<16;i++){
					for(int j=0;j<10;j++){
						for(int k=0;k<10;k++){
							printf("%1.1f ",hconv2[i*100+j*10+k]);
						}
						cout<<"\n";
					}
					cout<<"\n";
				}
		sw_ctr_pool2.start();
		//POOLING_LAYER_2_SW(hconv2,Wpool2,bpool2,pool2);
		MAXPOOL_2_SW(hconv2,pool2);
		sw_ctr_pool2.stop();
		for(int i=0;i<16;i++){
					for(int j=0;j<5;j++){
						for(int k=0;k<5;k++){
							printf("%1.1f ",pool2[i*25+j*5+k]);
						}
						cout<<"\n";
					}
					cout<<"\n";
				}
		sw_ctr_conv3.start();
		CONVOLUTION_LAYER_3_SW(pool2,Wconv3,bconv3,hconv3);
		sw_ctr_conv3.stop();

		for(int i=0;i<120;i++){
				printf("%1.1f ",hconv3[i]);
			}
		sw_ctr_fc1.start();
		FULLY_CONNECTED_LAYER_1_SW(hconv3,Wfc1,bfc1,hfc1);
		sw_ctr_fc1.stop();
		for(int i=0;i<84;i++){
					printf("%1.1f ",hfc1[i]);
				}cout<<"\n";

		sw_ctr_fc2.start();
		FULLY_CONNECTED_LAYER_2_SW(hfc1,Wfc2,bfc2,output);
		sw_ctr_fc2.stop();

		sw_ctr_tot.stop();

		for(int i=0;i<10;i++){
					printf("%f ",output[i]);
				}cout<<"\n";
		result_sw.push_back(equal(MNIST_LABEL[i],argmax(output)));
	}
	accuracy_sw = 1.0*accumulate(result_sw.begin(),result_sw.end(),0.0)/result_sw.size();
	cout<<"SW test completed"<<endl;
	cout<<"accuracy : "<<accuracy_sw<<endl;
#endif
	sds_free(input_layer);
	sds_free(hconv1);
	sds_free(hconv2);
	sds_free(hconv3);
	sds_free(pool1);
	sds_free(pool2);
	sds_free(output);
	sds_free(Wconv3);
	sds_free(bconv3);

	sds_free(Wconv1);
	sds_free(Wconv2);
	sds_free(bconv1);
	sds_free(bconv2);
	//sds_free(weights);
	free(MNIST_IMG);
	free(MNIST_LABEL);
	
#ifndef HW_TEST
	stringstream ss;
	ss <<"----------------------------------------------------------------------------"<<endl;
	double speedup_c1 = (double) sw_ctr_conv1.avg_cpu_cycles() / (double) hw_ctr_conv1.avg_cpu_cycles();
	ss <<"Average number of CPU cycles running C1 in software: "
		 <<sw_ctr_conv1.avg_cpu_cycles()<<endl;
	ss <<"Average number of CPU cycles running C1 in hardware: "
		 <<hw_ctr_conv1.avg_cpu_cycles()<<endl;
	ss <<"Speed up: "<<speedup_c1<<endl;
	ss <<"----------------------------------------------------------------------------"<<endl;
	double speedup_s1 = (double) sw_ctr_pool1.avg_cpu_cycles() / (double) hw_ctr_pool1.avg_cpu_cycles();
	ss <<"Average number of CPU cycles running S1 in software: "
		 <<sw_ctr_pool1.avg_cpu_cycles()<<endl;
	ss <<"Average number of CPU cycles running S1 in hardware: "
		 <<hw_ctr_pool1.avg_cpu_cycles()<<endl;
	ss <<"Speed up: "<<speedup_s1<<endl;
	ss <<"----------------------------------------------------------------------------"<<endl;
	double speedup_c2 = (double) sw_ctr_conv2.avg_cpu_cycles() / (double) hw_ctr_conv2.avg_cpu_cycles();
	ss <<"Average number of CPU cycles running C2 in software: "
		 <<sw_ctr_conv2.avg_cpu_cycles()<<endl;
	ss <<"Average number of CPU cycles running C2 in hardware: "
		 <<hw_ctr_conv2.avg_cpu_cycles()<<endl;
	ss <<"Speed up: "<<speedup_c2<<endl;
	ss <<"----------------------------------------------------------------------------"<<endl;
	double speedup_s2 = (double) sw_ctr_pool2.avg_cpu_cycles() / (double) hw_ctr_pool2.avg_cpu_cycles();
	ss <<"Average number of CPU cycles running S2 in software: "
		 <<sw_ctr_pool2.avg_cpu_cycles()<<endl;
	ss <<"Average number of CPU cycles running S2 in hardware: "
		 <<hw_ctr_pool2.avg_cpu_cycles()<<endl;
	ss <<"Speed up: "<<speedup_s2<<endl;
	ss <<"----------------------------------------------------------------------------"<<endl;
	double speedup_c3 = (double) sw_ctr_conv3.avg_cpu_cycles() / (double) hw_ctr_conv3.avg_cpu_cycles();
	ss <<"Average number of CPU cycles running C3 in software: "
		 <<sw_ctr_conv3.avg_cpu_cycles()<<endl;
	ss <<"Average number of CPU cycles running C3 in hardware: "
		 <<hw_ctr_conv3.avg_cpu_cycles()<<endl;
	ss <<"Speed up: "<<speedup_c3<<endl;
	ss <<"----------------------------------------------------------------------------"<<endl;
	double speedup_tot = (double) sw_ctr_tot.avg_cpu_cycles() / (double) hw_ctr_tot.avg_cpu_cycles();
	ss <<"Average number of CPU cycles running total model in software: "
		 <<sw_ctr_tot.avg_cpu_cycles()<<endl;
	ss <<"Average number of CPU cycles running total model in hardware: "
		 <<hw_ctr_tot.avg_cpu_cycles()<<endl;
	ss <<"Speed up: "<<speedup_tot<<endl;
	ss <<"----------------------------------------------------------------------------"<<endl;
	cout<<ss.str();

	print_log("/mnt/model_log/performance.log",&ss);
#endif
	cout<<"Test Completed"<<endl;
	return 0;

}
