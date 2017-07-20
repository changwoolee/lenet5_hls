/*
 * LOG.h
 *
 *  Created on: 2017. 5. 22.
 *      Author: woobes
 */

#ifndef SRC_LOG_H_
#define SRC_LOG_H_

#include <fstream>
#include <sstream>
//#include <iostream>

// print out log file
void print_log(string filename,float* arr, int size){
	ofstream file(filename.c_str(), ios::out);
	time_t timer;
	struct tm *t;
	if(file.is_open()){
		timer = time(NULL);
		t = localtime(&timer);
		file << "\n" << "C1 log("<<t->tm_mday <<"/"<<t->tm_mon+1<<"/"<<t->tm_year+1900<<" "<<t->tm_hour<<":"<<t->tm_min<<":"<<t->tm_sec<<endl;
		for(int i=0;i<size;i++){
			file<<arr[i]<<"\n";
		}
	}
	file.close();
}
void print_log(string filename,stringstream* ss){
	ofstream file(filename.c_str(), ios::out);
	if(file.is_open()){
		file<<ss->str();
	}
	file.close();
}
#ifdef LOG
void get_log(stringstream* ss, float* input_layer, float* hconv1, float* pool1, float* hconv2, float* pool2, float* hconv3, float* hfc1, float* output){
		for(int i=0;i<32;i++){
			for(int j=0;j<32;j++){
				(*ss)<<input_layer[i*32+j]<<" ";//printf("%1.1f ",input_layer[i*32+j]);
			}
			(*ss)<<"\n";
		}
		(*ss)<<"\n";
		for(int i=0;i<6;i++){
			for(int j=0;j<28;j++){
				for(int k=0;k<28;k++){
					(*ss)<<hconv1[i*28*28+j*28+k]<<" ";//printf("%1.1f ",hconv1[i*28*28+j*28+k]);
				}
				(*ss)<<"\n";
			}
			(*ss)<<"\n";
		}

		for(int i=0;i<6;i++){
			for(int j=0;j<14;j++){
				for(int k=0;k<14;k++){
					(*ss)<<pool1[i*14*14+j*14+k]<<" ";//printf("%1.1f ",pool1[i*14*14+j*14+k]);
				}
				(*ss)<<"\n";
			}
			(*ss)<<"\n";
		}

		for(int i=0;i<16;i++){
			for(int j=0;j<10;j++){
				for(int k=0;k<10;k++){
					(*ss)<<hconv2[i*100+j*10+k]<<" ";//printf("%1.1f ",hconv2[i*100+j*10+k]);
				}
				(*ss)<<"\n";
			}
			(*ss)<<"\n";
		}
		for(int i=0;i<16;i++){
			for(int j=0;j<5;j++){
				for(int k=0;k<5;k++){
					(*ss)<<pool2[i*25+j*5+k]<<" ";//printf("%1.1f ",pool2[i*25+j*5+k]);
				}
				(*ss)<<"\n";
			}
			(*ss)<<"\n";
		}

		for(int i=0;i<120;i++){
			(*ss)<<hconv3[i]<< " ";//printf("%1.1f ",hconv3[i]);
		}
		(*ss)<<"\n";

		for(int i=0;i<84;i++){
			(*ss)<<hfc1[i]<<" ";//printf("%1.1f ",hfc1[i]);
		}
		(*ss)<<"\n";

		for(int i=0;i<10;i++){
			(*ss)<<output[i]<<" ";//printf("%f ",output[i]);
		}
		(*ss)<<"\n";

}
#endif


#endif /* SRC_LOG_H_ */
