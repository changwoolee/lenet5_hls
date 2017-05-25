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
#include <iostream>

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

#endif /* SRC_LOG_H_ */
