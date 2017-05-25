#ifndef SRC_MNIST_DATA_H_
#define SRC_MNIST_DATA_H_
#include <iostream>
#include <fstream>

#define TRUE 1
#define FALSE 0

using namespace std;

int ReverseInt(int i) {
	
	
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;

}


// Read MNIST Data to padded array
void READ_MNIST_DATA(string filename, float* arr, int image_num=image_Move) {
	ifstream file(filename.c_str(), ios::binary);
	cout << "Read MNIST DATA..."<< endl;
	if (file.is_open())
	{
		
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		int batch = number_of_images > image_num ? image_num : number_of_images;
		for (int i = 0; i<batch; ++i)
		{
			for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					float _temp = ((float)temp) / 255;
					
					arr[i*(n_rows+4)*(n_cols+4) + r*(n_rows+4) + c+2] = _temp;
				}
			}
		}
		cout << "MNIST DATA is loaded" << endl;

	}
	else {
		cout << "Failed to read MNIST DATA" << endl;
	}


}
template <typename T>
void READ_MNIST_LABEL(string filename, T* label, int image_num=10000, int one_hot = TRUE)
{
	ifstream file(filename.c_str(), ios::binary);
	cout << "Read MNIST Label..." << endl;
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		int batch = number_of_images > image_num ? image_num : number_of_images;
		for (int i = 0; i<batch; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (one_hot) 
			{
				label[i * 10 + (int)temp] = 1;
			}
			else
			{
				label[i] = (T)temp;
			}
			
		}
		cout << "MNIST Label is loaded" << endl;
	}
	else
	{
		cout << "Failed to read MNIST Label" << endl;
	}
//	file.close();
}
#endif
