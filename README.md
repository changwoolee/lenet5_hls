LeNet-5 in HLS
===========
This repository is about our undergraduate graduation project, implementing LeNet-5 by using Vivado High Level Synthesis 2016.4 & Vivado   SDSoC 2016.4


![lenet5](https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/img/lenet.png "LeNet-5")


## Win 10 Test App
You can test the accelerator by your own handwritten digits image.  

### Youtube Video

[![Youtube Video Here](http://cfile21.uf.tistory.com/image/99C6A7335A1524F20AFF26)](https://youtu.be/C7MUhBBczss)

If you want to test the app, follow these instruction

1. Configure the IP address of Zedboard.  
```
	username@Zedboard:~# ifconfig
```
2. Start .elf file with port name argument (in here, 5555 is port name)    
```
	username@Zedboard:~# lenet5_test.elf 5555
```
3. Start the win 10 test application and input the IP address & port name.
4. Press connect
5. Open image file

I did not put a zoom in/out function to the app, so please suit the image size. 

## Model description
Used model is LeNet5-Like Deep CNN  
Input : -1.0 to 1.0  
Conv1 : 1x32x32 -> 6x28x28, ksize = 1x6x5x5, stride = 1  
Pool1 : 6x28x28 -> 6x14x14, average pooling, window size = 2x2, stride = 2  
Conv2 : 6x14x14 -> 16x10x10, ksize = 6x16x25, stride = 1  
Pool2 : 16x10x10 -> 16x5x5, average pooling, window size = 2x2, stride = 2  
Conv3 : 16x5x5 -> 120x1x1, ksize = 16x120x25, stride = 1  
FC1 : 120x84  
FC2 : 84x10    

## Environments
We used Zedboard(Zynq 7z020) for testing.  

HW Functions : CONVOLUTION_ LAYER_ 1, CONVOLUTION_ LAYER_ 2, and CONVOLUTION_ LAYER_ 3, Clk freq set as 100MHz.


## Accuracy  
	SW accuracy : 98.63% (single precision fp)    
	HW accuracy : 98.63% (single precision fp)  

## Runtime  
	# of images : 10,000, batch size : 1  
	
	SW runtime  : 59.4456 seconds  
	HW runtime  : 16.3954 seconds  

	speedup : 3.63x faster 

## Contributors
* Changwoo Lee (Hanyang University, Seoul, South Korea)
* Jeonghyun Woo (Hanyang University, Seoul, South Korea)


