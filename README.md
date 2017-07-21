LeNet-5 in HLS
===========
This repository is about my graduate report, implementing LeNet-5 in Vivado High Level Synthesis 2016.4 & Vivado   SDSoC 2016.4


![lenet5](https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/img/lenet.png "LeNet-5")




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

  


## Accuracy  
	SW accuracy : 98.63%    
	HW accuracy : 98.63%  

## Runtime  
	# of images : 10,000, batch size : 1  
	
	SW runtime  : 59.4456 seconds  
	HW runtime  : 16.3954 seconds  

	speedup : x3.63 faster 



