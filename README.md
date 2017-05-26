LeNet-5[^1] in HLS
===========
This repository is about my graduate report, implementing LeNet-5 in High Level Synthesis


![lenet5](https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/img/lenet.png "LeNet-5")



[^1]: Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, November 1998

_TO-DO_
-----
_1. Implement C3, S1, S2 layers_



### Convolution Layer 1

LUT usage : 67% -> 28%

sw cycle : 1511361  
hw cycle :  341729  
speed up : x4.42  

### Convolution Layer 2


sw cycle : 1807944  
hw cycle :  200834  
speed up : x9.00


### Convolution Layer 3

not implemented yet

