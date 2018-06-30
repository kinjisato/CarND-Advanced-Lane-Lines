# CarND-Advanced-Lane-Lines
Udacity CarND-Advanced-Lane-Lines submit

This repository has following files.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 66 x 200 x3 YUV image (normalized)  							| 
| Convolution 5x5     	| 2x2 stride, padding 'VALID', 24 depth 	|
| Activation					|	ReLU									|
| Convolution 5x5     	| 2x2 stride, padding 'VALID', 36 depth 	|
| Activation					|	ReLU									|
| Convolution 5x5     	| 2x2 stride, padding 'VALID', 48 depth 	|
