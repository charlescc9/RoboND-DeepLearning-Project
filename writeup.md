## Project: Follow Me

#### 1. Background
The goal of this project was to train a quadcopter to locate and follow a specific human (in simulation).
Accomplishing this goal required performing semantic segmentation on the images captured by the quadcopter's camera.
Semantic segmentation is the process of identifying which individual pixels within an image belong to specific objects
(in this case people, and more specifically the person whom the quadcopter is following). The current state-of-the-art 
technique for semantic segmentation is a fully convolutional network (FCN). 

FCNs work very well for semantic segmentation because they simultaneously perform object recognition, similarly to
regular convolutional neural networks (CNNs), and locate those objects at specific pixel locations in the image. 
This is accomplished by using an encoder-decoder architecture in which the image is first encoded into a high-level 
representation suitable for object recognition, and then decoded back into the same size as the image while maintaining 
the objects's spatial locations.

FCNs are very similar to CNNs, but there are a few different techniques used when build the layers of a FCN. The main
difference is the use of a 1x1 convolutional layer as opposed to fully connected layers.
This change is made in order to maintain spatial information; when a fully connected layer is used at the end of a CNN,
the output is flattened from 4 dimensions to 2 dimensions, which effectively erases all spatial information. A 1x1
convolution, on the other hand, will perform the same full transformation between all neurons, while maintaining the 
same shape and therefore conserving spatial information. Therefore, while a
fully connected layer is very helpful for simple object recognition, a 1x1 convolutional layer is very useful for
semantic segmentation. Another difference is the use of upsampling in the decoder to get the encoder's output back to
the same size as the original image. The final difference is the use of skip connections in the decoder, which help the 
FCN to learn where the detected objects are within the image by combining feature rich information from the encoder 
with the higher dimensional decoder layers. All of these specific techniques were used to create a high performing FCN
for semantic segmentation.  

#### 2. Architecture
Layer | Type | Filters | Kernel | Stride | Shape Change
--- | --- | --- | --- | --- | ---
1 | Encoder | 32 | 3 | 2 | (128, 128, 3) &rarr; (64, 64, 32)
2 | Encoder | 64 | 3 | 2 | (64, 64, 32) &rarr; (32, 32, 64)
3 | Encoder | 128 | 3 | 2 | (32, 32, 64) &rarr; (16, 16, 128)
4 | 1x1 | 256 | 1 | 1 | (16, 16, 128) &rarr; (16, 16, 256)
5 | Decoder | 128 | 3 | 1 | (16, 16, 256) &rarr; (32, 32, 128)
6 | Decoder | 64 | 3 | 1 | (32, 32, 128) &rarr; (64, 64, 64)
7 | Decoder | 32 | 3 | 1 | (64, 64, 64) &rarr; (128, 128, 32)
8 | Output | 3 | 1 | 1 | (128, 128, 32) &rarr; (128, 128, 3)

The FCN architecture can be divided into two main parts: encoder and decode. The encoder has 3 traditional convolutional layers with 
kernel size of 3 and a stride size of 2 followed by batch normalization. This increases the depth but halves the width 
and height at each layer, taking the shape from the image's original (128, 128, 3) to (16, 16, 128). After the encoder 
is the 1x1 layer, with increases the depth but leaves the width and height unchanged. The decoder then has 3 upsampling
layers that perform the inverse operations as the encoder's convolutions, doubling width and height and decreasing 
depth. The decoder also performs skip connections by concatenating the upsampled output from the previous layer with the
output from the corresponding encoder layer and then performing a convolution with batch normalization. Finally, The 
output layer is another 1x1 convolution with an output shape of (128, 128, 3), which matches the original image shape.

Concerning the architecture parameters, the kernel sizes was chosen simply based on a conventional value of 3 for the encoder and decoder
and a necessary value of 1 for the 1x1 layers. The number of filters was largely chosen to be the largest number possible while the
FCN could fit on GPU memory. The number of encoder and decoder layers, 3, was a hyperparameter that was chosen 
experimentally. The FCN was implemented primarily in the `encoder_block`, `decoder_block`, and `fcn_model` functions.

#### 3. Hyperparameters
Hyperparameter | Value
--- | ---
Learning rate | 0.01
Batch size | 32
Number of epochs | 25
Number of steps per epoch (training) | 200
Number of steps per epoch (validation) | 50
Number of workers | 2

During the training process, a number of hyperparameters were tuned in order to obtain optimal performance. This tuning
was performed largely by intuition and brute force. The learning rate ended up being a relatively large 0.01 (the
default value of the Adam optimizer used is 0.001), which may indicate a relative low variance in
the data. The batch size, on the other hand, worked best as a relatively small 32. This small
batch size along a relatively high number of steps per training epoch (200 being higher than 130, which is the
conventional training set size divided by the batch size) and decently high number of epochs, 25, means that the FCN
needed to train on many minibatches to converge optimally. The final two hyperparameters, number of steps per epoch for
validation and number of workers, were system parameters that did not actually affect training.  

### 4. Discussion
This FCN achieved a final 42% IoU on the evaluation dataset. The FCN would have probably performed better given more 
training data and a deeper architecture. This FCN architecture very generic, and would work well if asked to recognize
objects other than humans. The data, however, was specifically labeled with the pixel positions of humans, and
therefore would not generalize to other objects. The data would require relabeling to be useful in other scenarios.