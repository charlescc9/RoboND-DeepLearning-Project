## Project: Follow Me

#### 1. Background
The goal of the Follow Me project is to train a quadcopter to locate and follow a specific human (in simulation).
Accomplishing this goal required performing semantic segmentation on the images captured by the quadcopter's camera.
Semantic segmentation is process of identifying which individual pixels within an image belong to specific objects
(in this case people, and more specifically the person whom the quadcopter is following). There are a number of 
techniques used for semantic segmentation, but the current state-of-the-art is obtained by using a fully convolutional
network (FCN). 

FCNs work very well for semantic segmentation because they simultaneously perform object recognition, similarly to
regular convolutional neural networks (CNNs), and give an output of the same width and height of the input image, 
which allows for classification on a per-pixel level. This is accomplished by using an encoder-decoder architecture in 
which the image is first encoded into a high-level representation suitable for object recognition, and the decoded back 
into the same size as the image, which places the any recognized objects into the correct location within the image.

FCNs are very similar to CNNs, but there are a few different techniques used when build the layers of a FCN. The main
difference is the use of a 1x1 convolutional layer at the end of the encoder as opposed to fully connected layers.
This change is made in order to maintain spatial information; when a fully connected layer is used at the end of a CNN,
the output is flattened from 4 dimensions to 2 dimensions, which effectively erases all spatial information. A 1x1
convolution, on the other hand, will perform the same full transformation between all neurons that a fully connected 
layer does, while maintaining the exact same shape and therefore conserving spatial information. Therefore, while a
fully connected layer is very helpful for simple object recognition, a 1x1 convolutional layer is very useful for
FCNs. Another difference is the use of transposed convolutional layers to upsample the encoder's output back to
the same size as the original image. The final difference is the use of skip connections in the decoder, which help the 
FCN to learn where the detected objects are within the image by combining feature rich, lower dimensional information 
into the higher dimensional decoder layers. All of these specific techniques were used to create a high performing FCN
for semantic segmentation.  

#### 2. Network Architecture
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

The FCN can be divided into two main parts: encoder and decode. The encoder has 3 traditional convolutional layers with 
kernel size of 3 and a stride size of 2 followed by batch normalization. This increases the depth but halves the width 
and height at each layer, taking the shape from the image's original (128, 128, 3) to (16, 16, 128). After the encoder 
is the 1x1 layer, with increases the depth but leaves the width and height unchanged. The decoder then has 3 upsampling
layers that perform the inverse operation as the encoder's convolutions, doubling width and height and decreasing 
depth. The decoder as performs skip connections by concatenating the upsampled output from the previous layer with the
output from the corresponding encoder layer and then performing a convolution with batch normalization. Finally, The 
output layer is another 1x1 convolution with an output shape of (128, 128, 3), which matches the original image shape. 

#### 3. Hyperparameter Choices
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
default value of the Adam optimizer used is 0.001). This large learning rate may indicate a relative low variance in
the data and lack of local minima. The batch size, on the other hand, worked best as a relatively small 32. This small
batch size along a high number of steps per training epoch (200 being mych )   

### 4. Discussion
Todo