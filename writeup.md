## Project: Follow Me

#### 1. Introduction
The goal of this project was to train a quadcopter to locate and follow a specific human (in simulation).
Accomplishing this goal required performing semantic segmentation on the images captured by the quadcopter's camera.
Semantic segmentation is the process of identifying which individual pixels within an image belong to specific objects
(in this case people, and more specifically the person whom the quadcopter is following). The current state-of-the-art 
technique for semantic segmentation is a fully convolutional network (FCN). 

#### 2. Architecture
##### 2.1 Summary Table
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

##### 2.2 Encoder-Decoder Architecture for Image Manipulation
FCNs work very well for semantic segmentation because they use an encoder-decoder architecture in which the image is first 
encoded into a high-level representation suitable for object recognition, and then decoded back into the same size as 
the image while maintaining the objects's spatial locations. The encoder is build by from multiple convolutional layers,
ending with a 1x1 convolution, and extracts features from the image. The decoder is build from either transposed
convolution layers or layers of upsampling followed by convolution, producing an output with the same dimensionality
as the input image. The decoder often also uses skip connections to integrate encoder feature information with the 
previous decoder layer. This encoder-decoder architecture works very well for semantic segmentation because is achieves
both object recognition and segmentation of those objects within the original image. In general, encoder and decoding
images should be used whenever information needs to be extracted from the image while also localized within it. The largest
problem that may occur when using this encoder-decoder architecture is that it may lead to very large networks that are
computationally expensive to train. 

##### 2.3 1x1 Convolution Layer
Once of the main differences between FCNs and CNNs is the use of 1x1 convolution layer as opposed a fully connected layer.
CNNs use one or more fully connected layer after all the convolution layers and subsequently perform softmax operation
in order to get class probabilities. However, using a fully connected layer requires flattening the convolutional layers's output
from 4 dimensions to 2 dimensions, which erases all spatial information. This is not an issue when using a CNN for 
object classification, as the object's location within the image is not relevant. However, when performing semantic
segmentation, the object's spatial information must be maintained in order to output a accurate per-pixel segmented image.
A 1x1 convolution is used instead. A 1x1 convolution is simply one with a kernel and stride sizes of 1, which results 
in an output with the same height and width. This provides the model with dimensionality reduction without losing spatial
information. Consequently, fully connected layers are useful when doing classification tasks that don't require spatial
information, with 1x1 convolutions are useful when reducing dimensionality while preserving spatial information, for
example when performing semantic segmentation.

##### 2.2 Layers and Parameter Choices 
The FCN architecture can be divided into two main parts: encoder and decode. The encoder has 3 traditional convolutional layers with 
kernel size of 3 and a stride size of 2 followed by batch normalization. This increases the depth but halves the width 
and height at each layer, taking the shape from the image's original (128, 128, 3) to (16, 16, 128). The encoder is then
followed by a 1x1 convolution layer. The first three layers utilize separable convolutional layers, which perform only
one convolution for each input channel. This reduces the FCN's total number of parameters greatly, thus speeding up training.
Each layer (as well as the decoder layers) also utilize batch normalization, which is the process of normalizing, or 
reducing variance, for each minibatch during training. Batch normalization helps to regularize the network, which helps
combat overfitting and speed training.

The decoder has 3 upsampling layers that perform the inverse operations as the encoder's convolutions, doubling width and height and decreasing 
depth. This is accomplished through bilinear upsampling, which created new pixels (in this case a 2x2 grid for each pixel)
from a weighted average of the nearest known pixels. The decoder also performs skip connections by concatenating the 
upsampled output from the previous layer with the output from the corresponding encoder layer. This helps the decoder
maintain the encoder's extracted features when increasing dimensionality. Each decoder layer then performs two convolution
layers with batch normalization to learn these features. Finally, The  output layer is another 1x1 convolution with an 
output shape of (128, 128, 3), which matches the original image shape, allowing for accurate per-pixel semantic segmentation.

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

#### 4. Discussion
This FCN achieved a final 42% IoU on the evaluation dataset. The FCN would have probably performed better given more 
training data and a deeper architecture. This FCN architecture very generic, and would work well if asked to recognize
objects other than humans. The data, however, was specifically labeled with the pixel positions of humans, and
therefore would not generalize to other objects. The data would require relabeling to be useful in other scenarios.