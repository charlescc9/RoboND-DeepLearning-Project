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
Todo

#### 3. Hyperparameter Choices
Todo

### 4. Discussion
Todo