# Breast-Tumour-Segementation-UNet_Architecture

## Critical Components of a CNN 

1. Convolutional Layers: CNNs comprise a collection of learnable filters (kernels) convolved with the input picture or feature maps. Each filter applies element-wise multiplication and summing to produce a feature map highlighting specific patterns or local features in the input. These filters can capture many visual elements, such as edges, corners, and textures.

2. Pooling Layers: Create the feature maps by the convolutional layers that are downsampled using pooling layers. Pooling reduces the spatial dimensions of the feature maps while maintaining the most critical information, lowering the computational complexity of succeeding layers and making the model more resistant to input fluctuations. The most common pooling operation is max pooling, which takes the most significant value within a given neighborhood.

3. Activation Functions: Introduce the Non-linearity into the CNN model using activation functions. Apply them to the outputs of convolutional or pooling layers element by element, allowing the network to understand complicated associations and make non-linear decisions. Because of its simplicity and efficiency in addressing the vanishing gradient problem, the Rectified Linear Unit (ReLU) activation function is common in CNNs.

4. Fully Connected Layers: Fully connected layers, also called dense layers, use the retrieved features to complete the final classification or regression operation. They connect every neuron in one layer to every neuron in the next, allowing the network to learn global representations and make high-level judgments based on the previous layers’ combined input.

## Need for a Fully Connected Network

Traditional CNNs are generally intended for image classification jobs in which a single label is assigned to the whole input image. On the other hand, traditional CNN architectures have problems with finer-grained tasks like semantic segmentation, in which each pixel of an image must be sorted into various classes or regions. Fully Convolutional Networks (FCNs) come into play here.

## Limitations of Traditional CNN Architectures in Segmentation Tasks

### Loss of Spatial Information: 

Traditional CNNs use pooling layers to gradually reduce the spatial dimensionality of feature maps. While this downsampling helps capture high-level features, it results in a loss of spatial information, making it difficult to precisely detect and split objects at the pixel level.

### Fixed Input Size:

CNN architectures are often built to accept images of a specific size. However, the input images might have various dimensions in segmentation tasks, making variable-sized inputs challenging to manage with typical CNNs.

### Limited Localisation Accuracy:

Traditional CNNs often use fully connected layers at the end to provide a fixed-size output vector for classification. Because they do not retain spatial information, they cannot precisely localize objects or regions within the image.

# Few Important Terms 

## Feature Map

A feature map, also known as an activation map, is a 3-dimensional array resulting from applying a set of filters (kernels) to an input image or intermediate layer in a convolutional neural network (CNN).

Here's how feature maps are generated:

Convolution Operation: In a CNN, convolutional layers consist of learnable filters or kernels that slide over the input image or feature map. Each filter computes a dot product between its weights and a small region of the input data, producing a single value in the output feature map.

Multiple Filters: Typically, a convolutional layer consists of multiple filters. Each filter is responsible for detecting different features or patterns in the input data. For example, one filter might detect edges, another might detect textures, and so on.

Depth Dimension: The depth of the feature map corresponds to the number of filters applied in the convolutional layer. Each filter produces a 2-dimensional activation map, and stacking these activation maps along the depth dimension creates the final 3-dimensional feature map.


## Segmentation Map

A segmentation map, also known as a semantic segmentation map, is an image where each pixel is labeled with a class label representing the category of the object it belongs to. 

## Transposed Convolution 

Transposed convolution, also known as fractionally strided convolution or deconvolution, is a technique used in convolutional neural networks (CNNs) for upsampling or increasing the spatial resolution of feature maps.

In a traditional convolutional layer, we apply a filter/kernel to an input feature map to extract features. Transposed convolution works in the opposite direction – it increases the spatial resolution of the input by applying a kernel to it. This operation effectively "upsamples" the feature map.

Here's how it works:

Padding: First, zeros are typically added to the input feature map to increase its spatial dimensions. This step helps control the spatial size of the output feature map.

Convolution: Next, a convolution operation is applied to the padded feature map using a learnable kernel (often referred to as the transposed convolution kernel or deconvolution kernel). This kernel is learned during the training process like any other parameter in the network.

Stride: During the convolution operation, the stride determines the spacing between the positions where the kernel is applied on the input feature map. A stride of 1 means the kernel moves one pixel at a time, while a stride of 2 means it moves two pixels at a time, effectively reducing the spatial dimensions of the output feature map.

## Skip Connections 

Skip connections, also known as shortcut connections or residual connections, are connections in neural network architectures that allow information to bypass certain layers and be passed deeper into the network. They are commonly used in deep convolutional neural networks (CNNs) and particularly popular in architectures like ResNet.

Here's how skip connections work:

Identity Mapping: In a typical neural network layer, the output of one layer is directly fed as input to the next layer. With skip connections, the output of one layer is added to the output of another layer located deeper in the network.

Preserving Information: Skip connections help in preserving information and gradients throughout the network. They allow gradients to flow directly backward through the network without passing through potentially numerous non-linear activation functions, which can mitigate the vanishing gradient problem commonly encountered in deep networks.

# Fully Convolutional Networks (FCNs) as a Solution for Semantic Segmentation

By working exclusively on convolutional layers and maintaining spatial information throughout the network, Fully Convolutional Networks (FCNs) address the constraints of classic CNN architectures in segmentation tasks. FCNs are intended to make pixel-by-pixel predictions, with each pixel in the input image assigned a label or class. FCNs enable the construction of a dense segmentation map with pixel-level forecasts by upsampling the feature maps. Transposed convolutions (also known as deconvolutions or upsampling layers) are used to replace the completely linked layers after the CNN design. The spatial resolution of the feature maps is increased by transposed convolutions, allowing them to be the same size as the input image.

During upsampling, FCNs generally use skip connections, bypassing specific layers and directly linking lower-level feature maps with higher-level ones. These skip relationships aid in preserving fine-grained details and contextual information, boosting the segmented regions’ localization accuracy. FCNs are extremely effective in various segmentation applications, including medical picture segmentation, scene parsing, and instance segmentation. It can now handle input images of various sizes, provide pixel-level predictions, and keep spatial information across the network by leveraging FCNs for semantic segmentation

# Understanding the UNet Architecture 

## Flaws of the previous processes 

### Manual Annotation:

Manual annotation entails sketching and marking image boundaries or regions of interest. While this method produces reliable segmentation results, it is time-consuming, labor-intensive, and susceptible to human mistakes. Manual annotation is not scalable for large datasets, and maintaining consistency and inter-annotator agreement is difficult, especially in sophisticated segmentation tasks.

### Pixel-wise Classification: 

Another common approach is pixel-wise classification, in which each pixel in an image is classified independently, generally using algorithms such as decision trees, support vector machines (SVM), or random forests. Pixel-wise categorization, on the other hand, struggles to capture global context and dependencies among surrounding pixels, resulting in over- or under-segmentation problems. It cannot consider spatial relationships and frequently fails to offer accurate object boundaries.

## How UNet Architecture Overcomes this issue 

### End-to-End Learning:
UNET takes an end-to-end learning technique, which means it learns to segment images directly from input-output pairs without user annotation. UNET can automatically extract key features and execute accurate segmentation by training on a large labeled dataset, removing the need for labor-intensive manual annotation.

### Fully Convolutional Architecture:
UNET is based on a fully convolutional architecture, which implies that it is entirely made up of convolutional layers and does not include any fully connected layers. This architecture enables UNET to function on input images of any size, increasing its flexibility and adaptability to various segmentation tasks and input variations.

### U-shaped Architecture with Skip Connections:
The network’s characteristic architecture includes an encoding path (contracting path) and a decoding path (expanding path), allowing it to collect local information and global context. Skip connections bridge the gap between the encoding and decoding paths, maintaining critical information from previous layers and allowing for more precise segmentation.

### Contextual Information and Localisation:
The skip connections are used by UNET to aggregate multi-scale feature maps from multiple layers, allowing the network to absorb contextual information and capture details at different levels of abstraction. This information integration improves localization accuracy, allowing for exact object boundaries and accurate segmentation results.

### Data Augmentation and Regularization:
UNET employs data augmentation and regularisation techniques to improve its resilience and generalization ability during training. To increase the diversity of the training data, data augmentation entails adding numerous transformations to the training images, such as rotations, flips, scaling, and deformations. Regularisation techniques such as dropout and batch normalization prevent overfitting and improve model performance on unknown data.

# UNet Architecture 

<img width="473" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/0a94f7c5-2283-41cf-bf24-78f68cecfe23">


## Introduction 

UNET is a fully convolutional neural network (FCN) architecture built for image segmentation applications. It was first proposed in 2015 by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNET is frequently utilized for its accuracy in picture segmentation and has become a popular choice in various medical imaging applications. UNET combines an encoding path, also called the contracting path, with a decoding path called the expanding path. The architecture is named after its U-shaped look when depicted in a diagram. Because of this U-shaped architecture, the network can record both local features and global context, resulting in exact segmentation results.

<img width="896" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/b690f84b-3cb5-4a4b-89ac-33a34cd6084a">


## Critical Components of the UNET Architecture

## Contracting Path (Encoding Path):
UNET’s contracting path comprises convolutional layers followed by max pooling operations. This method captures high-resolution, low-level characteristics by gradually lowering the spatial dimensions of the input image. The channels are doubled after every downsampling operation to compensate for loss of spatial information.

<img width="922" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/5fc24e05-1365-4d76-bdd2-5ce28aece906">


## Expanding Path (Decoding Path):
Transposed convolutions, also known as deconvolutions or upsampling layers, are used for upsampling the feature maps from the encoding path in the UNET expansion path. The feature maps’ spatial resolution is increased during the upsampling phase, allowing the network to reconstitute a dense segmentation map.

<img width="928" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/f52cb6a9-cc00-4694-9383-798fd0225c09">


## Skip Connections:
Skip connections are used in UNET to connect matching layers from encoding to decoding paths. These links enable the network to collect both local and global data. The network retains essential spatial information and improves segmentation accuracy by integrating feature maps from earlier layers with those in the decoding route.

<img width="896" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/7bc20659-c7ca-440a-915e-a5842bd5b8b4">


## Concatenation:
Concatenation is commonly used to implement skip connections in UNET. The feature maps from the encoding path are concatenated with the upsampled feature maps from the decoding path during the upsampling procedure. This concatenation allows the network to incorporate multi-scale information for appropriate segmentation, exploiting high-level context and low-level features.

<img width="611" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/b9285627-da38-4607-b6ad-841cd3d31953">


## Fully Convolutional Layers:
UNET comprises convolutional layers with no fully connected layers. This convolutional architecture enables UNET to handle images of unlimited sizes while preserving spatial information across the network, making it flexible and adaptable to various segmentation tasks.

<img width="888" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/9e0b5322-7944-4f46-bf8e-46986723eb94">

This is where the Encoder switches into Decoder 


# Deep Dive in the Architecture 

## Convolutional Layers

The encoding process begins with a set of convolutional layers. Convolutional layers extract information at multiple scales by applying a set of learnable filters to the input image. These filters operate on the local receptive field, allowing the network to catch spatial patterns and minor features. With each convolutional layer, the depth of the feature maps grows, allowing the network to learn more complicated representations.

## Activation Function

Following each convolutional layer, an activation function such as the Rectified Linear Unit (ReLU) is applied element by element to induce non-linearity into the network. The activation function aids the network in learning non-linear correlations between input images and retrieved features.

## Pooling Layers

Pooling layers are used after the convolutional layers to reduce the spatial dimensionality of the feature maps. The operations, such as max pooling, divide feature maps into non-overlapping regions and keep only the maximum value inside each zone. It reduces the spatial resolution by down-sampling feature maps, allowing the network to capture more abstract and higher-level data.

The encoding path’s job is to capture features at various scales and levels of abstraction in a hierarchical manner. The encoding process focuses on extracting global context and high-level information as the spatial dimensions decrease.


