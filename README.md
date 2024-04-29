# Breast-Tumour-Segementation-UNet_Architecture

## Critical Components of a CNN 

1. Convolutional Layers: CNNs comprise a collection of learnable filters (kernels) convolved with the input picture or feature maps. Each filter applies element-wise multiplication and summing to produce a feature map highlighting specific patterns or local features in the input. These filters can capture many visual elements, such as edges, corners, and textures.

2. Pooling Layers: Create the feature maps by the convolutional layers that are downsampled using pooling layers. Pooling reduces the spatial dimensions of the feature maps while maintaining the most critical information, lowering the computational complexity of succeeding layers and making the model more resistant to input fluctuations. The most common pooling operation is max pooling, which takes the most significant value within a given neighborhood.

3. Activation Functions: Introduce the Non-linearity into the CNN model using activation functions. Apply them to the outputs of convolutional or pooling layers element by element, allowing the network to understand complicated associations and make non-linear decisions. Because of its simplicity and efficiency in addressing the vanishing gradient problem, the Rectified Linear Unit (ReLU) activation function is common in CNNs.

4. Fully Connected Layers: Fully connected layers, also called dense layers, use the retrieved features to complete the final classification or regression operation. They connect every neuron in one layer to every neuron in the next, allowing the network to learn global representations and make high-level judgments based on the previous layers’ combined input.

<img width="533" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/74e43f1c-fea6-4ea4-bead-1fe80f12b6e2">


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

https://vigneshkumarkorakki.medium.com/mathematics-in-transposed-convolution-explained-vignesh-kumar-korakki-bf133c74958

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

## Skip Connections

The availability of skip connections that connect appropriate levels from the encoding path to the decoding path is one of the UNET architecture’s distinguishing features. These skip links are critical in maintaining key data during the encoding process.

Feature maps from prior layers collect local details and fine-grained information during the encoding path. These feature maps are concatenated with the upsampled feature maps in the decoding pipeline utilizing skip connections. This allows the network to incorporate multi-scale data, low-level features and high-level context into the segmentation process.

By conserving spatial information from prior layers, UNET can reliably localize objects and keep finer details in segmentation results. UNET’s skip connections aid in addressing the issue of information loss caused by downsampling. The skip links allow for more excellent local and global information integration, improving segmentation performance overall.

To summarise, the UNET encoding approach is critical for capturing high-level characteristics and lowering the spatial dimensions of the input image. The encoding path extracts progressively abstract representations via convolutional layers, activation functions, and pooling layers. By integrating local features and global context, introducing skip links allows for preserving critical spatial information, facilitating reliable segmentation outcomes.

## Decoding Path in UNET
A critical component of the UNET architecture is the decoding path, also known as the expanding path. It is responsible for upsampling the encoding path’s feature maps and constructing the final segmentation mask.

## Upsampling Layers (Transposed Convolutions)
To boost the spatial resolution of the feature maps, the UNET decoding method includes upsampling layers, frequently done using transposed convolutions or deconvolutions. Transposed convolutions are essentially the opposite of regular convolutions. They enhance spatial dimensions rather than decrease them, allowing for upsampling. By constructing a sparse kernel and applying it to the input feature map, transposed convolutions learn to upsample the feature maps. The network learns to fill in the gaps between the current spatial locations during this process, thus boosting the resolution of the feature maps.

## Concatenation
The feature maps from the preceding layers are concatenated with the upsampled feature maps during the decoding phase. This concatenation enables the network to aggregate multi-scale information for correct segmentation, leveraging high-level context and low-level features. Aside from upsampling, the UNET decoding path includes skip connections from the encoding path’s comparable levels.

The network may recover and integrate fine-grained characteristics lost during encoding by concatenating feature maps from skip connections. It enables more precise object localization and delineation in the segmentation mask.

The decoding process in UNET reconstructs a dense segmentation map that fits with the spatial resolution of the input picture by progressively upsampling the feature maps and including skip links.

The decoding path’s function is to recover spatial information lost during the encoding path and refine the segmentation findings. It combines low-level encoding details with high-level context gained from the upsampling layers to provide an accurate and thorough segmentation mask.

UNET can boost the spatial resolution of the feature maps by using transposed convolutions in the decoding process, thereby upsampling them to match the original image size. Transposed convolutions assist the network in generating a dense and fine-grained segmentation mask by learning to fill in the gaps and expand the spatial dimensions.

In summary, the decoding process in UNET reconstructs the segmentation mask by enhancing the spatial resolution of the feature maps via upsampling layers and skip connections. Transposed convolutions are critical in this phase because they allow the network to upsample the feature maps and build a detailed segmentation mask that matches the original input image.

## Expanding and Contracting Paths 

In summary, UNET’s contracting and expanding routes resemble an “encoder-decoder” structure. The expanding path is the decoder, recovering spatial information and generating the final segmentation map. In contrast, the contracting path serves as the encoder, capturing context and compressing the input image. This architecture enables UNET to encode and decode information effectively, allowing for accurate and thorough image segmentation.

### Skip Connections in UNET
Skip connections are essential to the UNET design because they allow information to travel between the contracting (encoding) and expanding (decoding) paths. They are critical for maintaining spatial information and improving segmentation accuracy.

Preserving Spatial Information
Some spatial information may be lost during the encoding path as the feature maps undergo downsampling procedures such as max pooling. This information loss can lead to lower localization accuracy and a loss of fine-grained details in the segmentation mask.

By establishing direct connections between corresponding layers in the encoding and decoding processes, skip connections help to address this issue. Skip connections protect vital spatial information that would otherwise be lost during downsampling. These connections allow information from the encoding stream to avoid downsampling and be transmitted directly to the decoding path.

Multi-scale Information Fusion
Skip connections allow the merging of multi-scale information from many network layers. Later levels of the encoding process capture high-level context and semantic information, whereas earlier layers catch local details and fine-grained information. UNET may successfully combine local and global information by connecting these feature maps from the encoding path to the equivalent layers in the decoding path. This integration of multi-scale information improves segmentation accuracy overall. The network can use low-level data from the encoding path to refine segmentation findings in the decoding path, allowing for more precise localization and better object boundary delineation.

Combining High-Level Context and Low-Level Details
Skip connections allow the decoding path to combine high-level context and low-level details. The concatenated feature maps from the skip connections include the decoding path’s upsampled feature maps and the encoding path’s feature maps.

This combination enables the network to take advantage of the high-level context recorded in the decoding path and the fine-grained features captured in the encoding path. The network may incorporate information of several sizes, allowing for more precise and detailed segmentation.

UNET may take advantage of multi-scale information, preserve spatial details, and merge high-level context with low-level details by adding skip connections. As a result, segmentation accuracy improves, object localization improves, and fine-grained information in the segmentation mask is retained.

In conclusion, skip connections in UNETs are critical for maintaining spatial information, integrating multi-scale information, and boosting segmentation accuracy. They provide direct information flow across the encoding and decoding routes, allowing the network to collect local and global details, resulting in more precise and detailed image segmentation.

# Loss Function in UNET
It is critical to select an appropriate loss function while training UNET and optimizing its parameters for picture segmentation tasks. UNET frequently employs segmentation-friendly loss functions such as the Dice coefficient or cross-entropy loss.

## Dice Coefficient Loss
The Dice coefficient is a similarity statistic that calculates the overlap between the anticipated and true segmentation masks. The Dice coefficient loss, or soft Dice loss, is calculated by subtracting one from the Dice coefficient. When the anticipated and ground truth masks align well, the loss minimizes, resulting in a higher Dice coefficient.

The Dice coefficient loss is especially effective for unbalanced datasets in which the background class has many pixels. By penalizing false positives and false negatives, it promotes the network to divide both foreground and background regions accurately.

Dice Coeffiecient = 2 TP /(2TP+FN+FP)

## Cross-Entropy Loss
Use cross-entropy loss function in image segmentation tasks. It measures the dissimilarity between the predicted class probabilities and the ground truth labels. Treat each pixel as an independent classification problem in image segmentation, and the cross-entropy loss is computed pixel-wise.

The cross-entropy loss encourages the network to assign high probabilities to the correct class labels for each pixel. It penalizes deviations from the ground truth, promoting accurate segmentation results. This loss function is effective when the foreground and background classes are balanced or when multiple classes are involved in the segmentation task.

The choice between the Dice coefficient loss and cross-entropy loss depends on the segmentation task’s specific requirements and the dataset’s characteristics. Both loss functions have advantages and can be combined or customized based on specific needs.

## Comparative Study 
| Aspect              | UNet                           | Attention-UNet                                    | UNet++                                                     |
|---------------------|--------------------------------|---------------------------------------------------|------------------------------------------------------------|
| Architecture        | Follows a U-shaped architecture with encoder and decoder paths connected by skip connections. | Extends UNet with attention gates, incorporating self-attention mechanisms within the network. | Improves skip connections by introducing nested skip pathways, enhancing feature propagation. |
| Skip Connections    | Utilizes basic skip connections to concatenate feature maps from the encoder to the decoder. | Enhances skip connections by incorporating attention gates to adaptively weight feature maps. | Introduces nested skip connections, allowing for multiple levels of feature aggregation and refinement. |
| Attention Mechanism | Doesn't explicitly incorporate attention mechanisms for focusing on relevant features. | Integrates attention gates in the skip connections to emphasize informative regions and suppress noise. | Doesn't include attention mechanisms but focuses on nested skip connections for feature refinement. |
| Feature Propagation | Relies on simple concatenation of feature maps from encoder to decoder, lacking adaptive feature selection. | Improves feature propagation by selectively attending to relevant information using attention gates. | Enhances feature propagation through nested skip connections, facilitating iterative refinement of features. |
| Contextual Information | Limited capability to capture long-range dependencies or global context within the input data. | Enhances the model's understanding of global context by leveraging attention mechanisms for feature modulation. | Improves contextual information by utilizing multi-level skip connections, capturing information at different scales. |
| Performance         | Achieves competitive performance in various segmentation tasks but may struggle with capturing fine details. | Demonstrates improved segmentation accuracy and finer delineation of object boundaries, especially in complex scenes. | Offers enhanced segmentation results by effectively leveraging multi-level features and feature refinement. |

## Reasons for Using 3x3 Convolutions in UNet

1. **Local Receptive Field**: A 3x3 convolutional kernel has a local receptive field, allowing the network to capture local spatial dependencies effectively.

2. **Parameter Efficiency**: Using 3x3 convolutions reduces the number of parameters compared to larger kernel sizes while still capturing complex patterns efficiently.

3. **Hierarchical Feature Learning**: Stacking multiple layers of 3x3 convolutions enables the network to learn hierarchical features of increasing complexity.

4. **Translation Invariance**: Convolutional layers with 3x3 kernels provide translation invariance, crucial for recognizing patterns regardless of their position in the input image.

5. **Compatibility with Downsampling and Upsampling**: 3x3 convolutions are often used with pooling layers for downsampling and transposed convolutions for upsampling, preserving spatial information effectively.

Overall, the use of 3x3 convolutions in UNet balances capturing local dependencies, parameter efficiency, hierarchical feature learning, translation invariance, and compatibility with downsampling and upsampling operations.

