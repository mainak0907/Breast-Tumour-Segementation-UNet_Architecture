##  What is attention?
Attention, in the context of image segmentation, is a way to highlight only the relevant activations during training. This reduces the computational resources wasted on irrelevant activations, providing the network with better generalisation power. Essentially, the network can pay “attention” to certain parts of the image.

### a. Hard Attention

Attention comes in two forms, hard and soft. Hard attention works on the basis of highlighting relevant regions by cropping the image or iterative region proposal. Since hard attention can only choose one region of an image at a time, it has two implications, it is non-differentiable and requires reinforcement learning to train.

Since it is non-differentiable, it means that for a given region in an image, the network can either pay “attention” or not, with no in-between. As a result, standard backpropagation cannot be done, and Monte Carlo sampling is needed to calculate the accuracy across various stages of backpropagation. Considering the accuracy is subject to how well the sampling is done, there is a need for other techniques such as reinforcement learning to make the model effective.

### b. Soft Attention

Soft attention works by weighting different parts of the image. Areas of high relevance is multiplied with a larger weight and areas of low relevance is tagged with smaller weights. As the model is trained, more focus is given to the regions with higher weights. Unlike hard attention, these weights can be applied to many patches in the image.

Due to the deterministic nature of soft attention, it remains differentiable and can be trained with standard backpropagation. As the model is trained, the weighting is also trained such that the model gets better at deciding which parts to pay attention to.

## Why is Attetion needed in UNet ?
During upsampling in the expanding path, spatial information recreated is imprecise. To counteract this problem, the U-Net uses skip connections that combine spatial information from the downsampling path with the upsampling path. However, this brings across many redundant low-level feature extractions, as feature representation is poor in the initial layers.

Soft attention implemented at the skip connections will actively suppress activations in irrelevant regions, reducing the number of redundant features brought across.

<img width="542" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/487c7935-4bc5-49a4-b683-3ee85d50bf74">

