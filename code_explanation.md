## opendataset library 
The opendatasets library is a Python library designed to simplify the process of downloading and working with various open datasets. It provides a convenient interface for accessing datasets from sources like Kaggle, GitHub, and other online repositories

## Dataloader Function 
1. **`framObjTrain = {'img' : [], 'mask' : []}`**: This line initializes a Python dictionary named `framObjTrain` with two keys: `'img'` and `'mask'`. The corresponding values for these keys are empty lists. This dictionary is intended to store image and mask data.

2. **`def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 256):`**: This line defines a Python function named `LoadData` with four parameters:
   - `frameObj`: A dictionary to store image and mask data. It's initialized as `None` by default.
   - `imgPath`: The path to the directory containing the image files.
   - `maskPath`: The path to the directory containing the mask files.
   - `shape`: The desired shape (width and height) for resizing the images and masks. It's set to 256 pixels by default.

3. **`imgNames = os.listdir(imgPath)`**: This line uses the `os.listdir()` function to retrieve a list of filenames in the directory specified by `imgPath`. These filenames represent the image files.

4. **`names = []`**, **`maskNames = []`**, **`unames = []`**: These lines initialize empty lists to store filenames and unique identifiers extracted from the filenames.

5. **`for i in range(len(imgNames)): unames.append(imgNames[i].split(')')[0])`**: This loop iterates over the list of image filenames (`imgNames`). For each filename, it splits the filename based on the ')' character and retrieves the substring before ')' (i.e., the unique identifier). It then appends this identifier to the `unames` list.

6. **`unames = list(set(unames))`**: This line converts the `unames` list into a set to remove duplicates, then converts it back into a list. This ensures that each unique identifier appears only once in the list.

7. **`for i in range(len(unames)): names.append(unames[i]+').png')`** and **`maskNames.append(unames[i]+')_mask.png')`**: These loops iterate over the unique identifiers (`unames`) and construct filenames for both images and masks based on these identifiers. They append these filenames to the `names` and `maskNames` lists respectively.

8. **`imgAddr = imgPath + '/'`** and **`maskAddr = maskPath + '/'`**: These lines construct the full paths to the image and mask directories by concatenating `imgPath` and `maskPath` with a forward slash ('/').

9. **`for i in range(len(names)): img = plt.imread(imgAddr + names[i]) mask = plt.imread(maskAddr + maskNames[i])`**: This loop iterates over the constructed image filenames (`names`) and mask filenames (`maskNames`). For each iteration, it reads the corresponding image and mask files using `plt.imread()` function from matplotlib, and stores them in the variables `img` and `mask` respectively.

10. **`img = cv2.resize(img, (shape, shape))`** and **`mask = cv2.resize(mask, (shape, shape))`**: These lines resize the image and mask to the desired shape (`shape`), using the `cv2.resize()` function from OpenCV.

11. **`frameObj['img'].append(img)`** and **`frameObj['mask'].append(mask)`**: These lines append the resized image and mask to the respective lists (`'img'` and `'mask'`) inside the `frameObj` dictionary.

12. **`return frameObj`**: Finally, this line returns the `frameObj` dictionary containing the image and mask data after processing all the files in the directories specified by `imgPath` and `maskPath`.

Resizing of image helps in standardization , computational efficiency , prevents overfitting

## Conv2DBlock

```
def Conv2dBlock(inputTensor, numFilters, kernelSize=3, doBatchNorm=True):
    # First Convolutional Layer
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)
```

This line creates a 2D convolutional layer using tf.keras.layers.Conv2D. It specifies the number of filters (numFilters), the kernel size (kernelSize), the weight initialization method ('he_normal'), and padding to maintain the spatial dimensions of the input ('same'). The input to this layer is the inputTensor.

<img width="487" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/9785e476-de4d-424c-8f04-62c971d20447">

<img width="471" alt="image" src="https://github.com/mainak0907/Breast-Tumour-Segementation-UNet_Architecture/assets/88925745/498a102c-9362-4851-8b0d-12728a3b4bbc">

