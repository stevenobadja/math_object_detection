# Math Operators Object Detection (Video Demo)

### Test 1 - Simple Math
[![Custom Math Operators TensorFlow Object Detection - Test 1](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Screen%20Shot%202018-01-04%20at%205.58.45%20PM.png?raw=true)](https://youtu.be/iss52uQS6jo)

### Test 2 - Linear Algebra
[![Linear Algebra TensorFlow Object Detection - Test 2](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Screen%20Shot%202018-01-07%20at%201.35.57%20PM.png?raw=true)](https://youtu.be/uqbdKshCXgQ)

Utilizing TensorFlow **Object Detection** API open source framework makes it feasible to construct, train and deploy a custom object detection model with ease. The detection model shown above uses TensorFlow's API and detects **handwritten digits** and **simple math operators**. In addition, the output of the predicted objects (numbers & math operators) are then evaluated and solved. Currently, the model created above is limited to **basic math** and **linear algebra**.

# Model
### Powered by: Tensorflow
**Model:** ssd_mobilenet_v1_coco_2017_11_17<br/>
**Config:** ssd_mobilenet_v1_coco<br/>

### SSD Mobilenet Architecture
[![SSD Mobilenet Architecture](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/SSD%20Mobilenet%20Architecture.png?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/SSD%20Mobilenet%20Architecture.png?raw=true)

- **Convolution Neural network** - The convolutional layer is the core building block of a CNN. The layer's parameters consist of a set of learnable filters (or kernels), which have a small receptive field, but extend through the full depth of the input volume. During the forward pass, each filter is convolved across the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional activation map of that filter. As a result, the network learns filters that activate when it detects some specific type of feature at some spatial position in the input. (**Source:** [Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network))<br/><br/>
- **Batch Normalization** - Batch normalization potentially helps in two ways: faster learning and higher overall accuracy. The improved method also allows you to use a higher learning rate, potentially providing another boost in speed. Normalization (shifting inputs to zero-mean and unit variance) is often used as a pre-processing step to make the data comparable across features. As the data flows through a deep network, the weights and parameters adjust those values, sometimes making the data too big or too small again - "internal covariate shift". By normalizing the data in each mini-batch, this problem is largely avoided. (**Source:** [Derek Chan ~ Quora](https://www.quora.com/Why-does-batch-normalization-help))<br/><br/>
- **Rectified linear unit or (ReLU)** - ReLu is an activation function. In biologically inspired neural networks, the activation function is usually an abstraction representing the rate of action potential firing in the cell. In its simplest form, this function is binary—that is, either the neuron is firing or not. ReLU is half rectified from the bottom. It is f(s) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero. With a range of 0 to infinity. (**Source:** [Towards Data Science](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6))<br/>

[![Batch Normalization](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Batch%20Normalization.png?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Batch%20Normalization.png?raw=true)

[![ReLu](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/ReLu.png?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/ReLu.png?raw=true)

# Summary of training this model

#### Step 1:
**Create an image library** - The pre-existing [ssd_mobilenet_v1_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) model was trained with a custom, created from scratch, [image library](https://github.com/stevenobadja/math_object_detection/tree/master/images) (of math numbers & operators). This image library can be substituted with any object or objects of choice. Due to the constraint of time, the model above was trained on a total of 345 images of which 10% was allocated for test validation.

[![train image](https://github.com/stevenobadja/math_object_detection/blob/master/images/testadd2.3.jpg?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/images/testadd2.3.jpg?raw=true)

#### Step 2:
**Box & label each class** - In order to train and test the model, TensorFlow requires that a box is drawn for each class. To be more specific, it needs the X and Y axis (ymin, xmin, ymax, xmax) of the box in relation to the image. These coordinates is then respectively divided by the lenght or width of the image and is stored as a float. An example of the process is shown below. (Note: the current model contains 23 classes) Thanks to tzutalin [tzutalin, labelImg](https://github.com/tzutalin/labelImg), with the creation of GUI that makes this process easy.

[![Box Process](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Screen%20Shot%202018-01-04%20at%2011.12.01%20PM.png?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Screen%20Shot%202018-01-04%20at%2011.12.01%20PM.png?raw=true)

#### Step 3:
**Convert files** - Once the labeling process is complete the folder will be full with XML files, however this cannot be used yet by TensorFlow for training and testing. Instead the XML files needs to be converted into a CSV file. Then the CSV file will then be converted to tfrecords file for training.

#### Step 4:
**Create pbtxt** - Create a pbtxt file by creating ID's and Name (labels) for each class. This file will be used with the finished model as an category_index.

#### Step 5:
**Train the model** - (See model above)<br/>
Summary: input layer --> weights --> batch normalization --> hidden layer 1 (activation function: ReLu) --> weights batch normalization --> hidden layer 2 (activation function: ReLu) --> output layer.

After the output layer, it compares the output to the intended output --> cost function (weighted_sigmoid) --> optimization function (optimizer) --> minimize cost (rms_prop_optimizer, learning rate = 0.004)

1 cycle of summary above = 1 Global Step

[![Global Step](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Screen%20Shot%202018-01-04%20at%2011.26.40%20PM.png?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Screen%20Shot%202018-01-04%20at%2011.26.40%20PM.png?raw=true)

This process requires heavy computing power, due to the constraints of hardware (CPU only), it took approximately 4 days & 7 hours to complete 50k Global Step.

[![Duration](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Loss%20Relative%20at%2050k.png?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Loss%20Relative%20at%2050k.png?raw=true)

[![Graph](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Loss%20Chart%20at%2050k.png?raw=true)](https://github.com/stevenobadja/math_object_detection/blob/master/s_img/Loss%20Chart%20at%2050k.png?raw=true)

#### Step 6:
**Export inference graph** - Once a model is trained with an acceptable loss rate. It is stopped by the user manually. As the model is being trained it is creates a checkpoint file after each set milestone. This checkpoint file is then converted into an inference graph which is used for deployment/serving.

# Source & Support Files

**Google's object detection** [(Link)](https://github.com/tensorflow/models/tree/master/research/object_detection)

Will contain:
- train.py
- export_inference_graph.py
- All other support docs...

**Tensorflow detection model zoo** [(Link)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Will contain:
- ssd_mobilenet_v1_coco_2017_11_17 (model used for this demo)
- All other models released by Tensorflow...

**Tensorflow config files** [(Link)](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

Will contain:
- ssd_mobilenet_v1_coco (config used for this demo)
- All other configs released by Tensorflow...

**Racoon's object detection** [(Link)](https://github.com/datitran/raccoon_dataset)

Will contain:
- xml_to_csv.py
- generate_tfrecord.py

**Label images with labelImg** [(Link)](https://github.com/tzutalin/labelImg)

Will contain:
- labelImg.py

# Special Thanks!
To Harrison Kinsley [(Sentdex)](http://sentdex.com/)
