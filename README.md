# CNNMnist
Use the convolutional neural network for training deep learning models to identify the hand-written numbers in the Mnist dataset. Coding from scratch, only use numpy and some basic libraries.

## Formula behind the code:
The formula for forwarding and backwarding can be found easily on the internet, the most easy to understand for me is:
  - https://mukulrathi.com/demystifying-deep-learning
  - https://cs231n.github.io/convolutional-networks/
  - ....

## Instruction
I designed a class called MnistTraining for training models to identify numbers in the Mnist dataset. The dataset is loaded from Keras and is shuffled after each epoch. Users can design the model as follows:
-  First, to create a model, just call: `model_name = MnistTraining()`
-  Then, you can load data and set the number of samples of training and validation set with `model_name.loadDataFromKeras(n,m)`
-  You can set batch-size and learning rate by `model_name.setBacthSize(n)` and `model_name.setLearningRate(r)`, default values is 100 and 0.01
-  To design the model, you can use `model_name.add()`, in this method, you have to pass a layer with the specific type and other needed arguments, there are 5 types of layers here:
    - Convolutional layer: `model_name.add(layer("conv", filter_shape, input_shape, activation, padding))`; filter_shape is a 4D tensor with shape (H, W, D, N) - N is the number of filters; input_shape is a 3D tensor with shape (H, W, D); activation allowed is "relu", "tanh", "sigmoid", and "softmax" for last layer; padding is "same" or "valid". The stride for this layer is set to 1 and you can not change it.
    - Max Pooling layer: `model_name.add(layer("maxPool"))`, you can just use a 2x2 filter and stride 2 for the pooling layer.
    - Average Pooling layer: `model_name.add(layer("avgPool"))`
    - Flatten layer: `model_name.add(layer("flatten", output_shape, input_shape))`, you have to pass both input_shape and output_shape here =)). The input_shape is a 4D tensor: (M, H, W, D) - M is the number of samples. The output_shape should be (H *W *D, M), you have to calculate it though it seems kinda unnecessary, but it's still ok right?
    - Dense layer: `model_name.add(layer("dense",n_node,n_node_prv,activation))`, n_node is the number of nodes in this layer, n_node_prv is number of node in previous layer.
-  The last layer should have 10 nodes for 10 numbers and use softmax.
-  Finally, you can use `model_name.trainingMnist(number_of_epochs)` to train your model and `model_name.testing()` to test your model with validation test loaded before.

## Few drawbacks:
-  The source code still has some bugs, because I haven't handled exploding numbers yet. I will update it when I stop being lazy.
-  Function for forwarding and backwarding of the convolutional layer is better than before but it's not ultimately optimized, with 60k samples, each epoch will take about 1 min.
