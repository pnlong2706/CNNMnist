import timeit
from matplotlib import pyplot as plt

from CNNMnist import MnistTraining
from CNNMnist import layer

#MAIN FUNCTION
TrainModel = MnistTraining()

TrainModel.loadDataFromKeras(1000,100)
TrainModel.setBacthSize(100)
TrainModel.setLearningRate(0.006)

# TrainModel.add(layer("conv",(3,3,1,8),(28,28,1),"relu","same"))
# TrainModel.add(layer("avgPool"))
# TrainModel.add(layer("conv",(3,3,8,16),(14,14,8),"relu","same"))
# TrainModel.add(layer("avgPool"))
# TrainModel.add(layer("flatten", 784, (7,7,16)))
# TrainModel.add(layer("dense",128,784,"relu"))
# TrainModel.add(layer("dense",10,128,"softmax"))

# TrainModel.add(layer("conv",(3,3,1,2),(28,28,1),"relu","same"))
# TrainModel.add(layer("conv",(3,3,2,4),(28,28,2),"relu","same"))
# TrainModel.add(layer("flatten", 3136, (28,28,4)))
# TrainModel.add(layer("dense",128,3136,"relu"))
# TrainModel.add(layer("dense",10,128,"softmax"))

# TrainModel.add(layer("conv",(3,3,1,8),(28,28,1),"relu","same"))
# TrainModel.add(layer("avgPool"))
# TrainModel.add(layer("flatten", 1568, (14,14,8)))
# TrainModel.add(layer("dense",128,1568,"relu"))
# TrainModel.add(layer("dense",10,128,"softmax"))

# TrainModel.add(layer("flatten", 784, (28,28,1)))
# TrainModel.add(layer("dense",256,784,"relu"))
# TrainModel.add(layer("dense",128,256,"relu"))
# TrainModel.add(layer("dense",10,128,"softmax"))

TrainModel.add(layer("flatten", 784, (28,28,1)))
TrainModel.add(layer("dense",128,784,"relu"))
TrainModel.add(layer("dense",10,128,"softmax"))

def train(n):
    start = timeit.default_timer()
    TrainModel.trainingMnist(n)
    stop = timeit.default_timer()
    print('Time training: ', stop - start)

def test():
    print("TEST SET:")
    TrainModel.testing()

def showGraph():
    plt.plot(TrainModel.costX, TrainModel.costY)
    plt.show()

def visual(n):
    TrainModel.visualTest(n)
