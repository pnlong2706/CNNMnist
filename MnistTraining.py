import timeit
from matplotlib import pyplot as plt

from CNNMnist import MnistTraining
from CNNMnist import layer

#MAIN FUNCTION
model = MnistTraining()

model.loadDataFromKeras(30000,5000)
model.setBacthSize(100)
model.setLearningRate(0.01)

# model.add(layer("conv",(3,3,1,8),(28,28,1),"relu","same"))
# model.add(layer("avgPool"))
# model.add(layer("conv",(3,3,8,16),(14,14,8),"relu","same"))
# model.add(layer("avgPool"))
# model.add(layer("flatten", 784, (7,7,16)))
# model.add(layer("dense",128,784,"relu"))
# model.add(layer("dense",10,128,"softmax"))

# model.add(layer("conv",(3,3,1,2),(28,28,1),"relu","same"))
# model.add(layer("conv",(3,3,2,4),(28,28,2),"relu","same"))
# model.add(layer("flatten", 3136, (28,28,4)))
# model.add(layer("dense",128,3136,"relu"))
# model.add(layer("dense",10,128,"softmax"))

model.add(layer("conv",(3,3,1,8),(28,28,1),"relu","same"))
model.add(layer("avgPool"))
model.add(layer("flatten", 1568, (14,14,8)))
model.add(layer("dense",128,1568,"relu"))
model.add(layer("dense",10,128,"softmax"))

# model.add(layer("flatten", 784, (28,28,1)))
# model.add(layer("dense",256,784,"relu"))
# model.add(layer("dense",128,256,"relu"))
# model.add(layer("dense",10,128,"softmax"))

# model.add(layer("flatten", 784, (28,28,1)))
# model.add(layer("dense",128,784,"relu"))
# model.add(layer("dense",10,128,"softmax"))

def train(n):
    start = timeit.default_timer()
    model.trainingMnist(n)
    stop = timeit.default_timer()
    print('Time training: ', stop - start)

def test():
    print("TEST SET:")
    model.testing(1)

def showLossGraph():
    model.lossGraph()
    
def showAccuracyGraph():
    model.accuracyGraph()

def visual(n):
    model.visualTest(n)
