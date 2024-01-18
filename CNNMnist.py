import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hiding INFO and WARNING debugging information

from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import timeit

def conv_forward(x,w,b,padding="same"):
    f = w.shape[0] #size of filter

    if padding=="same":
        pad = (f-1)//2
    else: #padding is valid - i.e no zero padding
        pad =0
    n = (x.shape[1]-f+2*pad) +1 #ouput width/height

    y = np.zeros((x.shape[0],n,n,w.shape[3])) #output array

    #pad input
    x_padded = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values = 0)

    #flip filter to cancel out reflection
    w = np.flip(w,0)
    w = np.flip(w,1)


    for m_i in range(x.shape[0]): #each of the training examples
        for k in range(w.shape[3]): #each of the filters
            for d in range(x.shape[3]): #each slice of the input
                y[m_i,:,:,k]+= ndimage.convolve(x_padded[m_i,:,:,d],w[:,:,d,k])[f//2:-(f//2),f//2:-(f//2)] #sum across depth

    y = y + b #add bias (this broadcasts across image)
    return y

def pool_forward(x,mode="max"):
    if(x.shape[1]%2==1): x = x[:,1:x.shape[1],1:x.shape[2],:]
    x_patches = x.reshape(x.shape[0],x.shape[1]//2, 2,x.shape[2]//2, 2,x.shape[3])

    if mode=="max":
        out = x_patches.max(axis=(2,4))
        mask  =np.isclose(x,np.repeat(np.repeat(out,2,axis=1),2,axis=2)).astype(int)
    elif mode=="average":
        out =  x_patches.mean(axis=(2,4))
        mask = np.ones_like(x)*0.25
    return mask, out

def pool_backward(dx, mask):
    return mask*(np.repeat(np.repeat(dx,2,axis=1),2,axis=2))

def conv_backward(dZ,x,w,padding="same"):
    m = x.shape[0]

    db = (1/m)*np.sum(dZ, axis=(0,1,2), keepdims=True)

    if padding=="same":
        pad = (w.shape[0]-1)//2
    else: #padding is valid - i.e no zero padding
        pad =0
    x_padded = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values = 0)

    #this will allow us to broadcast operations
    x_padded_bcast = np.expand_dims(x_padded, axis=-1) # shape = (m, i, j, c, 1)
    dZ_bcast = np.expand_dims(dZ, axis=-2) # shape = (m, i, j, 1, k)

    dW = np.zeros_like(w)
    f=w.shape[0]
    w_x = x_padded.shape[1]
    
    for a in range(f):
        for b in range(f):
            #note f-1 - a rather than f-a since indexed from 0...f-1 not 1...f
            dW[a,b,:,:] = (1/m)*np.sum(dZ_bcast*
                x_padded_bcast[:,a:w_x-(f-1 -a),b:w_x-(f-1 -b),:,:],
                axis=(0,1,2))
            

    dx = np.zeros_like(x_padded,dtype=float)
    Z_pad = f-1
    dZ_padded = np.pad(dZ,((0,0),(Z_pad,Z_pad),(Z_pad,Z_pad),
    (0,0)),'constant', constant_values = 0)

    for m_i in range(x.shape[0]):
        for k in range(w.shape[3]):
            for d in range(x.shape[3]):
                dx[m_i,:,:,d]+=ndimage.convolve(dZ_padded[m_i,:,:,k],
                w[:,:,d,k])[f//2:-(f//2),f//2:-(f//2)]
    dx = dx[:,pad:dx.shape[1]-pad,pad:dx.shape[2]-pad,:]
    return dx,dW,db

class layer:    
    #This class will hold value of w and b, input size and size of w, activation type and padding depend on layer type
    def __init__(self, type, size = (2,2), input_size = (2,2), activation = "relu", padding = "same"):
        self.type = type
        self.input_size = input_size
        if(type == "maxPool"): return
        if(type == "avgPool"): return
        if(type == "flatten"): return
        if(type == "dense"):
            if(activation!="relu" and activation!="tanh" and activation!="sigmoid" and activation!= "softmax"):
                raise Exception("Sorry, activation functions allowed is: relu, tanh, sigmoid, softmax")
            self.activation = activation
            self.size = size
            self.W = np.random.rand(size, input_size) - 0.5
            self.b = np.random.rand(size,1) - 0.5
            return
        if(type == "conv"):
            if(activation!="relu" and activation!="tanh" and activation!="sigmoid"):
                raise Exception("Sorry, activation functions allowed in 'conv' is: relu, tanh, sigmoid")
            if(padding != "same" and padding != "valid"): raise Exception("Sorry, padding types allowed is: valid or same")
            self.padding = padding
            self.activation = activation
            self.size = size
            self.W = np.ones(size) / 100
            self.b = np.ones((size[3])) / 100

class MnistTraining:
    X_alltrain = []
    Y_alltrain = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    loss = [] 
    acu = []
    loss_t = []
    acu_t = []
    nLayer = 0
    rate = 0.01
    batchSize = 100
    layers = []
        
    def loadDataFromKeras(self, n_train, n_test):
        # Load Mnist data from keras, the training set will have shape: (n_train, 28, 28, 1), similar with validation set
        (self.X_alltrain, self.Y_alltrain), (self.X_test, self.Y_test) = keras.datasets.mnist.load_data()

        self.X_alltrain = np.expand_dims(self.X_alltrain, -1)
        self.X_test = np.expand_dims(self.X_test, -1)
        
        self.X_alltrain = self.X_alltrain.astype("float32") / 255
        self.X_test = self.X_test.astype("float32") / 255
        self.Y_alltrain = self.Y_alltrain.astype("float32")
        self.Y_test = self.Y_test.astype("float32")
        
        if(n_train == 60000 and n_test == 10000): return
        
        self.X_alltrain = self.X_alltrain[0:n_train,:,:,:]
        self.Y_alltrain = self.Y_alltrain[0:n_train]
        self.X_test = self.X_test[0:n_test,:,:,:]
        self.Y_test = self.Y_test[0:n_test]
        
    def shullfeData(self):
        # Shuffle data for each epoch
        per =  np.random.permutation(self.X_alltrain.shape[0])
        self.X_alltrain = self.X_alltrain[per,:,:,:]
        self.Y_alltrain = self.Y_alltrain[per]
        self.iter = 0
        
    def loadBacth(self,iter):
        self.X_train = self.X_alltrain[iter*self.batchSize: (iter+1)*self.batchSize, :, :, :]
        self.Y_train = self.Y_alltrain[iter*self.batchSize: (iter+1)*self.batchSize]
        
    def setBacthSize(self, a):
        self.batchSize = a
        
    def add(self,layer):
        self.layers.append(layer)
        self.nLayer += 1
            
    def ActivationFunc(self, Z, type):
        if(type=="relu"):
            return np.maximum(0,Z)
        elif(type=="tanh"):
            return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        elif(type=="sigmoid"):
            return 1/(1+np.exp(-Z))
        elif(type == "softmax"):
            exps=np.exp(Z-Z.max())
            return exps / exps.sum(axis=0) + 1e-7
        
    def setLearningRate(self, rate):
        self.rate = rate
        
    def DeriAcFunc(self, Z, type):   
        # derivative of activation function
        if(type=="relu"):
            return Z > 0
        elif(type=="tanh"):
            return 1 - np.square(self.ActivationFunc(Z))
        elif(type=="sigmoid"):
            exps=np.exp(Z-Z.max())
            return exps / (exps.sum(axis=0)+1e-8) + 1e-8
    
    def feedForward(self, X_train):
        A = []
        A.append((X_train,X_train))
        for iL in range (0, self.nLayer):
            layer = self.layers[iL]
            if(layer.type == "maxPool"):
                A.append(pool_forward(A[-1][1],"max"))
            elif(layer.type == "avgPool"):
                A.append(pool_forward(A[-1][1],"average"))
            elif(layer.type == "conv"):
                Z = conv_forward(A[-1][1],layer.W, layer.b, layer.padding)
                A.append( (Z, self.ActivationFunc(Z,layer.activation)) )
            elif(layer.type == "dense"):
                Z = np.dot(layer.W, A[-1][1]) + layer.b
                A.append( (Z, self.ActivationFunc(Z, layer.activation) ))
            elif(layer.type == "flatten"):
                A.append( (A[-1][1], A[-1][1].reshape( A[-1][1].shape[0], A[-1][1].shape[1] * A[-1][1].shape[2] * A[-1][1].shape[3] ).T) )
                
        return A
    
    def checkRes(self, Y):
        L = len(Y)
        YY = np.zeros((10,L))
        for j in range (0,L):
            for i in range (0,10):
                if( i == Y[j] ):
                    YY[i][j]=1
        return YY
    
    def cost(self,Y, A2):
        L = len(Y)
    
        tt=0
        for j in range (0, L):
            a=0
            for i in range(0,10):
                a=a+Y[i][j]*np.log(A2[i][j])
            tt+=a
        
        return -tt/L
    
    def predict(self,A2):
        return np.argmax(A2,0)
    
    def accuracy(self,predict, Y):
        return np.sum(predict == Y) / Y.size
    
    def feedBackward(self, A):
        dW = 0
        db = 0
        dZ = 0
        dA = 0
        for iL in range(self.nLayer,0,-1):
            layer = self.layers[iL-1]
            
            if(layer.type == "dense" and layer.activation == "softmax"):
                Ycheck = self.checkRes(self.Y_train)
                m = self.batchSize
                dZ = A[iL][1] - Ycheck
                dW = 1 / m * np.dot(dZ, A[iL-1][1].T)
                db = 1 / m * np.sum(dZ)
                dA = np.dot( layer.W.T, dZ )
                
            elif(layer.type == "dense"):
                m = self.batchSize
                dZ = dA * self.DeriAcFunc(A[iL][0], layer.activation)
                dW = 1 / m * np.dot(dZ, A[iL-1][1].T)
                db = 1 / m * np.sum(dZ)
                dA = np.dot( layer.W.T, dZ )
            
            elif(layer.type == "flatten"):
                dZ = dA.T.reshape( dA.shape[1], layer.input_size[0], layer.input_size[1], layer.input_size[2])
                dA = dZ
                continue
            
            elif(layer.type == "avgPool" or layer.type == "maxPool" ):
                dZ = pool_backward(dA, A[iL][0])
                dA = dZ
                continue
            
            elif(layer.type == "conv"):
                dZ = dA * self.DeriAcFunc(A[iL][0], layer.activation)
                dA, dW, db = conv_backward(dZ, A[iL-1][1], self.layers[iL-1].W, layer.padding)
                
            else: continue
            
            Lrate = 0.006
            if( layer.type == "dense" ): Lrate = self.rate * 10
            else: Lrate = self.rate
             
            self.layers[iL-1].W = self.layers[iL-1].W - Lrate * dW
            self.layers[iL-1].b = self.layers[iL-1].b - Lrate * db
     
    def training(self):
        A = self.feedForward(self.X_train)
        self.feedBackward(A)
        
        return self.accuracy( self.predict(A[-1][1]), self.Y_train ), A[-1][1]
                
    def trainingMnist(self, epoch):
        for i in range (0, epoch):
            s = 0
            cs = 0
            self.shullfeData()
            n_iter = round(self.Y_alltrain.size / self.batchSize)
            for j in range (0, n_iter):
                self.loadBacth(j)
                m, A = self.training()
                s += m
                cs += self.cost(self.checkRes(self.Y_train),A)
                
            A_test, Acu_test = self.testing(0)
            Cost_test = self.cost(self.checkRes(self.Y_test), A_test)
            
            print("[Epoch]", i, ", Train accuracy =", s/n_iter, ", Validation accuracy =", Acu_test)
            self.loss.append(cs/n_iter)
            self.acu.append(s/n_iter)
            self.loss_t.append(Cost_test)
            self.acu_t.append(Acu_test)
            
    def testing(self, showInfo):
        A = self.feedForward(self.X_test)
        Pred = self.predict(A[-1][1])
        Acurracy = self.accuracy( Pred , self.Y_test )
        if(showInfo):
            print("Accuracy: ", Acurracy )
            print("Tests failed: ", np.nonzero(Pred - self.Y_test)[0] )
            
        return A[-1][1], Acurracy

    def lossGraph(self):
        plt.plot(self.loss, label = "Training loss" )
        plt.plot(self.loss_t, label = "Validation loss")
        plt.title("Loss graph")
        plt.xlabel("Loss")
        plt.ylabel("Epoch")
        plt.legend()
        plt.show()
        
    def accuracyGraph(self):
        plt.plot(self.acu, label = "Training accuracy" )
        plt.plot(self.acu_t, label = "Validation accuracy" )
        plt.title("Accuracy graph")
        plt.xlabel("Accuracy")
        plt.ylabel("Epoch")
        plt.legend()
        plt.show()

    def visualTest(self, n):
        A = self.feedForward(np.expand_dims(self.X_test[n,:,:,:], axis = 0 ))
        print("Test", n)
        print("Predict:", self.predict(A[-1][1])[0])
        print("Label:", self.Y_test[n])
        plt.imshow(self.X_test[n,:,:,0], cmap='gray' )
        plt.show()



