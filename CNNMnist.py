import numpy as np
import csv
from matplotlib import pyplot as plt

def convolve(mat, fil, st):
    H = mat.shape[0]
    W = mat.shape[1]
    D = mat.shape[2]
    B = mat.shape[3]

    sz = fil.shape[0]
    nF = fil.shape[3]
    
    nH = (H - sz)//st + 1
    nW = (W - sz)//st + 1
    
    ans = np.zeros((nH,nW,nF,B))
    
    for ib in range(0, B):
        for ft in range(0, nF):
            for i in range(0, nH):
                for j in range(0, nW):
                    ans[i][j][ft][ib] = np.sum(mat[ (i*st):(i*st + sz), (j*st):(j*st + sz),:,ib] * fil[:,:,:,ft])
                    
    return ans

def AvrgPooling(mat, fil, st):
    H = mat.shape[0]
    W = mat.shape[1]
    D = mat.shape[2]
    B = mat.shape[3]

    sz = fil.shape[0]
    
    nH = (H - sz)//st + 1
    nW = (W - sz)//st + 1
    
    ans = np.zeros((nH,nW,D,B))
    
    for ib in range(0, B):
        for i in range(0, nH):
            for j in range(0, nW):
                temp = mat[ (i*st):(i*st + sz), (j*st):(j*st + sz),:,ib] * fil[:,:,:]
                ans[i,j,:,ib] = np.sum(temp, (0,1))
                    
    return ans

def backPooling(size, dP, st):
    H = dP.shape[0]
    W = dP.shape[1]
    
    #print(size)
    
    ans = np.zeros(size)
    
    for ib in range (0, size[3]):
        for d in range (0, size[2]):
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    if(i//st < H and j//st < W): ans[i][j][d][ib] = 0.25 * dP[i//st][j//st][d][ib]
                    
    return ans

def dwConvolve( size, mat, dZc ):
    sz = dZc.shape[0]
    
    ans = np.zeros(size)
    
    for ib in range(0, mat.shape[3]):
        for nf in range(0, size[3]):
            for i in range(0, size[0]):
                for j in range(0, size[1]):

                    tt = dZc[:,:,nf,ib].reshape(sz,sz,1)
                    temp = mat[ i:i+sz, j:j+sz, :, ib ] * tt
                    
                    ans[i,j,:,nf] +=  np.sum(temp, (0,1))
    
    return ans 

class MnistTraing:
    X_alltrain = []
    Y_alltrain = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    costX = []
    costY = []
    nLayer = 0
    nNode = []
    arW = []
    arb = []
    caches = []
    rate = 0.1
    batchSize = 100
    iter = 0
    Activation = ""
    
    Zc1 = []
    Ac1 = []
    Ac2 = []
    Ac3 = []
    Xc = []
    
    Conv1Filter = np.zeros(1)
    Conv1Pool = np.zeros((2,2,8)) + 0.25
    
    def loadDataFromFile(self,s_train,M,s_test,L):
        CX=[]
        CY=[]
        with open(s_train) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line=0
            for row in csv_reader:
                if line==0:
                    line +=1
                else:
                    CY.append( eval(row[0]) )
                    CX.append( [ eval(i)/255 for i in row[1:785] ] )
                    line+=1
            
                if line==M+1:
                    break
        
        self.X_alltrain=(np.array(CX)).T
        self.Y_alltrain=np.array(CY)
        
        CX=[]
        CY=[]
        with open(s_test) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line=0
            for row in csv_reader:
                if line==0:
                    line +=1
                else:
                    CY.append( eval(row[0]) )
                    CX.append( [ eval(i)/255 for i in row[1:785] ] )
                    line+=1
            
                if line==L+1:
                    break
                
        self.X_test=(np.array(CX)).T
        self.X_test = self.X_test.reshape((28,28,1,L))
        self.Y_test=np.array(CY)
        
    def shullfeData(self):
        per =  np.random.permutation(self.X_alltrain.shape[1])
        self.X_alltrain = self.X_alltrain[:, per]
        self.Y_alltrain = self.Y_alltrain[per]
        self.iter = 0
        
    def loadBacth(self,iter):
        self.X_train = self.X_alltrain[:,iter*self.batchSize: (iter+1)*self.batchSize ]
        self.Y_train = self.Y_alltrain[iter*self.batchSize: (iter+1)*self.batchSize ]
        self.X_train = self.X_train.reshape((28,28,1,self.batchSize))
        #print(self.X_train.shape)
        
    def setBacthSize(self, a):
        self.batchSize = a
    
    def setLayer(self,x, A ):
        self.nLayer = x
        for i in A:
            self.nNode.append(i)
            
    def setActivation(self,s):
        self.Activation = s
        if(s!="relu" and s!="tanh" and s!="sigmoid" ):
            raise Exception("Sorry, activation functions allowed is: relu, tanh, sigmoid")
        
    def setLearningRate(self,a):
        self.rate = a
        
    def randomPara(self):
        for i in range (0,self.nLayer-1) :
            temp_W = np.random.rand( self.nNode[i+1] ,self.nNode[i])-0.5
            self.arW.append( temp_W )
            temp_b = np.random.rand(self.nNode[i+1],1)-0.5
            self.arb.append( temp_b )
            
        self.Conv1Filter = np.random.randn(3,3,1,8)
            
    def ActivationFunc(self, Z):
        if(self.Activation=="relu"):
            return np.maximum(0,Z)
        elif(self.Activation=="tanh"):
            return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        elif(self.Activation=="sigmoid"):
            return 1/(1+np.exp(-Z))
        
    def DeriAcFunc(self, Z):
        if(self.Activation=="relu"):
            return Z > 0
        elif(self.Activation=="tanh"):
            return 1 - np.square(self.ActivationFunc(Z))
        elif(self.Activation=="sigmoid"):
            exps=np.exp(Z-Z.max())
            return exps / exps.sum(axis=0)
        
    def SoftMaxFunc(self, Z):
        exps=np.exp(Z-Z.max())
        return exps / exps.sum(axis=0)
            
    def feedForward(self, X_train):
        self.Zc1 = convolve(X_train, self.Conv1Filter,1)
        self.Ac1 = self.ActivationFunc(self.Zc1)
        # print(self.Conv1Pool)
        self.Ac2 = AvrgPooling(self.Ac1, self.Conv1Pool, 2)
        self.Ac3 = AvrgPooling(self.Ac2, self.Conv1Pool, 2)
        #print(self.Ac3.shape)

        X = self.Ac3.reshape((288,self.Ac3.shape[3]))
        self.Xc = X
        
        self.caches.clear()
        Aprv = X
        for i in range (0, self.nLayer-2):
            Z = np.dot(self.arW[i],Aprv)
            A = self.ActivationFunc(Z)
            Aprv = A
            self.caches.append((Z,A))
            
        id = self.nLayer-2
        
        Z = np.dot(self.arW[id], Aprv)
        A = self.SoftMaxFunc(Z)
        self.caches.append((Z,A))

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
    
    def feedBackward(self):
        Ycheck = self.checkRes(self.Y_train)
        m = self.Y_train.size
        id = self.nLayer-2
        dW = []
        db = []
        
        t_dZ = self.caches[id][1] - Ycheck
        t_dW = 1 / m * np.dot(t_dZ,self.caches[id-1][1].T) 
        t_db = 1 / m * np.sum(t_dZ)
        prv_dZ = t_dZ
        
        dW.append(t_dW)
        db.append(t_db)
        
        for i in range (self.nLayer-3,-1,-1):
            if(i==0):
                t_dZ = np.dot( self.arW[i+1].T, prv_dZ ) * self.DeriAcFunc(self.caches[i][0])
                t_dW = 1/m * np.dot( t_dZ, self.Xc.T )
                t_db = 1/m * np.sum(t_dZ)
                prv_dZ = t_dZ
            else:
                t_dZ = np.dot( self.arW[i+1].T, prv_dZ ) * self.DeriAcFunc(self.caches[i][0])
                t_dW = 1/m * np.dot( t_dZ, self.caches[i-1][1].T )
                t_db = 1/m * np.sum(t_dZ)
                prv_dZ = t_dZ
                
            dW.append(t_dW)
            db.append(t_db)
                
        dW.reverse()
        db.reverse()
        
        #print(prv_dZ)
        
        dXc = np.dot( self.arW[0].T, prv_dZ ) * self.DeriAcFunc(self.Xc)
        #print(dXc[:,0])
        dAc3 = dXc.reshape((6,6,8,self.batchSize))
        
        dAc2 = backPooling(self.Ac2.shape, dAc3,2)
        dAc1 = backPooling(self.Ac1.shape, dAc2,2)
        dZc1 = dAc1 * self.DeriAcFunc(self.Zc1)
        
        dF1 = dwConvolve( ((3,3,1,8)), self.X_train, dZc1)/self.batchSize
        
        #print(dF1[:,:,0,:])
        
        return dW, db, dF1
    
    def gradientDescent(self, dW, db, dF1):
        for i in range (0, self.nLayer-1):
            self.arW[i] = self.arW[i] - self.rate * dW[i]
            self.arb[i] = self.arb[i] - self.rate * db[i]
            
        self.Conv1Filter = self.Conv1Filter - self.rate * dF1 
            
    def training(self):
        A = self.feedForward(self.X_train)
        dW, db, DF1 = self.feedBackward()
        self.gradientDescent(dW, db, DF1)
        
        return self.accuracy( self.predict(A), self.Y_train ), A
                
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
                
            print("[Epoch] ", i, ". Accuracy: ", s/n_iter, ";")
            self.costX.append(i)
            self.costY.append(cs/self.batchSize)
            
    def testing(self):
        A = self.feedForward(self.X_test)
        print("Accuracy: ", self.accuracy( self.predict(A), self.Y_test ) )
    
#MAIN FUNCTION    
Train = MnistTraing()

Train.loadDataFromFile("mnist_train.csv",210,"mnist_test.csv",67)

Train.setBacthSize(30)
Train.setLayer(3,[288,128,10])
Train.setActivation("relu")
Train.setLearningRate(0.1)
Train.randomPara()

# Train.loadBacth(0)
# A = Train.feedForward(Train.X_train)
# Train.feedBackward()
# print(Train.accuracy( Train.predict(A), Train.Y_train ))

Train.trainingMnist(11)

print("TESTING SET:")
Train.testing()

plt.plot(Train.costX, Train.costY)
plt.show()




