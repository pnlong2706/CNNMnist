import matplotlib.image as img
from matplotlib import pyplot as plt
import numpy as np

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

def RELU(Z):
    return np.maximum(0,Z)

image = img.imread("./flow.jpg")
print(image.shape)

image = image / 256
print(image)

image = image.reshape((image.shape[0],image.shape[1],3,1))

fil = [ [ [[0], [0], [0]], [[1], [1], [1]], [[0], [0], [0]] ], [ [[1], [1], [1]], [[-4], [-4], [-4]], [[1], [1], [1]] ], [ [[0], [0], [0]], [[1], [1], [1]], [[0], [0], [0]] ] ]
fil = np.array(fil)

#fil = np.random.rand(3,3,3,3) - 0.5

pool = np.zeros((2,2,1)) + 0.25

print(fil.shape)
print(fil[:,:,0,0])

ans = convolve(image, fil, 1)
ans = RELU(ans)

ans = AvrgPooling(ans, pool, 2)
print(ans.shape)
ans = ans.reshape((ans.shape[0],ans.shape[1])) * 255

plt.imshow(ans)
plt.show()

