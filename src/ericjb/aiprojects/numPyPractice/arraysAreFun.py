'''
Created on Nov 1, 2012

@author: Erics
'''
import numpy as np
import math
import copy

def multiply(x, y):
    return x*y

def xdoty(x, y):
    power = len(str(y))
    return x+(y*(10**-power))

# Flattens an array to one dimension, performs function on each element of a1 and a2,
# then expands the array back out to whatever dimensions are denoted by shape
# If any of the elements of shape are -1, the appropriate dimension will be filled in 
def arrayElementOps(function, a1, a2, shape, decimals=0):
    flatArray1 = a1.flat
    flatArray2 = a2.flat
    if decimals <= 0:
        flatArray3 = np.empty(len(a1.flat), dtype=int)
    else:
        flatArray3 = np.empty(len(a1.flat))
    for i in range(0, len(a1.flat)):
        num1 = flatArray1[i]
        num2 = flatArray2[i]
        result = function(num1, num2)
        flatArray3[i] = result
    flatArray3 = np.round(flatArray3, decimals)
    flatArray3 = flatArray3.reshape(shape)
    return flatArray3

def newLowerTriangleArray(shape):
    x,y = shape
    initialized = False
    triangleArray = np.array([])

    for i in range(1, x+1):
        tempArray = np.array([])
        tempList = []
        for j in range(1, i+1):
            tempList.append(j)
        for j in range(x-i):
            tempList.append(0)
            
        tempArray = np.array(tempList)
        if not initialized:
            initialized = True
            triangleArray = copy.deepcopy(tempArray)
        else:
            triangleArray = np.vstack([triangleArray, tempArray])
    return triangleArray



test1 = np.array([1, 2, 3, 4, 5])
test2 = np.array([1, 0, 1, -1, 1])
test3 = test1*test2
print test3
triangleArray = newLowerTriangleArray((5, 5))
print triangleArray

#b[:,1]  (all rows from second column)
data1 = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12]])
data2 = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12]])
#data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#data2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

newData = arrayElementOps(multiply, data1, data2, (-1,3), -1)
newData2 = arrayElementOps(xdoty, data1, data2, (-1,3), 2)
newData3 = arrayElementOps(xdoty, data1, data2, (-1,2), 2)

print newData
print newData2
print newData3

print '?'





w = np.random.random(data1.shape[1] + 1) * .01 - .005




def decimalArray(x, y):
    print y
    return 10+10*x+y
    
def rowNumber(x, y):
    return x
def columnNumber(x, y):
    return y

red = [0, 0, 1, 0, 1, 1, 1]
def rDist(x):
    return x


r = np.fromfunction(rowNumber, (4, 5))

c = np.fromfunction(columnNumber, (4, 5))

redThing = np.zeros((256,), dtype=int)
for i in range(len(redThing)):
    redThing[i] = red[(i % 7)]
print '?'
