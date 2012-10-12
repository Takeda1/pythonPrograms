""" A short script that will contain solutions to numpy exercises
for CS444 HW6. 
"""

import numpy as np


# Declare numpy arrays here...  

A = np.array([[ 2, -5,  1],
              [ 1,  4,  5],
              [ 2, -1,  6]])

B = np.array([[ 1,  2,  3],
              [ 3,  4, -1]])

y = np.array([[ 2],
              [-4],
              [ 1]])

z = np.array([[-15],
              [ -8],
              [-22]])


# Print out the result of computations... 
print "BA:\n",np.dot(B,A)
print "\nAB':\n",np.dot(A,B.T)
print "\nAy:",np.dot(A,y)
print "\ny'z:",np.dot(y.T,z)
print "\nyz':\n",np.dot(y,z.T)
print "\nx:", np.dot(np.linalg.inv(A),z)

# Practice looping and slicing... 
print "\nPrinting rows of A:"
for row in A:
    print row

print "\nPrinting columns of A:"
for i in range(0, len(A)):
    print A[:,i]
