""" Logistic regression based on stochastic gradient descent. 

  UNFINISHED!

Author: Nathan Sprague
Modified by: 
Version: Oct. 2, 2012

"""
import matplotlib.pyplot as plt
import numpy as np


def logistic(a):
    """ The logistic (sigmoid) function. """
    return 1. / (1. + np.exp(-a))

def logistic_prime(a):
    """ The derivative of the logistic function """
    return logistic(a) * (1. - logistic(a))

def train_logistic(data, labels, epochs, alpha):
    """
    Use stochastic gradient descent to train single layer, single
    output neural network with a sigmoid activation
    function. (Equivalently, find weights for logistic regression.)
    
    Parameters:
    data -      n x m numpy array, n inputs of m dimensions each
    labels -    length n array containing labels (0 or 1) 
    epochs -    number of times to run through the data set
    alpha -     learning rate parameter
    
    Returns:
    w -            final weight vector of length m + 1 (includes bias weight)
    errors -       array of length epochs.  Each entry contains the sum
                   squared error for the corresponding epoch.
    error_counts - Array of length epochs.  Each entry contains the number
                   of misclassified points for the corresponding epoch. 
    """
    #randomly generate  a weight vector:
    w = np.random.random(data.shape[1] + 1) * .01 - .005
    
    #add a column of bias values to the input data:
    data = np.append(np.ones((data.shape[0], 1)), data, axis=1)
    
    #initialize the error arrays:
    errors = np.zeros(epochs)
    error_counts = np.zeros(epochs)
    
    #-------------- YOUR CODE HERE!-------------------
    #YOU NEED TO FILL IN THE CODE TO HANDLE THE WEIGHT
    #UPDATES AND ERROR EVALUATION
    
    # Once you get the train_logisistic function working, 
    # you should be able to use to do written character recognition 
    # by calling test_digits from the main. 
    
    #    w_i <- w_i + alpha*(y-h_w(x))*h_w(x)(1-h_w(x))*x_i
    #    w   <-  w  + a*(y-g(wT*x))*g'(wT*x)*x
    #    x's = "data".row
    #    y's = "label"
    #      g = logistic
    #    w   <-  w  + alpha*(y-g(wT*x))*g'(wT*x)*x
    
    
    
    def grad_logistic(x):
        idxplus=np.where(x>0)
        idxminus=np.where(x<=0)
        grad=np.zeros(len(x))
        xplus=x[idxplus]
        grad[idxplus]=-np.exp(-xplus)/(1.0+np.exp(-xplus))
        xminus=x[idxminus]
        grad[idxminus]=-1.0/(1.0+np.exp(xminus))
        return grad    
    
    for epoch in range(epochs):
        for row in range(data.shape[0]):
            activation = logistic(np.inner(w, data[row,:]))
            diff = labels[row]-activation
            gp = logistic_prime(np.inner(w, data[row,:]))
            w += alpha*diff*gp*data[row, :]
            if labels[row] != np.round(activation):
                error_counts[epoch] += 1
            errors[epoch] += .5 * diff **2

    return w, errors, error_counts
            
#-----------------------------------------------------
#TEST CODE STARTS HERE::::::::::::::::::::::::::::::::
#-----------------------------------------------------

def test_logic():
    """ Train a logistic regression classifier to learn the OR function. 
    """
    data = np.array([[ 0., 0.], [ 0., 1.], [ 1., 0.], [ 1., 1.]])
    labels = np.array([0., 1., 1., 1.]) # OR function
    w, errors, error_counts = train_logistic(data,labels,500,.05)
    plt.plot(errors)
    plt.plot(error_counts)
    plt.xlabel('epoch')
    plt.legend(['sum squared error', 'error count'])
    plt.show()
    
def test_digits():
    """ Train the classifer to distinguish between handwritten 0's and 1's. 
    """
    usps = np.load('usps.npy')
    #WE CAN SHOW SOME SAMPLE DIGITS LIKE THIS:
    plt.subplot(1,2,1)
    #show the 20th "1"
    plt.imshow(np.reshape(usps[:,20,0],(16,16)).T,
               cmap=plt.cm.get_cmap('r', None), interpolation='nearest')
    plt.subplot(1,2,2)
    #show the third "0"
    plt.imshow(np.reshape(usps[:,3,9],(16,16)).T,
               cmap=plt.cm.get_cmap('b', None), interpolation='nearest')
    plt.show()

    #create an array containing the images of 1's
    data = usps[:,:,0].T
    #append images of 0's
    data = np.append(data,usps[:,:,9].T,axis=0)

    #rescale the data to a more reasonable range (-.5, .5): 
    data = data / 255.0 - .5

    #set up label array
    labels = np.zeros(data.shape[0]);
    labels[0:1100] = 1

    #train!
    w, errors, error_counts = train_logistic(data, labels, 200, .05)
    plt.plot(errors)
    plt.plot(error_counts)
    plt.xlabel('epoch')
    plt.legend(['sum squared error', 'error count'])
    plt.show()

if __name__ == "__main__":
    test_logic()

