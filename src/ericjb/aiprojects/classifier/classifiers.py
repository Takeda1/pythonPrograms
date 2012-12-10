""" This module contains image classifiers. 

"""

import numpy as np
from math import log
import math




def Harmonic(n):
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

def reverseHarmonic(h):
    error = 1.0
    r = 1.0
    prev = 0
    tested = []
    close = False
    reverse = 0
    bestError = 1
    while (error > .001 or error < -.001) and int(r) not in tested:
        tested.append(int(r))
        if prev < h and close == False:
            r = r + math.ceil(h/100.0)
            prev = r*Harmonic(r)
        elif prev > h:
            close = True
            r = r - math.ceil(h/1000.0)
            prev = r*Harmonic(r)
        error = (h - prev)/float(h)
        if abs(error) < abs(bestError):
            bestError = error
            reverse = int(r)
    return reverse




class ImageClassifier(object):
    """ ImageClassifier is an abstract superclass for classes that are
    designed to solve image classification problems.  All image
    classifier classes must inherit from this class and override the
    train and classify methods.
    """
    
    def __init__(self, images, labels):
        """ ImageClassifier constructor.  
        
        Arguments: 
           images - a list of length n of numpy arrays representing images.  
               Each entry in the list is of size h x w x 3 where h is
               the number of rows, w the number of columns, and 3 is
               the number of colors in the image-- 0 corresponds to red,
               1 corresponds to blue, and 2 corresponds to green.

               For example, the following statement accesses the red
               value of the pixel at row 20 column 13 of image 4.
  
                         row  col color
                           \   \   \
                            v   v   v
                  images[4][20, 13, 0]

           labels - A numpy array of length n containing the labels for all
                    images.  For example, 0 for indoor, 1 for outdoor.

        """
        self._train_images = images
        self._labels = labels
        self._train_features = self._compute_features(images)

    def _compute_features(self, images):
        """ Convert each image to a feature vector.  

        This method represents each images as a vector with three
        entries, where each entry represents the mean of one of the
        color channels in the corresponding image.

        THIS IS NOT A VERY GOOD FEATURE. You will get better
        performance if you overide this method with something more
        clever in your subclasses.
        """
        features = np.zeros((len(images), 3))
        for i in range(len(images)):
            features[i,0] = np.mean(images[i][:,:,0]) #mean red value;
            features[i,1] = np.mean(images[i][:,:,1]) #mean blue value;
            features[i,2] = np.mean(images[i][:,:,2]) #mean green value;
        return features

    def train(self):
        """ Train this classifier using self._train_images and self._labels
        as the training data.
        """
        raise NotImplementedError

    def classify(self, images):
        """  Classify a set of images. 

        Arguments: 
            images - a list of images of length n.
            
        Returns: 
            a numpy array of length n containing 0's and 1's corresponding
            to the predicted classes for each image. 

        Precondition: train must have already been called.
        """
        raise NotImplementedError



class NNClassifier(ImageClassifier):
    """ Nearest Neighbor Classifier

    This is a sample classifier class that classifies images using
    the 1-nearest-neighbor algorithm and the default feature set defined in 
    the ImageClassifier superclass.
    """

    def __init__(self, images, labels):
        """ This will just call the superclass constructor.  For a
        more complex classifier, you might want to use this method to
        set up parameters needed for your learning algorithm.
        """
        # Call the superclass constructor...
        super(NNClassifier, self).__init__(images, labels)
        # Additional set-up code could go here...
    
    def train(self):
        """ No training is required for nearest-neighbor classification """
        pass

    def classify(self, test_images):
        """ Classify each image from images using the 1-nearest-neighbor
        classification algorithm. 
        """
        
        #Create a numpy array to hold the labels...
        labels = np.zeros(len(test_images))

        # Calculate features for the images that need to be classified...
        test_features = self._compute_features(test_images)

        # This is a loop that will classify each image in turn...
        for row in range(len(test_images)):
            # Grab the feature vector of the current image...
            cur_features = test_features[row,:]

            # Fancy numpy code that calculates the Euclidian distance
            # from the current image to all images in the training data...
            dists = np.sqrt(np.sum((self._train_features - cur_features)**2, 
                                   axis=1))
            
            # Sort the distances so we can find the most similar image...
            sorted_indices = np.argsort(dists)

            # Label the current image according to the label of its nearest
            # neighbor... 
            labels[row] = self._labels[sorted_indices[0]]
        return labels

class KNNClassifier(ImageClassifier):
    """ Nearest Neighbor Classifier

    This is a sample classifier class that classifies images using
    the 1-nearest-neighbor algorithm and the default feature set defined in 
    the ImageClassifier superclass.
    """

    def __init__(self, images, labels):
        """ This will just call the superclass constructor.  For a
        more complex classifier, you might want to use this method to
        set up parameters needed for your learning algorithm.
        """
        # Call the superclass constructor...
        super(KNNClassifier, self).__init__(images, labels)
        # Additional set-up code could go here...
    
    def train(self):
        """ No training is required for nearest-neighbor classification """
        pass

    def classify(self, test_images):
        """ Classify each image from images using the 1-nearest-neighbor
        classification algorithm. 
        """
        neighbors = 60
        
        
        #Create a numpy array to hold the labels...
        labels = np.zeros(len(test_images))

        # Calculate features for the images that need to be classified...
        test_features = self._compute_features(test_images)

        # This is a loop that will classify each image in turn...
        for row in range(len(test_images)):
            # Grab the feature vector of the current image...
            cur_features = test_features[row,:]

            # Fancy numpy code that calculates the Euclidian distance
            # from the current image to all images in the training data...
            dists = np.sqrt(np.sum((self._train_features - cur_features)**2, 
                                   axis=1))
            
            # Sort the distances so we can find the most similar image...
            sorted_indices = np.argsort(dists)

            # Label the current image according to the label of its nearest
            # neighbor... 
            temp_labels = []
            weights = []
            for i in range(0, neighbors):
                weights.append(1/dists[sorted_indices[i]])
                temp_labels.append(self._labels[sorted_indices[i]])
            for i in range(len(temp_labels)):
                if temp_labels[i] == 0.0: temp_labels[i] = -1
            classification = sum(np.array(temp_labels)*np.array(weights))
            if classification >0 : labels[row] = 1
            else:                  labels[row] = 0
        return labels


class KNNBetterClassifier(KNNClassifier):
    def _compute_features(self, images):
        gridSize = 4
        imageSize = 64
        subImageSize = imageSize/gridSize
        features = np.zeros((len(images), 158))
        
        for i in range(len(images)):
            
            subImages = []
            for r in range(gridSize):
                for c in range(gridSize):
                    if r == (gridSize-1):
                        if c == (gridSize-1):
                            subImages.append(images[i][(r*subImageSize):(imageSize-1), (c*subImageSize):(imageSize-1), :])
                        else:
                            subImages.append(images[i][(r*subImageSize):(imageSize-1), (c*subImageSize):((c*subImageSize)+(subImageSize)), :])
                    elif c == (gridSize-1):
                        subImages.append(images[i][(r*subImageSize):((r+1)*subImageSize), (c*subImageSize):(imageSize-1), :]) 
                    else:
                        subImages.append(images[i][(r*subImageSize):((r+1)*subImageSize), (c*subImageSize):((c*subImageSize)+(subImageSize)), :])
            
            count = 0
            
            for sub in subImages:
                fRed = np.mean(sub[:,:,0])
                fGreen = np.mean(sub[:,:,1])
                fBlue = np.mean(sub[:,:,2])
                features[i,count] = np.median(sub[:,:,0]) #mean red value;
                count += 1
                features[i,count] = np.median(sub[:,:,1]) #mean blue value;
                count += 1
                features[i,count] = np.median(sub[:,:,2]) #mean green value;
                count += 1
                features[i,count] = np.var(sub[:,:,0]) #mean red value;
                count += 1
                features[i,count] = np.var(sub[:,:,1]) #mean blue value;
                count += 1
                features[i,count] = np.var(sub[:,:,2]) #mean green value;
                count += 1
                features[i, count] = fRed
                count += 1
                features[i, count] = fGreen
                count += 1
                features[i, count] = fBlue
                count += 1
                """
                numBins = 3
                
                newsImage = np.sum( [(np.multiply(images[i][:,:,0], numBins**1).astype(int)).flatten(), 
                                    (np.multiply(images[i][:,:,1], numBins**2).astype(int)).flatten(), 
                                    (np.multiply(images[i][:,:,2], numBins**3).astype(int)).flatten()], axis=0)
                fsColorDist = np.argsort(np.bincount(newsImage, minlength=subImageSize**2))[::-1]
                #fsColor2 = fsColorDist[0]
                fsColor1 = fsColorDist[-1]
                features[i, count] = fsColor1
                count += 1
                #features[i, count] = fsColor2
                #count += 1
                """
            
            
            
            
            
            numBins = 11
            newImage = np.sum( [(np.multiply(images[i][:,:,0], numBins**1).astype(int)).flatten(), 
                                (np.multiply(images[i][:,:,1], numBins**2).astype(int)).flatten(), 
                                (np.multiply(images[i][:,:,2], numBins**3).astype(int)).flatten()], axis=0)
            fColorDist = np.argsort(np.bincount(newImage, minlength=4096))[::-1]
            fColor1 = fColorDist[-1]

            
            features[i,count]   = np.mean(images[i][:,:,0]) #mean red value;
            features[i,count+1] = np.mean(images[i][:,:,1]) #mean blue value;
            features[i,count+2] = np.mean(images[i][:,:,2]) #mean green value;
            features[i,count+3] = np.median(images[i][:,:,0]) #mean red value;
            features[i,count+4] = np.median(images[i][:,:,1]) #mean blue value;
            features[i,count+5] = np.median(images[i][:,:,2]) #mean green value;
            features[i,count+6] = np.var(images[i][:,:,0]) #mean red value;
            features[i,count+7] = np.var(images[i][:,:,1]) #mean blue value;
            features[i,count+8] = np.var(images[i][:,:,2]) #mean green value;
            features[i, count+9] = fColor1
        return features


"""
class LogisticClassifier(ImageClassifier):
    
    def __init__(self, images, labels):
        # Call the superclass constructor...
        super(LogisticClassifier, self).__init__(images, labels)
        self.weights = None

    def logistic(self, a):
        return 1. / (1. + np.exp(-a))
    
    def logistic_prime(self, a):
        return self.logistic(a) * (1. - self.logistic(a))
    
    def train(self, epochs=500, alpha=.05):
        data = np.array(self._train_features)
        labels = np.array(self._labels)
        
        #randomly generate  a weight vector:
        w = np.random.random(data.shape[1] + 1) * .01 - .005
        
        #add a column of bias values to the input data:
        data = np.append(np.ones((data.shape[0], 1)), data, axis=1)
        
        #initialize the error arrays:
        errors = np.zeros(epochs)
        error_counts = np.zeros(epochs)

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
            for row in np.random.permutation(self._train_features.shape[0]):
                activation = self.logistic(np.inner(w, data[row,:]))
                diff = labels[row]-activation
                gp = self.logistic_prime(np.inner(w, data[row,:]))
                w += alpha*diff*gp*data[row, :]
                if labels[row] != np.round(activation):
                    error_counts[epoch] += 1
                errors[epoch] += .5 * diff **2
        self.weights = w
    
    def classify(self, test_images):
        labels = np.zeros(len(test_images))
        test_features = self._compute_features(test_images)
        test_features = np.append(np.ones((test_features.shape[0], 1)), test_features, axis=1)
        for row in range(len(test_images)):
            # Grab the feature vector of the current image...
            cur_features = test_features[row,:]
            # Multiply the feature vector by the trained weights, and get the sum
            classification = sum(cur_features*self.weights)
            if classification >=0: labels[row] = 1
            else:                  labels[row] = 0
        return labels


class LogisticBetterClassifier(LogisticClassifier):
    def _compute_features(self, images):
        gridSize = 4
        imageSize = 64
        subImageSize = imageSize/gridSize
        features = np.zeros((len(images), 51))
        
        for i in range(len(images)):
            subImages = []
            for r in range(gridSize):
                for c in range(gridSize):
                    if r == (gridSize-1):
                        if c == (gridSize-1):
                            subImages.append(images[i][(r*subImageSize):(imageSize-1), (c*subImageSize):(imageSize-1), :])
                        else:
                            subImages.append(images[i][(r*subImageSize):(imageSize-1), (c*subImageSize):((c*subImageSize)+(subImageSize)), :])
                    elif c == (gridSize-1):
                        subImages.append(images[i][(r*subImageSize):((r+1)*subImageSize), (c*subImageSize):(imageSize-1), :]) 
                    else:
                        subImages.append(images[i][(r*subImageSize):((r+1)*subImageSize), (c*subImageSize):((c*subImageSize)+(subImageSize)), :])
            count = 0
            for sub in subImages:
                fRed = np.mean(sub[:,:,0])
                fGreen = np.mean(sub[:,:,1])
                fBlue = np.mean(sub[:,:,2])
                feature = 1
                features[i, count] = fRed
                count += 1
                features[i, count] = fGreen
                count += 1
                features[i, count] = fBlue
                count += 1
            features[i,count]   = np.mean(images[i][:,:,0]) #mean red value;
            features[i,count+1] = np.mean(images[i][:,:,1]) #mean blue value;
            features[i,count+2] = np.mean(images[i][:,:,2]) #mean green value;
        return features
"""
    
    
    
    
    
    






class NNRedClassifier(NNClassifier):
    """
    The NNRedClassifier performs nearest neighbor classification
    based only on the average of the red channel.  This is even worse
    than the default feature vector.  This class is included only to
    demonstrate that we can easily test different feature functions by
    creating new sub-classes that overide the _compute_features
    method.
    """
    
    def _compute_features(self, images):
        features = np.zeros((len(images), 1))
        for i in range(len(images)):
            features[i,0] = np.mean(images[i][:,:,0]) #mean red value;
        return features




if __name__ == "__main__":
    pass