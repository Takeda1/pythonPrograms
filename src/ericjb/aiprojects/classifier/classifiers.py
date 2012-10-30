""" This module contains image classifiers. 

"""

import numpy as np

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
               value of the pixel at row 20 colum 13 of image 4.
  
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
            features[i,1] = np.mean(images[i][:,:,1]) #mean red value;
            features[i,2] = np.mean(images[i][:,:,2]) #mean red value;
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

class NNRedClassifier(NNClassifier):
    """ The NNRedClassifier performs nearest neighbor classification
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