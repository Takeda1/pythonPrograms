""" Testing and utility code for the indoor/outdoor classification project.
Author: Nathan Sprague
Date: 10/26/2012
"""

import classifiers
import numpy as np
import matplotlib.pylab as pylab
import os

def load_images(directory, ext):
    """ This method loads all images with a particular extension
    from the indicated directory.

    PARAMETERS: 
    directory -  A string containing the appropriate directory.
    ext -  The desired file extension. For example '.jpg'.
    
    RETURNS
       images - A list of images represented as numpy arrays. 
    """
    files = [f for f in os.listdir(directory) if f.endswith(ext)]
    images = []
    for i in range(len(files)):
        images.append(pylab.imread(directory + files[i]))

    return images
  

def evaluate_classifier(classifier, testing_data, testing_labels):
    """ This method tests a classifier object on the provided
    testing data, and prints the results.
    """
    classifier.train()
    output_labels = classifier.classify(testing_data)
    errors = np.sum(np.abs(output_labels - testing_labels))
    percent_correct = 100.0 - 100.0 * (float(errors) / len(testing_labels))

    output_str = "Errors for {}: {}/{}, {}% correct."

    print(output_str.format(classifier.__class__.__name__,
                            errors,
                            len(output_labels),
                            percent_correct))


def load_data():
    """ Load images for the indoor/outdoor classification task.
    """
    indoor_test_images = load_images("./images/test_indoor/", "png")    
    outdoor_test_images = load_images("./images/test_outdoor/", "png")
    indoor_train_images = load_images("./images/train_indoor/", "png")    
    outdoor_train_images = load_images("./images/train_outdoor/", "png")

    all_train = indoor_train_images + outdoor_train_images
    all_train_labels = np.append(np.zeros(len(indoor_train_images)),
                                 np.ones(len(outdoor_train_images)))

    all_test = indoor_test_images + outdoor_test_images
    all_test_labels = np.append(np.zeros(len(indoor_test_images)),
                                 np.ones(len(outdoor_test_images)))
    
    return all_train, all_train_labels, all_test, all_test_labels
   
def test_all():
    """ Peek into the classifier module to find all classes whose name end in 
    'Classifier'.  Instantiate and test each of these. 
    """

    all_train, all_train_labels, all_test, all_test_labels = load_data()

    constructors = [item for item in dir(classifiers) if \
                        item.endswith('Classifier')]
    constructors.remove("ImageClassifier")

    for constructor in constructors:
        classifier = classifiers.__dict__[constructor](all_train, 
                                                       all_train_labels)

        evaluate_classifier(classifier, all_test, all_test_labels)    

def test_one():
    """ If you wanted to set up a method to test just one classifier class,
    you could do it like this. 
    """
    all_train, all_train_labels, all_test, all_test_labels = load_data()
    nn_classifier = classifiers.NNClassifier(all_train, all_train_labels)
    evaluate_classifier(nn_classifier, all_test, all_test_labels) 

if __name__ == "__main__":
    test_all()
    #test_one()