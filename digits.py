#!/usr/bin/env python
'''
Recognize Hand-Written Digits -- A "Getting Started" Kaggle Competition

    The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is.

@Author: Mikhail Klassen
@E-mail: mikhail.klassen@gmail.com

    Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
    Each pixel has a single pixel-value associated with it, indicating the lightness or darkness 
    of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 
    and 255, inclusive.

    The training data set, (train.csv), has 785 columns. The first column, called "label", is the 
    digit that was drawn by the user. The rest of the columns contain the pixel-values of the 
    associated image.

    Each pixel column in the training set has a name like pixelx, where x is an integer between 0 
    and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as 
    x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located 
    on row i and column j of a 28 x 28 matrix, (indexing by zero).

'''

import pylab as plt
from sklearn import datasets, svm, metrics
import numpy as np

print 'Reading in training data.'
training_data = np.genfromtxt('train.csv',delimiter=',',skip_header=1)
targets = training_data[:,0]
data    = training_data[:,1:]

print 'Reading in test data.'
test_data     = np.genfromtxt('test.csv',delimiter=',',skip_header=1)

n_samples = len(targets)

# As the classifier, we will use a supper vector machine (SVM).
# The default kernel for the support vector classifier (SVC) is 
# a Gaussian radial basis function with a gamma of 0.001
classifier = svm.SVC(kernel='poly')

# Train the SVM on part of the training data
#classifier.fit(data:1000, targets[:1000])
classifier.fit(data,targets)

# Now predict the value of the digit on the second half:
expected = targets[1000:]
predicted = classifier.predict(data[1000:])

print "Classification report for classifier %s:\n%s\n" % (
    classifier, metrics.classification_report(expected, predicted))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)

# Now run on real data
digits = classifier.predict(test_data)

f = open('submission.txt','w')
for i in range(len(digits)):
    f.write(str(int(digits[i])))
    f.write('\n')
f.close()
