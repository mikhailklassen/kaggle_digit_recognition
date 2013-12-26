# Digit Recognizer 

My submission to the Digit Recognizer challenge on Kaggle. See [here](http://www.kaggle.com/c/digit-recognizer) for details. Uses the scikit-learn machine learning libraries for Python to fit a model to the training data (`train.csv`) and run it on the test data (`test.csv`). The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (`train.csv`), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as u = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

## Prerequisites:

Python 2.6+

[scikit-learn](http://scikit-learn.org/stable/)

[NumPy](http://www.numpy.org/)

## Run:

From the command line:

    > python digits.py


