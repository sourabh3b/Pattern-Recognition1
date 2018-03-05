# Pattern-Recognition Assignment 1
In this problem we will apply discriminant analysis to recognize the digits in the MNIST data set (http://yann.lecun.com/exdb/mnist/). As a bonus problem we will construct "Fisher digits". We will train our model using the training data sets ("train-images-idx3-ubyte.gz" and "train-labels-idx1-ubyte.gz") and test the performance using the test data set ("t10k-images-idx3-ubyte.gz" and "t10k-labels-idx1-ubyte.gz").
1. The images are 28 x 28 pixels in gray-scale. The categories are 0, 1, ... 9. We concatenate the image rows into a 28 x 28 vector and treat this as our feature, and assume the feature vectors in each category in the training data "train-images-idx3-ubyte.gz") have Gaussian distribution. Draw the mean and standard deivation of those features for the 10 categories as 28 x 28 images using the training images ("train-images-idx3-ubyte.gz"). There should be 2 images for each of the 10 digits, one for mean and one for standard deviation. We call those "mean digits" and "standard deviation digits" in CSE455/555.

2. Classify the images in the testing data set ("t10k-images-idx3-ubyte.gz") using 0-1 loss function and Bayesian decision rule and report the performance. Why it doesn't perform as good as many other methods on LeCuns web page? Before coding the discriminant functions, review Section 2.6.


Running the program :
> go run main.go 

``follow the instructions in the command line``

```python
To run part 2

sourabh:Pattern-Recognition1 sourabh$ go run main.go 
Part 1 (type 1) or Part 2 (type 2)
1
executing part 1
Enter digit followed by type (0 for mean, 1 for standard deviation) :
0 1
Total image given in the data set  =  60045
Calculating standard Deviation 


0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4.24 4.7 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 3.1 4.7 5.04 5.04 5.04 3.03 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 4.8 5.04 4.7 4.8 4.01 0 3.04 4.5 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 3.98 5.04 5.04 3.05 0 0 0 4.01 4.30 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 4.88 5.04 5.04 0 0 0 0 0 0 3.4 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 4.5 4.7 2.7 0 0 0 0 0 0 0 4.5 5.7 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 4.7 5.04 3.2 0 0 0 0 0 0 0 4.7 2.4 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 2.05 4.7 3.74 0 0 0 0 0 0 0 0 4.7 5.04 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 5.04 4.9 0 0 0 0 0 0 0 0 0 4.7 5.04 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 2.44 5.04 3.56 0 0 0 0 0 0 0 0 0 4.7 3.41 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 3.78 4.7 0 0 0 0 0 0 0 0 0 4.7 4.5 4.7 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.06 5.04 0 0 0 0 0 0 0 0 0 4.88 4.7 4.6 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.8 5.04 0 0 0 0 0 0 0 0 4.7 5.04 4.7 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 5.04 5.04 0 0 0 0 0 0 0 0 4.88 5.04 3.4 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 5.04 5.04 0 0 0 0 0 0 0 3.6 5.04 5.04 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.49 4.7 0 0 0 0 0 0 4.24 4.7 4.7 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.5 4.7 4.9 0 0 0 3.1 4.7 5.04 5.04 5.2 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 5.04 4.7 3.75 3.36 4.8 5.04 4.7 4.8 2.31 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 4.04 4.7 5.04 5.04 5.04 5.04 4.9 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 3.55 5.04 5.04 3.47 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Next Step : 

Put this image to matlab for getting image in png format
command : 
>> m = [
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4.24 4.7 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 3.1 4.7 5.04 5.04 5.04 3.03 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 4.8 5.04 4.7 4.8 4.01 0 3.04 4.5 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 3.98 5.04 5.04 3.05 0 0 0 4.01 4.30 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 4.88 5.04 5.04 0 0 0 0 0 0 3.4 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 4.5 4.7 2.7 0 0 0 0 0 0 0 4.5 5.7 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 4.7 5.04 3.2 0 0 0 0 0 0 0 4.7 2.4 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 2.05 4.7 3.74 0 0 0 0 0 0 0 0 4.7 5.04 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 5.04 4.9 0 0 0 0 0 0 0 0 0 4.7 5.04 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 2.44 5.04 3.56 0 0 0 0 0 0 0 0 0 4.7 3.41 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 3.78 4.7 0 0 0 0 0 0 0 0 0 4.7 4.5 4.7 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.06 5.04 0 0 0 0 0 0 0 0 0 4.88 4.7 4.6 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.8 5.04 0 0 0 0 0 0 0 0 4.7 5.04 4.7 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 5.04 5.04 0 0 0 0 0 0 0 0 4.88 5.04 3.4 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 5.04 5.04 0 0 0 0 0 0 0 3.6 5.04 5.04 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.49 4.7 0 0 0 0 0 0 4.24 4.7 4.7 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 4.5 4.7 4.9 0 0 0 3.1 4.7 5.04 5.04 5.2 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 5.04 4.7 3.75 3.36 4.8 5.04 4.7 4.8 2.31 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 4.04 4.7 5.04 5.04 5.04 5.04 4.9 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 3.55 5.04 5.04 3.47 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
>> image(m)

```

This returns :  
![alt text](https://raw.githubusercontent.com/sourabh3b/Pattern-Recognition1/master/i0.png "Test Image")


```python
To run part 2
sourabh:Pattern-Recognition1 sourabh$ go run main.go 
Part 1 (type 1) or Part 2 (type 2)
2
executing part 2
Running  1  test case  10 times
test case #  1
test case #  2
test case #  3
test case #  4
test case #  5
test case #  6
test case #  7
test case #  8
test case #  9
test case #  10
Accuracy :=  80  %
```



References:

[1] : Pedro Domingos , Michael Pazzani, On the Optimality of the Simple Bayesian Classifier under Zero-One Loss, Machine Learning, v.29 n.2-3, p.103-130, Nov./Dec. 1997 

[2] : Richard O. Duda , Peter E. Hart , David G. Stork, Pattern Classification (2nd Edition), Wiley-Interscience, 2000


