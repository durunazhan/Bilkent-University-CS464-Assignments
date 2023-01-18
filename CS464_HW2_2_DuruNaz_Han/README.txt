For Q1:

In the first part 5239 images of dogs are resized into 64x64 pixels by using bilinear interpolation. Next, they are all flatten into 64x64x3 to separate the R,G,B channels later on. A 3D array of X contains 5239 images of size 4096x3 and the array is sliced into 3 (R,G,B) as x_1, x_2 and x_3. First 10 PCAs of these three channels are found and stacked on top of each other to obtain 10 different images of dogs in the end. 

In the second part using the PCAs found in the first part the first image in the dog pictures data set is reconstructed using different values of k. The number of k can be altered manually. The entire script needs to be reran for this part to work correctly. 

For Q2:

A data set of smart grids with size (60000x12) is used to detect whether they are stable or unstable. First, the data is split into three: train (70%), test (10%), validation (20%). Next, logistic regression classifier with mini batch gradient ascent and full batch gradient ascent is applied with initial weight taken from Gaussian distribution (completely random). The confusion matrix and accuracies for both models is displayed alongside with a heat map of the confusion matrix. Initialization is also changed to uniform distribution and zeros for mini-bath gradient descent and the results are also displayed with confusion matrix, its heat map and accuracies.  