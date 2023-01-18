Multinomial Naive Bayes Classification using MLE and MAP estimators for the CS464: Introduction to Machine Learning course homework 1.
Implemented on Python Jupyter Notebook

The dataset consists of two parts: training (552 x 4164) and validation (185 x 4164). There are 4163 words (columns) in both of the sets and 552 news articles in the training and 185 in the validation data set. Each cell represents the amount of the word in the document. The last column shows the class labels. It is taken from BBC sports news and the purpose of the model is to classify 185 documents into 5 sports news categories: athletics (class label = 0), cricket (1), football (2), rugby (3) and tennis (4).

Bag-of-Words Representation is used to train the models. MLE estimator gives 31.35% accuracy and MAP estimator gives 97.30% accuracy.
MAP estimator is applied with Laplace smoothing (alpha = 1) to increase the accuracy of the model.

First, an analysis is done on the data sets to see their class distributions. They both turned out to be skewed towards class (2). Next, for the MLE classifier, the pi value is computed for each of the classes. The pi value for a class yk indicates the probability of a particular document having class label yk. Then the theta value is computed for each of the words given in the document. Theta represents the probability a particular word existing  in a document with label yk.

Pi is a dictionary object and is computed by counting the number of documents with the specific label then dividing that number to the total number of documents. This is done for all classes. The keys are the class labels and the values are the probabilities. 

Theta is a dictionary object and its keys are the class labels. The values are also dictionaries with keys as the words and values as the probabilities. Theta is computed by looking at the amount of the specific word in a document with a specific class label and dividing it with the total number of words with that class label. Feature extraction for words are done before (in an array).

Theta and pi are used in the MNB function. Function takes a test data and converts it into an array (for faster run time). Then guesses the label for each document in the data using bag of words representation (multiplying the appropriate theta and pi functions). Each label is put in an array and the function returns this array. Logarithmic expression is used for easier computation. Some of the values in theta are 0 therefore they are represented with np.nan_to_num(-np.inf) (log0 = -inf).

Accuracy is computed using calc_accuracy function which compares the class labels in the validation data and the class labels predicted by the MNB function when the validation data is put without the class labels. 

Confusion matrix is produced to see how many incorrect guesses were made by the model. 

MAP estimator follows the same steps except its theta function is altered by adding 1 to the numerator and 1 + (total words) to the denominator.
