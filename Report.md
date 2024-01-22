# Introduction
Hello there! This report aims to be a comprehensive overview of this project. Happy Reading! 

# Implementation
This library consists of 7 classes. 6 of these classes implement the functions required for algorithm. And one class contains some useful functions which are regularly used in training a machine learning model. I have provided great detail about the arguments and returns as comments in my code, hence this section shall only dive into how I trained the models effectively. The plots have also been provided in the .iypnb notebook.

## Linear Regression
This was a very straight forward to fine tune the hyperparameters. After trying out different alpha values, I eventually settled with an alpha value of 0.01 with 20000 epochs. Mean Absolute Error: 0.08024026566607234, 
Mean Squared Error: 0.010129193965037607, 
R2 Score: 0.9999999999216167 

## Polynomial Regression
This was also another easy process and it was made easier after visualizing the data. On visualization, the data seems to be symmettric, hence ruling out all odd polynomials. Personally, 6th degree polynomial performed the best with alpha 0.1 and 50,000 epochs. Mean Absolute Error: 3.330624893074551e-08,
Mean Squared Error: 2.2836959870525088e-14,
R2 Score: 1.0

### poly_features() and generate_feature_combinations()
This pair of functions is the backbone of polynomial regression. poly_features is written in simple code which is understood easily in first go, but generate_feature_combinations() uses recursive methods to get a list of every possible combinations, hence requires some documentation on how it works.

The recursive function has two base cases and one recursive case.

Base Case One : When the current degree of the combination exceeds the degree of polynomial required, an empty list is returned, and all the stacks merge to form the required combination.

Base Case Two : When the index of the combination reaches the end of the feature list, this results in end of number of features, hence the current combination is returned as such and the resulting stacks generates the required combination

Recursive Case : A for loop with the i on the range equal to the difference in degrees + 1 (because of the last number being exclusive), this adds i with the current combination to result in a new combination, which is then recursively called.

Logic : This method traverses through all possible combinations and repeately checks if they are at the limit (if max degree or max index is reached). Till then, the combinations are produced and when the limit reaches, the stacks create the combinations required.

## Logistic Regression
Only one problem was encountered during the training of this model, and that was overflow error. This was countered after clipping x to (-700, 700). In the end, the alpha was chosen to be 0.1 and 6000 epochs. The model successfully predicted 5780 out of 6000 with an accuracy: of = 0.964

## K Nearest Neighbours
This was a simple model to code. The implementation of distances function which calculates multiple distances in one go, powered by NumPy's vectorization makes the code run much faster. Although this is a computationally expensive model, Optimization methods have been utilized as much as possible to run it efficiently. The selection of k value was expiremented upon, but finalized to 5. The model successfully predicted 5842 out of 6000 with an accuracy of 0.974.

## Neural Networks
Coding, as well as fine-tuning hyperparameters, were not as straight-forward and easy as other models were upto this point. I was heavily inspired by Tensorflow and hence I wanted to implement this library similar to the interface of tensorflow, hence the dense functions and so on.

While exploring the derivatives of the commonly used functions, I came across the derivative of Categorical crossentropy directly with respect to the input of softmax. The solution of this was very elegant as the final derivative ended up being just (predicted_values - true_values)

Since this neural network had to be implemented for the given Classification dataset, it does not make any sense to have anything other than softmax layer as the final layer and Catergorical Crossentropy as the loss function. It is much faster to compute the simplified gradient of Crossentropy with input of softmax instead of multiplying with softmax derivative.

Another important desicion I had to take during the coding of this model was to choose the type of gradient descent to be implemented. Stochastic gradient has a noisy convergence and Batch gradient was computationally expensive, hence, I decided to go with mini-batch gradient descent, which striked the right balance between convergence and computational complexity.

Now, the last thing to do was to just fine-tune the parameters. I experimented with batches from 16 to 64, and ended with 32. Alpha was decided to be 0.05 and epochs to be just 40, as neural networks tend to overfit very quickly. The model successfully predicted 5885 out of 6000 with an accuracy of 0.981

## K Means Clustering
After having tackled neural networks, this was a far easier program to code. The right value of k was ambiguous to be decided. I have implemented a function to plot a scatter plot of any two features at a time, since visualization is key for K Means Clustering. 

## Normalization
Normalization is a very important thing to do. Normalization was done by default on every model with the option for the user to not implement normalization. Hence, the self.mean = 0 and self.std = 1 will not affect the data if user decides that normalization is not to be implemented.

# GPU Compatibility
During the testing of my models, I was interested to see the models utilize my entire laptop's processing power to train my models as fast as possible. Hence, while running my models, I monitored my CPU and GPU usage and to my surprise, no matter how long the model ran, the model never used my GPU to run it. This felt weird since while running games, my GPU always was being used but not here. 


I did some research on why this was happening and found out that NumPy was not meant to be run on GPUs. From the research I did, CPUs and GPUs compute very differently from eachother. CPUs are well-suited for tasks that require high single-threaded performance and general-purpose computation.
GPUs excel in parallel processing and are highly efficient for tasks that can be parallelized, such as matrix multiplications commonly found in machine learning


While researching this, I also came across CuPy, an alternative to NumPy, but could take use of GPUs. With Cuda toolkit and Cudnn, this library has the potential to make models train at very fast rates although the code remains the same. This was some new knowledge I learned during the Winter of Code, hence the reason I decided to include this in my report. Since Cupy was out of scope for the project, I have implemented matrix multiplications, taking as much advantage I could out of NumPy's vectorization.

# Conclusion
From implementing basic regression models to understanding neural networks and computer architecture, I have learnt a lot during the course of this one month. It was a very fun experience to be a part of and I sincerely thank Cyberlabs for providing me this opportunity to work on this project. 


Thank you!
