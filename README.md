This project is a huble beginning in the ML world. It's goal was to get familiar with one of the pillars of ML which is backpropagation using gradient descend (currently it's stochastic). The imposed conditions while developing it were:
1. No ML libraries
2. Don't look at any type of code
3. Only theoretical materials were allowed

Here you won't see "beautiful" or "optimized and performant" code. This was out of scope.

How to run:
1. Extract the mnist_db.zip in RunNetwork\Data and set them to be copied to output dir. (The test and training data are from the MINST databse. They contain handwritten digits with labels. The images are grayscaled 28 by 28 pixels. They are represented as arrays of integers where the first index is the label)
2. Define desired network structure (There is commented out example in Program.cs. The given structure + learning rate for 1000 epochs results in 85-87% accuracy on unseen test data)
3. Load the training data and train.
4. Load the testing data and test :)

There is an option to store a trained model and reload it for future training/testing.
