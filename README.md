## Revisiting an algorithm: KNN
The goal of this is to revisit an old algorithm that I had learnt in undergrad in my current language of choice (Go). KNN (K nearest neighbors) is a lazy algorithm to classify (i.e. match by closeness) input to one of the types in the training set.

[Source](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
[Additional reading](https://machinelearningmastery.com/k-fold-cross-validation/)

What I appreciated about Go while writing this (vs Python):
- I know what (type) each function is going to return. For instance the `crossValidationSplit` function; while reading the python example it was hard for me to figure what will be the structure type i.e. the return type for the function.
- Python passing-by is object reference for mutable types (like lists) but by copy (or reference value) for immutable types (like strings), it takes a bit to track that down. In Go its fairly clear - everything is by value, for sharing memory we share pointers.
