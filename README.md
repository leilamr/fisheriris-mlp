# Dataset Iris plant
Iris flower classification with MLP using MATLAB.

#### Atribute informations
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class: Iris Setosa, Iris Versicolour and Iris Virginica.

#### Class coding
* Setosa = [1 0 0 ]
* Versicolor = [0 1 0]
* Virginica = [0 0 1]

#### Network settings: 
```
Train = 70%, Validation = 15% and Testing = 15%
Number hidden of nodes = 4 
Epochs = 1000
Trainng Function = trainlm
Transfer Function (layer 1) = tansig
Trasnfer Function (layer 2) = purelin

Accuracy = 99.3%
``` 

![Alt text](https://github.com/leilamr/fisheriris-mlp/blob/master/all-confusion-matrix.jpg?raw=true?raw=true "Matrix Confusion")


#### Cross-validation: k-fold 
In fisherIris_mpl_kfold.m the dataset was divided into 10 folds. Each k-folds has size 15x5. 

The best configuration obtained from the network with the cross validation technique was:

```
Number hidden of nodes = 4 
Epochs = 1000
Trainng Function = trainlm
Transfer Function (layer 1) = tansig
Trasnfer Function (layer 2) = purelin

-- Average accuracy = 94.667%
```

#### References
https://la.mathworks.com/help/deeplearning/gs/classify-patterns-with-a-neural-network.html
https://la.mathworks.com/help/deeplearning/ref/plotconfusion.html
https://la.mathworks.com/help/deeplearning/ref/patternnet.html
