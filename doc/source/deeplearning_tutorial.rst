.. Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
.. in Cooperation with Max Planck Society

.. SPDX-License-Identifier: CC-BY-NC-SA-1.0

Deep Learning Tutorial
----------------------

.. note::
    These examples were run on the ESI HPC cluster. This is why we have to use the `esi_cluster_setup` function to set up the cluster.
    They are perfectly reproducable on any other cluster or local machine by using the `local_cluster_setup` or `slurm_cluster_setup` function instead.

The following Python code demonstrates how to use ACME to perform parallel deep learning model fitting with `PyTorch <https://pytorch.org/>`_ to evaluate the best model for a dataset.
This is a somewhat toy example, in which we will vary the model architecture randomly. Nevertheless, this general approach can be used to perform a grid search over a set of parameters.
This problem is inspired by some fantastic DeepLearning course from `Mike X. Cohen <https://www.mikexcohen.com/>`_.

First, we import the necessary packages:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,TensorDataset
    from sklearn.model_selection import train_test_split
    import numpy as np
    import scipy.stats as stats
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython import display
    from acme import cluster_cleanup, esi_cluster_setup,  ParallelMap
    import itertools


Building the network & related functions
-----------------------------------------

Next, we define our NeuralNet model. We will use a simple fully connected network with 5 hidden layers and vary the number of units in each layer.
The goal of the model is to binary classify if a wine was rated good or bad. This classification is based on some chemical compound analysis of the wine.

We define our model as follows:

.. code-block:: python

    class NeuralNet(nn.Module):
        def __init__(self,param):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.layers.update({'input':nn.Linear(11,16)})
            for ilay,layer in enumerate(param):
                if ilay==0:
                    self.layers['l'+str(ilay)] = nn.Linear(16,layer)
                else:
                    self.layers['l'+str(ilay)] = nn.Linear(param[ilay-1],layer)
            self.layers.update({'output':nn.Linear(param[ilay],1)})

        def forward(self,x,param):
            x = F.relu( self.layers['input'](x))
            for ilay in range(len(param)):
                x = F.relu( self.layers['l'+str(ilay)](x))
            return self.layers['output'](x)


We can observe that we construct the number of units in each layer based some input into the model.

We need to define a function that trains the network:

.. code-block:: python

    def trainTheModel(model,nepochs,trainLoader,testLoader,param):
        lossfun   = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=.01)
        losses    = torch.zeros(nepochs)
        trainAcc  = []
        testAcc   = []
        # loop over epochs
        for epochi in range(nepochs) :
            model.train() # put the model into training mode is not necessary here, but good practice
            batchAcc  = []
            batchLoss = []
            for X,y in trainLoader:
            yHat = model.forward(X,param)
            loss = lossfun(yHat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchLoss.append(loss.item())
            batchAcc.append( 100*torch.mean(((yHat>0) == y).float()).item() )

            trainAcc.append( np.mean(batchAcc) )
            losses[epochi] = np.mean(batchLoss)

            model.eval()
            X,y = next(iter(testLoader))
            with torch.no_grad(): # deactivates autograd
            yHat = model.forward(X,param)
            testAcc.append( 100*torch.mean(((yHat>0) == y).float()).item() )
        return trainAcc,testAcc,losses


    def parallel_model_eval(param,trainLoader,testLoader,nepochs=500):
        # this function is called by the parallel map function
        model = NeuralNet(param)
        trainAcc,testAcc,losses = trainTheModel(model=model,nepochs=nepochs,trainLoader=trainLoader,testLoader=testLoader,param=param)
        return trainAcc,testAcc,losses


The second function `parallel_model_eval` is later called by the :class:`~acme.ParallelMap` class. Within `parallel_model_eval`, we first build our model based on the
parameters and then train and evaluate the model. The function returns the training and test accuracy as well as the loss function over the epochs.
It is also possible that ACME return the model itself, since it is pickable. However, this is not necessary here.


Getting the data ready
-----------------------
We will pass the PyTorch dataloaders along with the model parameters to the :class:`~acme.ParallelMap` class.

.. code-block:: python

    url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url,sep=';')
    data = data[data['total sulfur dioxide']<200] # drop a few outliers

    # z-score all columns except for quality
    norm_cols = data.keys().drop('quality')
    data[norm_cols ] = data[norm_cols].apply(stats.zscore)

    # create a new column for binarized (boolean) quality
    data['binqual'] = data.apply(lambda x: 1 if x['quality']>5 else 0,axis=1)

    X_train, X_test, y_train, y_test = train_test_split(torch.tensor( data[norm_cols].values ).float(),\
        torch.tensor( data['binqual'].values ).float()[:,None], test_size=.1)

    # then convert them into PyTorch Datasets (note: already converted to tensors)
    trainLoader = DataLoader(TensorDataset(X_train,y_train),batch_size=32,shuffle=True)
    testLoader  = DataLoader(TensorDataset(X_test,y_test),batch_size=X_test.shape[0],shuffle=True)


Here we generate the inputs to our parallel function. We vary the number of units for each layer as powers of 2 from 16 to 512 and use all possible permuations of this set.

.. code-block:: python

    # Prepare inputs for parallelization
    params = list(itertools.permutations([2**i for i in range(4,10)]))

    # set up client
    client = esi_cluster_setup(partition="8GBS",n_workers=200)

    # compute
    with ParallelMap(parallel_model_eval, params, trainLoader, testLoader, n_inputs=len(params), write_worker_results=False) as pmap:
        results = pmap.compute()

NOTE: In this example we do not write the results to disk, because `write_worker_results=False`. If we want to save the models however, or if the output becomes larger,
it is highly recommended to save to disk and not collect in local memory.

After the computation is done, we can inspect the different outcome parameters that were returned:
- test set accuracy time courses (as a function of epochs)
- train set accuracy time courses
- losses

.. code-block:: python

    for i, param in enumerate(params):
        trainAcc,testAcc,losses = results[i]
        plt.plot(testAcc,label=str(param))
    plt.legend()

Which model performed best over the last 50 epochs?

.. code-block:: python

    bestModel = np.argmax([np.mean(model[0][-50:]) for model in results])
