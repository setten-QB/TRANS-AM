# TRANS-AM

This repository provides source code for a method, called TRANS-AM.
The method allow us to transform the input vector to satisfy some condition.
If we want to transform the input vector so that a objective variable of a transformed vector will be greater than the given threshold, the TRANS-AM can discover the transformed input vector satisfying such a condition.
The details of the method will be explained in DaWaK 2018.


# How to Use

You can use prediction models built by random forest regressor of the scikit-learn.
When you use the TRANS-AM, you should give the function "transam" in transam.py pickle file of trained random forest regressor, one input vector, threshold of the objective variable, parameter epsilon and cost function which measure the distance between input vectors.
