"""
Coder       -   Dipanshu Verma
Roll number -   B18054
Assignment 1
Question1
"""

# some common imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

#----- functions for use -----------

# returns the log Likelihood of the datapoint wrt to given class
def logLikelihood(mu, cov, x):
    dimensions = len(mu)
    cov_inverse = np.linalg.inv(cov)
    return -0.5*(np.matmul(np.matmul((x-mu).T,cov_inverse),x-mu) + dimensions*np.log(2*np.pi) + np.log(np.linalg.det(cov)))

# returns the number of correctly judged and wrongly judged samples
def predictions(actual_class,classWise_priors, classWise_mu, classWise_cov, test_data):
    number_of_classes = len(classWise_priors)
    correct=0
    wrong=0
    for i in range(len(test_data)):
        g = []
        # calculating logLikelihoods probablities
        for j in range(number_of_classes):
            g.append(logLikelihood(classWise_mu[j], classWise_cov[j], np.array(test_data.iloc[i])))
        # adding log of priors to above terms
        for j in range(number_of_classes):
            g[j]+=np.log(classWise_priors[j])
        pred = g.index(max(g))
        if((pred+1)==actual_class):
            correct+=1
        else:
            wrong+=1
    return correct,wrong

#-----LINEARLY SEPERABLE DATA-------

# describing the file names
class1_file = 'linearly_seperable_data/Class1.txt'
class2_file = 'linearly_seperable_data/Class2.txt'

# reading the file names
class1_data = pd.read_csv(class1_file,sep="\t", usecols=[0,1], names=["attr1","attr2"])
class2_data = pd.read_csv(class2_file,sep="\t", usecols=[0,1], names=["attr1","attr2"])
n1 = len(class1_data)
n2 = len(class2_data)

# shuffling the dataset
class1_data = class1_data.sample(frac=1).reset_index(drop=True)
class2_data = class2_data.sample(frac=1).reset_index(drop=True)

# splitting into training and test data into 70:30 ratio
class1_train = class1_data[:int(n1*0.7)]
class1_test = class1_data[int(n1*0.7):]
class2_train = class2_data[:int(n2*0.7)]
class2_test = class2_data[int(n2*0.7):]

# plotting the 2D scatter plot to visualise
if(input("Do you want to visualize dataset ?(Y/N): ").lower() in ['y','yes']):
    plt.scatter(x=class1_data.attr1,y=class1_data.attr2)
    plt.scatter(x=class2_data.attr1,y=class2_data.attr2)
    plt.show()
if(input("Do you want to visualize train dataset ?(Y/N): ").lower() in ['y','yes']):
    plt.title('train data plot')
    plt.scatter(x=class1_train.attr1,y=class1_train.attr2)
    plt.scatter(x=class2_train.attr1,y=class2_train.attr2)
    plt.show()
if(input("Do you want to visualize test dataset ?(Y/N): ").lower() in ['y','yes']):
    plt.title('test data plot')
    plt.scatter(x=class1_test.attr1,y=class1_test.attr2)
    plt.scatter(x=class2_test.attr1,y=class2_test.attr2)
    plt.show()


# calulation prior probablities given the data
class1_prior = n1/(n1+n2)
class2_prior = 1-class1_prior

# determining the parameters for the classes
class1_mean = np.array(class1_train.mean())
class2_mean = np.array(class2_train.mean())

# determining the covariance matrices of classes
class1_cov = np.array(class1_train.cov())
class2_cov = np.array(class2_train.cov())

# testing using the learned parameters
class1_correct_pred, class1_wrong_pred = predictions(1,[class1_prior,class2_prior],[class1_mean, class2_mean], [class1_cov, class2_cov], class1_test)
class2_correct_pred, class2_wrong_pred = predictions(2,[class1_prior,class2_prior],[class1_mean, class2_mean], [class1_cov, class2_cov], class2_test)

# printing class wise accuracies
print("Accuracy for class1: ", class1_correct_pred/(n1*.3))
print("Accuracy for class2: ", class2_correct_pred/(n2*.3))

# calculating model accuracy
print("Overall Model accuracy: ",(class1_correct_pred+class2_correct_pred)/(n1*0.3+n2*0.3))