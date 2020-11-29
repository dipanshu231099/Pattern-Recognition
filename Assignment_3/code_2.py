import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#------------------ FUNCTIONS -------------------

# Creating a dataset as a numpy array
def dataCraete(filepath):
    data=pd.read_csv(filepath, sep='\t', header=None, usecols=[0,1])
    return data

def scatterPlot(dataSets, strType):
    prompt=input("Do you wish to see {} dataset(y/n): ".format(strType))
    if(prompt.lower() not in ['y', 'yes']):
        return
    for i in range(len(dataSets)):
        data_class = dataSets[i]
        plt.scatter(data_class[0], data_class[1], label="class"+str(i+1))
    plt.legend()
    plt.show()

def trainTestSplit(dataset):
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1).reset_index(drop=False)
    n = int(len(dataset)*0.7)
    trainData = dataset[:n]
    testData = dataset[n:]
    return trainData,testData

def gaussianPDF(mu, cov, x):
    d = len(mu)
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    exp = np.exp((-0.5)*np.matmul(np.matmul(x-mu, inv),x-mu))
    return 1/( ((2*np.pi)**.5) *(det**(d/2) )) *  exp

# calulates the log likelihood of data
def calculateLogLikelihood(dataset, means, covariances, weights):
    K = len(means)
    LogLikelihood=0
    for i in range(len(dataset)):
        x = np.array(dataset.iloc[i])[1:]
        tmp=0
        for k in range(K):
            tmp+=weights[k]*gaussianPDF(means[k], covariances[k], x)
        LogLikelihood+= np.log(tmp)
    return LogLikelihood


# given a class data will return the responsiblities of each of the k clusters and their means and covariance
def GMM(dataset, n_clusters ):
    print("Performing GMM...")
    MAX = (np.array(dataset.max())[1:])
    MIN = (np.array(dataset.min())[1:])
    RANGE = MAX-MIN
    # intialising the means and covariances
    dimensions = dataset.shape[1]-1
    means=[]
    covariances=[]
    weights=[]
    for i in range(n_clusters):
        means.append(MIN+RANGE*np.random.random(size=dimensions))
        covariances.append(np.identity(dimensions))
        weights.append(1/n_clusters)
    
    log_likelihood = calculateLogLikelihood(dataset, means, covariances, weights)

    responsiblities=[[0 for i in range(n_clusters)] for j in range(len(dataset))] # intialising the responsiblities

    iter=1
    while(True):
        # performing E-Step
        print("Doing iteration",iter,log_likelihood)
        effective_n = [0 for i in range(n_clusters)]
        for i in range(len(dataset)):
            x = np.array(dataset.iloc[i])[1:]
            numerator=[]
            for k in range(n_clusters):
                mu = means[k]
                cov = covariances[k]
                w = weights[k]
                numerator.append(w*gaussianPDF(mu,cov,x))
            evidence = sum(numerator)
            gamma = [numerator[k]/evidence for k in range(n_clusters) ]
            for k in range(n_clusters):
                effective_n[k]+=gamma[k]
            responsiblities[i] = gamma

        # performing M step
        for k in range(n_clusters):
            means[k] = np.zeros(n_clusters)
            covariances[k]=np.zeros(covariances[k].shape)
            #calculating new means
            for i in range(len(dataset)):
                x = np.array(dataset.iloc[i])[1:]
                means[k]+=responsiblities[i][k]*np.array(dataset.iloc[i])[1:]
            means[k]/=effective_n[k]

            #recalculating covs
            for i in range(len(dataset)):
                x = np.array(dataset.iloc[i])[1:]
                bb = np.array([x-means[k]])
                covariances[k]+=responsiblities[i][k]*np.matmul(bb.T,bb)
            covariances[k]/=effective_n[k]
            
            # recalculating weights
            weights[k] = effective_n[k]/len(dataset)
        
        previous_LL = log_likelihood
        log_likelihood = calculateLogLikelihood(dataset, means, covariances, weights)
        if(round(log_likelihood,4)==round(previous_LL,4)):
            break
        iter+=1

    return means, covariances, weights



#---------------- MAIN CODE ---------------------

# reading the data
D1 = dataCraete('data_2/Class1.txt')
D2 = dataCraete('data_2/Class2.txt')

# splitting the dataset
train1,test1 = trainTestSplit(D1)
train2,test2 = trainTestSplit(D2)

# scatter plots
scatterPlot([D1, D2], "original")
scatterPlot([train1, train2], "training")
scatterPlot([test1, test2], "test")

# Applying the GMM
K = 2
means_c1, covariances_c1, weights_c1 = GMM(train1, K)
print(means_c1,'\n', covariances_c1,'\n', weights_c1)
means_c2, covariances_c2, weights_c2 = GMM(train2, K)
print(means_c2,'\n', covariances_c2,'\n', weights_c2)

# plotting the new means
if(input("Do you want to see the new means ? (y/n) ").lower()=='y'):
    plt.scatter(D1[0],D1[1])
    plt.scatter(means_c1[0][0],means_c1[0][1],color='y')
    plt.scatter(means_c1[1][0],means_c1[1][1],color='y')
    plt.show()

    plt.scatter(D2[0],D2[1])
    plt.scatter(means_c2[0][0],means_c2[0][1],color='y')
    plt.scatter(means_c2[1][0],means_c2[1][1],color='y')
    plt.show()

# predicting on test data
confusion_matrix = np.zeros((2,2))
for i in range(len(test1)):
    x = np.array(test1.iloc[i])[1:]
    p1 = sum([weights_c1[k]*gaussianPDF(means_c1[k],covariances_c1[k],x) for k in range(K)])
    p2 = sum([weights_c2[k]*gaussianPDF(means_c2[k],covariances_c2[k],x) for k in range(K)])
    if(p1>p2):
        confusion_matrix[0][0]+=1
    else:
        confusion_matrix[0][1]+=1

for i in range(len(test2)):
    x = np.array(test2.iloc[i])[1:]
    p1 = sum([weights_c1[k]*gaussianPDF(means_c1[k],covariances_c1[k],x) for k in range(K)])
    p2 = sum([weights_c2[k]*gaussianPDF(means_c2[k],covariances_c2[k],x) for k in range(K)])
    if(p1>p2):
        confusion_matrix[1][0]+=1
    else:
        confusion_matrix[1][1]+=1

print("Confusion Matrix:\n",confusion_matrix)

# calculating the accuracy
print("Accuracy:",100*(confusion_matrix[0][0]+confusion_matrix[1][1])/sum(sum(confusion_matrix)))
