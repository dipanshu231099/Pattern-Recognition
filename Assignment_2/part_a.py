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

def distanceFinder(p1, p2):
    return ((p2-p1)**2).sum()**0.5

def KMeansClustering(dataset, k):
    n = len(dataset)
    MAX = (np.array(dataset.max())[1:])
    MIN = (np.array(dataset.min())[1:])
    RANGE = MAX-MIN
    dimension = dataset.shape[1]-1
    means=[]
    for i in range(k):
        means.append(MIN+np.random.random(dimension)*RANGE)
    
    n_iter=0
    COSTS=[]
    while(True):
        n_iter+=1
        costThisIteration=0
        print("Doing iteration number {} ...".format(n_iter))

        assign=[]
        for i in range(len(dataset)): # calculation per point of distances and thus their assignment
            p1 = np.array(dataset.iloc[i])[1:]
            distances=[]
            for j in means:
                distances.append(distanceFinder(p1,j))
            assign.append(distances.index(min(distances))*1.0)
            costThisIteration+=min(distances)
        
        COSTS.append(costThisIteration) # adding the cost this iteration

        for j in range(k): # assigning new means
            new_mean = np.array([0.0, 0.0])
            cnt=0
            for i in range(len(dataset)):
                if(assign[i]==j):
                    new_mean+=np.array(dataset.iloc[i])[1:]
                    cnt+=1
            new_mean/=cnt
            means[j] = new_mean
        
        if(n_iter>1 and (round(COSTS[-1],6) == round(COSTS[-2],6))): # breaking if cost is not changing much
            print("KMeans completed. Minimum Cost achieved {}... ".format(COSTS[-1]))
            break

    prompt=input("Do you wish to see {} dataset(y/n): ".format("clustered"))
    if(prompt.lower() in ['y', 'yes']):
        plt.scatter(dataset[0],dataset[1],c=assign)
        plt.show()

    return means


#----------------- DRIVER CODE ------------------

# obtaining main data
path_class1 = "/home/ninja/data/IIT-MANDI/semester-5/pattern/Assignment_2/non_linearly_seperable data/Class1.txt"
path_class2 = "/home/ninja/data/IIT-MANDI/semester-5/pattern/Assignment_2/non_linearly_seperable data/Class2.txt"

data_class1 = dataCraete(path_class1)
data_class2 = dataCraete(path_class2)

# splitting the data randomly
train1,test1 = trainTestSplit(data_class1)
train2,test2 = trainTestSplit(data_class2)
joinedTestData = pd.concat([test1, test2])

# scatter plots

scatterPlot([data_class1, data_class2], "original")
scatterPlot([train1, train2], "training")
scatterPlot([test1, test2], "test")

# Aplying KMeans algo
K = int(input("Enter the number of clusters requried: "))
kmeans1=KMeansClustering(train1, K)
kmeans2=KMeansClustering(train2, K)
plt.show()

# Testing for confusion matrix
n1 = len(test1)
n2 = len(test2)
labels = [0 for i in range(n1)]
labels.extend([1 for i in range(n2)])
kmeansJoined = kmeans1+[]
kmeansJoined.extend(kmeans2)
confusionMatrix = np.zeros((2,2))
predictions=[]
for i in range(len(labels)):
    p1 = np.array(joinedTestData.iloc[i])[1:]
    distances = []
    for m in range(len(kmeansJoined)):
        distances.append(distanceFinder(p1, kmeansJoined[m]))
    prediction = int(distances.index(min(distances))/K)
    predictions.append(prediction)
    confusionMatrix[predictions[i]][labels[i]]+=1
print("Printing the confusion matrix...\n",confusionMatrix)

# calculating accuracy from above computed confusion matrix
correctPredictions = 0
totalPredictions = 0
for i in range(confusionMatrix.shape[0]):
    for j in range(confusionMatrix.shape[1]):
        if(i==j):
            correctPredictions+=confusionMatrix[i][j]
        totalPredictions+=confusionMatrix[i][j]
print("Accuracy obtained...",correctPredictions/totalPredictions)