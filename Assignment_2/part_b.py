import numpy as np
import matplotlib.pyplot as plt
import cv2

#-----------

def distanceFinder(p1, p2):
    return ((p2-p1)**2).sum()**0.5

def KMeansClustering(dataset, k):
    n = len(dataset)
    dimension = len(dataset[0])
    means=[]
    for i in range(k):
        means.append(np.random.random(dimension))
    
    n_iter=0
    COSTS=[]
    while(True):
        if(n_iter>1 and ((COSTS[-2]-COSTS[-1])/COSTS[-2])<0.001): # breaking if cost is not changing much
            print("KMeans completed. Minimum Cost achieved {}... ".format(COSTS[-1]))
            break

        n_iter+=1
        costThisIteration=0
        print("Doing iteration number {} ...".format(n_iter));

        assign=[]
        for i in range(len(dataset)): # calculation per point of distances and thus their assignment
            p1 = np.array(dataset[i])/256
            distances=[]
            for j in means:
                distances.append(distanceFinder(p1,j))
            assign.append(distances.index(min(distances)))
            costThisIteration+=min(distances)
        
        COSTS.append(costThisIteration) # adding the cost this iteration

        for j in range(k): # assigning new means
            new_mean = np.zeros(dimension)
            cnt=0
            for i in range(len(dataset)):
                if(assign[i]==j):
                    new_mean+=np.array(dataset[i])/256
                    cnt+=1
            new_mean/=cnt
            means[j] = new_mean

    return means,assign

#-----------

img = "/home/ninja/data/IIT-MANDI/semester-5/pattern/Assignment_2/Image.jpg"
img = np.array(cv2.imread(img))

imageResolution = img.shape
image = cv2.resize(img  , (256 , 256*imageResolution[0]//imageResolution[1]))
cv2.imwrite("original.jpg",image)
imageResolution = image.shape

# preprocessing the image data
data_no_location=[]
data_with_loc=[]
for i in range(imageResolution[0]):
    for j in range(imageResolution[1]):
        data_no_location.append([image[i][j][0],image[i][j][1],image[i][j][2]])
        data_with_loc.append([image[i][j][0],image[i][j][1],image[i][j][2],i,j])


means, assign = KMeansClustering(data_no_location, 2)
for i in range(imageResolution[0]):
    for j in range(imageResolution[1]):
        image[i][j][0]=means[assign[i*imageResolution[1]+j]][0]*256
        image[i][j][1]=means[assign[i*imageResolution[1]+j]][1]*256
        image[i][j][2]=means[assign[i*imageResolution[1]+j]][2]*256
cv2.imwrite("segmented_only_rgb.jpg", image)

means, assign = KMeansClustering(data_with_loc, 2)
for i in range(imageResolution[0]):
    for j in range(imageResolution[1]):
        image[i][j][0]=means[assign[i*imageResolution[1]+j]][0]*256
        image[i][j][1]=means[assign[i*imageResolution[1]+j]][1]*256
        image[i][j][2]=means[assign[i*imageResolution[1]+j]][2]*256
cv2.imwrite("segmented_with_loc.jpg", image)
