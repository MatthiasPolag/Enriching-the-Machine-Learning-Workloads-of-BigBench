#source: https://stackoverflow.com/questions/28017091/will-pandas-dataframe-object-work-with-sklearn-kmeans-clustering
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
import pandas as pd
from matplotlib import style
import time

#start timer
t0= time.time()

#load dataframe
df = pd.read_csv('C:/Users/Matthias/Desktop/clusterData.csv', names=['cid', 'id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11', 'id12', 'id13', 'id14', 'id15', 'id16'])

# Convert DataFrame to matrix
mat = df.values
# Using sklearn
km = sklearn.cluster.KMeans(n_clusters=5)
km.fit(mat)
# Get cluster assignment labels
labels = km.labels_
# Format results as a DataFrame
results = pd.DataFrame(data=labels, columns=['cluster'])

#stop timer
t1 = time.time() 
print("Time elapsed: ", t1- t0)
