# Project 6

In this project, we used traffic data from Minneapolis to try and find anomalies. We chose to investigate the travel time data and were attempting to identify anomalies in this data using two methods.

One of the main dataset modifcation we made was to the timestamp variable. To perform anomaly detection we must have a numerical x value. To preserve the timestamp data we want to order the data by their timestamp and create a variable that measures how many minutes since the first entry the entry is.

## Anomaly Detection Method
We attempted to run two different anomaly detection methods: local outlier factor (LOF) and isolation forest.

### Local Outlier Factor
For the Local Outlier Factor model, we train the model on our data and then plot our points to see the results. The LOF is a number that represents how much of an outlier a specific datapoint is relative to the other data. We see the results of this by plotting red circles around the points with radius size relative to the LOF.

### Isolation Forest

For the Isolation Forest, this model measures the outliers in a little bit of a different way. This classifies each point as either an outlier or an inlier, instead of measuring it on a spectrum like LOF does. We plot the datapoints to see which ones are classified as outliers. 
