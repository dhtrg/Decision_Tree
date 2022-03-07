library(rpart)
library(rpart.plot)
library(plyr)
library(caTools)


#the original dataset has column names in a different file, so I needed to add the column names with my code
cols = list('Sex','Length','Diameter','Height','Whole.weight','Shucked.weight','Viscera.weight','Shell.weight','Rings')

# read data
m.data = read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",sep=",",header= FALSE,col.names=cols)


set.seed(615)

m.data$Rings <- cut(m.data$Rings, breaks = c(0,4,8,12,16,20,24,Inf),labels = c('[1-4]','[5-8]','[9-12]','[13-16]','[17-20]','[21-24]','[25-29]'))

# split data into training and tests sets
#SplitRatio = 5/6 =0.8333 because I use 6 fold cross validation
sample_data = sample.split(m.data, SplitRatio = 0.8333)

train_data = subset(m.data, sample_data==TRUE)
test_data = subset(m.data, sample_data==FALSE)

# build the tree
dtree = rpart(Rings~Sex+Diameter+Height+Whole.weight+Shucked.weight+Viscera.weight+Shell.weight, train_data)

#visualize the tree
rpart.plot::rpart.plot(dtree)

# predict the test set
predicted <- predict(dtree, test_data, type='class')

# table method to create the contigency/confusion marix
conf_mat = table(test_data$Rings, predicted)

# calculate acccuracy, etc.
accuracy = sum(diag(conf_mat))/sum(conf_mat)

# split data into kFolds; k=6

foldCount = 6

folds <- split(m.data, cut(sample(1:nrow(m.data)),foldCount))

errs <- rep(NA, length(folds))

for (i in 1:length(folds)) {
  # get the test set
  test_data_1 <- ldply(folds[i], data.frame)
  # get the training set
  train_data_1 <- ldply(folds[-i], data.frame)
  
  # build the tree
  tmp.model <- rpart(Rings~Sex+Length+Diameter+Height+Whole.weight+Shucked.weight+Viscera.weight+Shell.weight, train_data_1, method = "class")
  # test the model
  test.predict <- predict(tmp.model, newdata = test_data_1, type = "class")
  # create the confusion matrix
  conf.mat <- table(test_data_1$Rings, test.predict)
  # calculate the errors
  errs[i] <- 1-sum(diag(conf.mat))/sum(conf.mat)
  
}

# calculate the average error
print("Measures of accuracy/goodness of the models")
print(paste("Root mean squared error for 6 fold cross validation: ", mean(errs)))
print("Confusion matrix of the model:")
print(conf_mat)
print(paste("Accuracy: ", accuracy))
