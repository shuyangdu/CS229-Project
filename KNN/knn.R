library(dplyr)
library(class)
library(layout)
library(ISLR)
library(caret)
library(knnGarden)
library(zoo) # fillNAs function

## Data Processing

# rm(list=ls())
setwd('~/Amazon Drive/Stanford CS229/Project/')
data <- read.csv('Web Traffic Data/train_2.csv', header = TRUE, row.names = 1)
ds <- names(data) %>% 
  sapply(function(x) gsub('\\.', '-', x) %>% substr(2, 11)) %>% 
  t %>% 
  as.vector

data_sample <- data[1025:2048,]

# transform data and normalize

transformer <- function(y) {
  x <- log(y + 1)
  return(x - rowMeans(x))
}

inverse_transformer <- function(y, orig_mean) {
  x <- y + orig_mean
  return(exp(x) - 1)
}

## Visualize data 

s1 <- t(data_sample[sample(nrow(data_sample), 1), ]) %>% as.data.frame

ggplot(data = s1, aes(x = as.Date(ds), y = as.vector(s1)))+ 
  geom_line() +
  labs(title = names(s1), x = "Date", y = "Web Traffic Hits") +
  scale_y_continuous()

## KNN implementation

# transform data
data_sample_norm <- transformer(data_sample)

# split data
testSize <- 64
trainSet <- data_sample_norm[,1:(ncol(data_sample_norm)-testSize)] %>% t
testSet <- data_sample_norm[,(ncol(data_sample_norm)-testSize+1):ncol(data_sample_norm)] %>% t 

# get rid of NAs
trainSet_NNA <- na.locf(trainSet)
testSet_NNA <- na.locf(testSet)

trainLabels <- colnames(trainSet_NNA) %>% as.data.frame

# different distance metrics
distance <- c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")
kdist <- dist(data_sample_norm, method = "euclidean")
kdist_manhattan <- dist(data_sample_norm, method = "manhattan")
kdist_minkowski <- dist(data_sample_norm, method = "minkowski")
kdist_canberra <- dist(data_sample_norm, method = "canberra")
kdist_maximum <- dist(data_sample_norm, method = "maximum")
kdist_binary <- dist(data_sample_norm, method = "binary")

# assign different k-values
k <- c(1, 5, 10, 25, 50, 100, 200)

# to get repeatable results
# set.seed(999)

cl <- trainLabels[,1]
cl_NNA <- na.locf(cl)

# dim(trainSet_NNA)
# dim(testSet_NNA)
# length(cl_NNA)

# gives the correct prediction of the test and training sets
for (i in 1:length(distance)) {
  for (j in 1:length(k)) {
    y_pred[i][j] <- knnVCN(trainSet_NNA, cl_NNA, testSet_NNA, K = k[j], method = distance[i])
  }
}

# shows the confusion matrix between test values and predicted values

for (i in 1:length(distance)) {
  for (j in 1:length(k)) {
    table(testSet_NNA,y_pred[i][j])
  }
}

## Accuracy

smape <- function(y_true, y_pred) {
  return(mean((2 * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))) * 100))
}

## Graphs and Summary

for (i in 1:length(distance)) {
  for (j in 1:length(k)) {
    smape_arr[i][j] <- smape(testSet_NNA, y_pred[i][j])
    print(paste0("SMAPE: ", smape_arr[i][j], 
                 ", K = ", k[j], ", Distance: ", distance(i)))
  }
}

# optimal result
# max(smape_arr)

# transform back
y_pred_un <- inverse_transformer(y_pred, mean(rowMeans(data_sample)))

y_pred_opt <- y_pred_un[3][4] # k = 10; canberra distance

## Plotting out the results

s2 <- t(data_sample[sample(nrow(data_sample), 1), ]) %>% as.data.frame

s2 <- 
  ggplot(data = s1, aes(x = as.Date(ds), y = as.vector(s1)))+ 
  geom_line() +
  labs(title = names(s1), x = "Date", y = "Web Traffic Hits") + 
  geom_line(color='red',data = y_pred_opt, aes(x=ds, y=y_pred_opt))
