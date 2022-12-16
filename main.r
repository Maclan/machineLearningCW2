library(readr)
library(tidyverse)
library(palmerpenguins)
library(ggplot2)

source("neuralNetwork.r")
source("evaluateHiddenNeurons.r")
source("evaluateLearningRate.r")
source("evaluateEpochs.r")

set.seed(11)

penguins <- palmerpenguins::penguins

# Drop missing values
penguins <- penguins %>% drop_na()

# split the data, 60% training and 40% validation
trainrows <- sample(1:nrow(penguins), replace = F, size = nrow(penguins)*0.6)
x_train <- penguins[trainrows,]
x_test <- penguins[-trainrows,]

y_train <- model.matrix(~ species - 1, data = x_train)
y_test <- model.matrix(~ species - 1, data = x_test)

# Scale the data
x_train <- scale(x_train[,3:6])
x_test <- scale(x_test[,3:6])

# Convert data to matrix for neural
x_train <- t(as.matrix(x_train, byrow = TRUE))
y_train <- t(as.matrix(y_train, byrow = TRUE))

x_test <- t(as.matrix(x_test, byrow = TRUE))
y_test <- t(as.matrix(y_test, byrow = TRUE))

data <- list("x_train" = x_train,
              "y_train" = y_train,
              "x_test" = x_test,
              "y_test" = y_test)


epochs <- 10000
hidden_neurons <- 6
lr <- 0.02

bestHiddenNeurons <- evaluateHiddenNeurons(data, epochs, lr)
bestEpochs <- evaluateEpochs(data, bestHiddenNeurons, lr)
bestLr <- evaluateLearningRate(data, bestHiddenNeurons, bestEpochs)

cat("\n------ Best Parameters --------\n")
cat("Best Hidden Neurons: ", bestHiddenNeurons, "\n")
cat("Best Epochs: ", bestEpochs, "\n")
cat("Best Learning Rate: ", bestLr, "\n")
cat("-------------------------------\n")

neuralNetwork(data, hidden_neurons , epochs, lr)
# neuralNetwork(data, bestHiddenNeurons, bestEpochs, bestLr)