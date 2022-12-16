source("neuralNetwork.r")

evaluateLearningRate <- function(data, hidden_neurons, epochs){
  
  best_accuracy <- 0
  best_lr <- 0
  accuracy <- matrix(ncol = 2, nrow = 19)
  learningRate <- c(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
  costs <- c()
  learning_rate_list <- c()

  for (i in 1:10){
      lr <- learningRate[i]
      train_model <- train(data$x_train, data$y_train, epochs, hidden_neurons , lr)
      y_pred <- predict(data$x_test, data$y_test, hidden_neurons, train_model)
      y_pred <- round(y_pred)

      caculateAccuracy <- mean(y_pred == data$y_test) * 100
      cat("Learning Rate: ", lr, "\n")
      cat("Accuracy: ", caculateAccuracy, "%\n")

      accuracy <- rbind(accuracy, c(lr, caculateAccuracy))
      costs <- c(costs, train_model$cost)
      learning_rate_list <- c(learning_rate_list, lr)

      if (caculateAccuracy > best_accuracy){
          best_accuracy <- caculateAccuracy
          best_lr <- lr
      }
  }

  cat("Best Learning Rate: ", best_lr, "\n")
  cat("Best Accuracy: ", best_accuracy, "%\n")

  plot(accuracy[,1], accuracy[,2],type = "b",xlab = "Learning Rate", ylab = "Accuracy",main = "Accuracy vs. Learning Rate")
  plot(learning_rate_list, costs, type = "b", xlab = "Learning Rate", ylab = "MSE", main = "MSE vs. Learning Rate")

  return (best_lr)
}