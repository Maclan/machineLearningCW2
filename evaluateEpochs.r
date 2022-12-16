source("neuralNetwork.r")

evaluateEpochs <- function(data, hidden_neurons, lr){
  
  best_accuracy <- 0
  best_epochs <- 0
  accuracy <- matrix(ncol = 2, nrow = 19)
  costs <- c()
  epochs_list <- c()

  for (i in 1:10){
      epochs <- i * 1000
      train_model <- train(data$x_train, data$y_train, epochs, hidden_neurons , lr)
      y_pred <- predict(data$x_test, data$y_test, hidden_neurons, train_model)
      y_pred <- round(y_pred)
      caculateAccuracy <- mean(y_pred == data$y_test) * 100
      cat("Epochs: ", epochs, "")
      cat("Accuracy: ", caculateAccuracy, "%\n")

      accuracy <- rbind(accuracy, c(epochs, caculateAccuracy))
      costs <- c(costs, train_model$cost)
      epochs_list <- c(epochs_list, epochs)

      if (caculateAccuracy > best_accuracy){
          best_accuracy <- caculateAccuracy
          best_epochs <- epochs
      }
  }

  cat("Best Epochs: ", best_epochs, "\n")
  cat("Best Accuracy: ", best_accuracy, "%\n")

  plot(accuracy[,1], accuracy[,2],type = "b",xlab = "Epochs", ylab = "Accuracy",main = "Accuracy vs. Epochs")
  plot(epochs_list, costs, type = "b", xlab = "Epochs", ylab = "MSE", main = "MSE vs. Epochs")

  return (best_epochs)
}