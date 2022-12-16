source("neuralNetwork.r")

evaluateHiddenNeurons <- function(data, epochs, lr){
  
  best_accuracy <- 0
  best_neurons <- 0
  accuracy <- matrix(ncol = 2, nrow = 19)
  costs <- c()
  hidden_neurons_list <- c()

  for (i in 1:10){
      hidden_neurons <- i
      train_model <- train(data$x_train, data$y_train, epochs, hidden_neurons , lr)
      y_pred <- predict(data$x_test, data$y_test, hidden_neurons, train_model)
      y_pred <- round(y_pred)

      caculateAccuracy <- mean(y_pred == data$y_test)
      cat("Neurons: ", hidden_neurons, "")
      cat("Accuracy: ", caculateAccuracy, "%\n")

      accuracy <- rbind(accuracy, c(hidden_neurons, caculateAccuracy))
      costs <- c(costs, train_model$cost)
      hidden_neurons_list <- c(hidden_neurons_list, hidden_neurons)

      if (caculateAccuracy > best_accuracy){
          best_accuracy <- caculateAccuracy
          best_neurons <- hidden_neurons
      }
  }

  cat("Best Neurons: ", best_neurons, "\n")
  cat("Best Accuracy: ", best_accuracy, "%\n")

  plot(accuracy[,1], accuracy[,2],type = "b",xlab = "Hidden Neurons", ylab = "Accuracy",main = "Accuracy vs. Neurons")
  plot(hidden_neurons_list, costs, type = "b", xlab = "Hidden Neurons", ylab = "MSE", main = "MSE vs. Hidden Neurons")

  return (best_neurons)
}