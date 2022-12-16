
layerSize <- function(X, y, hidden_neurons, train=TRUE) {
    return(list("n_x" = dim(X)[1],
                "n_h" = hidden_neurons,
                "n_y" = dim(y)[1]))
}

setParameters <- function(X, layer_size){
  n_x <- layer_size$n_x
  n_h <- layer_size$n_h
  n_y <- layer_size$n_y

  return (list("W1" = matrix(runif(n_h * n_x), nrow = n_h, ncol = n_x, byrow = TRUE) * 0.01,
                  "b1" = matrix(rep(0, n_h), nrow = n_h),
                  "W2" = matrix(runif(n_y * n_h), nrow = n_y, ncol = n_h, byrow = TRUE) * 0.01,
                  "b2" = matrix(rep(0, n_y), nrow = n_y)))
}

# Activation function
sigmoid <- function(x){
    return(1 / (1 + exp(-x)))
}

# Backward propagation
backwardPropagation <- function(X, y, output, params, layer_size){

    m <- dim(X)[2]
    n_h <- layer_size$n_h
    n_y <- layer_size$n_y

    dZ2 <- output$A2 - y
    dZ1 <- (t(params$W2) %*% dZ2) * (1 - output$A1^2)

    return(list("dW1" = 1/m * (dZ1 %*% t(X)),
                  "db1" =  matrix(1/m * sum(dZ1), nrow = n_h),
                  "dW2" = 1/m * (dZ2 %*% t(output$A1)),
                  "db2" = matrix(1/m * sum(dZ2), nrow = n_y)
                  ))
}

# Forward propagation
forwardPropagation <- function(X, params){

    Z1 <- params$W1 %*% X
    A1 <- sigmoid(Z1)
    Z2 <- params$W2 %*% A1
    A2 <- sigmoid(Z2) # output (Activation function)

    return (list("Z1" = Z1,
                  "A1" = A1,
                  "Z2" = Z2,
                  "A2" = A2))
}

# MSE function
cost <- function(X, y, output) {
    return (1/nrow(y)*sum((y-output$A2)^2))
}

# Update parameters
updateParameters <- function(grads, params, learning_rate){
    return (list("W1" = params$W1 - learning_rate * grads$dW1,
                 "b1" = params$b1 - learning_rate * grads$db1,
                 "W2" = params$W2 - learning_rate * grads$dW2,
                 "b2" = params$b2 - learning_rate * grads$db2) )
}

train <- function(X, y, epochs, hidden_neurons, lr){

  layer_size <- layerSize(X, y, hidden_neurons)
  init_params <- setParameters(X, layer_size)
  cost <- 0

  for (i in 1:epochs) {
      forwardProp <- forwardPropagation(X, init_params)
      backProp <- backwardPropagation(X, y, forwardProp, init_params, layer_size)

      updateParams <- updateParameters(backProp, init_params, lr)
      init_params <- updateParams

      cost <- cost(X, y, forwardProp)
  }

  cat("MSE: ", cost, "\n")
  model <- list("updated_params" = updateParams, "cost" = cost)
  return (model)
}

predict <- function(X, y, hidden_neurons, train_model){
  params <- train_model$updated_params
  forwardProp <- forwardPropagation(X, params)

  return (forwardProp$A2)
}

neuralNetwork <- function(data, hidden_neurons, epochs, lr){
  train_model <- train(data$x_train, data$y_train, epochs, hidden_neurons, lr)
  y_pred <- predict(data$x_test, data$y_test, hidden_neurons, train_model)

  labels <- c("Adelie", "Chinstrap", "Gentoo")
  y_pred <- apply(y_pred, 2, function(x) labels[which.max(x)])
  y_pred <- as.factor(y_pred)

  y_test <- apply(data$y_test, 2, function(x) labels[which.max(x)])
  y_test <- as.factor(y_test)

  cat("Accuracy: ", mean(y_pred == y_test), "%\n")
  cat("-----------------------------\n")
  cat("Confusion Matrix: \n")
  tb_nn <- table(y_test, y_pred)
  tb_nn
}