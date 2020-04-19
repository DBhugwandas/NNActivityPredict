#Creating a neural network to predict human activity from accelarometer data

#Network - 561 X 20 X 12 - One hidden Layer

#Initalising theta value
epsilon_init  = 0.12;

Theta1 = rand(20, 562) * 2 * epsilon_init - epsilon_init;
Theta2 = rand(12, 21) * 2 * epsilon_init - epsilon_init;

lambda = 0.05;
alpha = 1;



for i = 1:500
  
  #Random Sampling
  size = 100;
  rand_int = round(rand() * (7767-size));
  Rand_X = X_train(rand_int:rand_int+size,:);
  Rand_Y = y_train(rand_int:rand_int+size,:);
  
 
[Theta1,Theta2] = BackProp(Theta1,Theta2,Rand_X,Rand_Y,lambda,alpha);

Cost = CostFunction(Theta1,Theta2,X_train,y_train,lambda);
Acc = Accuracy(X_test,y_test,Theta1,Theta2);
Acc1 = Accuracy(X_train,y_train,Theta1,Theta2)

if Acc1 > 0.6
  alpha = 0.01;
  
endif
  
if Acc1 >= 0.7
  break;
  
endif 


printf("Iteration: %d \n", i);
printf("Cost: %d \n", Cost);
printf("Accuracy Train: %d \n", Acc1);
printf("Accuracy Test: %d \n", Acc);


endfor



CostFunction(Theta1,Theta2,X_train,y_train,lambda)


