function [OptTheta1,OptTheta2] = BackProp(Theta1,Theta2,X,y,lambda,alpha);
  
  
m = size(X,1); #Number of Training Examples
CDelta1 = zeros(size(Theta1,1),size(Theta1,2)); # Partial Derivative Accumulator
CDelta2 = zeros(size(Theta2,1),size(Theta2,2)); # Partial Derivative Accumulator

for i = 1:m
  
  #Forward Propagation
  
  #Creating the Y_Val and X_Val Vector
  ypos = y(i);
  Y_Val = zeros(12,1);
  Y_Val(ypos) = 1;
  
  X_Val = [1;X(i,:)'];
  
  #Layer 1
  
  Z2 = (X_Val' * Theta1')'; # 20 X 1 Matrix
  A2 = sigmoid(Z2);         # 20 X 1 Matrix
  
  A2 = [1;A2];             # 21 X 1 Matrix 
  
  #Output Layer
  
  Z3 = (A2' * Theta2')'; #12 X 1 Matrix
  Predicted_Y  = sigmoid(Z3); #12 x 1
  
  #Backpropagating Error Term
  
  Delta3 = (Predicted_Y - Y_Val);
  Delta2 = Theta2' * Delta3 .* (A2 .* (1-A2) ); # 21 X 1 Matrix 
  
  Delta2 = Delta2(2:end);  #Exluding the error from the bias term
  
  CDelta2 = CDelta2 + (Delta3 * A2'); #12 X 21 Matrix
  CDelta1 = CDelta1 + (Delta2 * X_Val');  #21 X 562 Matrix
  
endfor

  Theta1_adj = Theta1;
  Theta1_adj(:,1) = 0;
  
  Theta2_adj = Theta2;
  Theta2_adj(:,1) = 0;
  
  

  Theta1_grad = (1/m * CDelta1) + (lambda/m * Theta1_adj);  #21X562
  Theta2_grad = (1/m * CDelta2) + (lambda/m * Theta2_adj);  #12X21
  
  
  #Adjusting Theta Values 
  
  OptTheta1 = Theta1 - alpha*(Theta1_grad);
  OptTheta2 = Theta2 - alpha*(Theta2_grad);


  
  
  





















endfunction

