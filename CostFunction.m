function [Cost] = CostFunction(Theta1,Theta2,X,y,lambda);
  
  
  m = size(X,1);
  TotalCost = 0 ;
  
  for i = 1:m
  
  #Creating the Y_Val and X_Val Vector
  ypos = y(i);
  Y_Val = zeros(12,1);
  Y_Val(ypos) = 1;
  
  X_Val = [1;X(i,:)'];
  
  #Cost per run - FeedFoward
  
  #Layer 1
  
  Z2 = (X_Val' * Theta1')'; # 20 X 1 Matrix
  A2 = sigmoid(Z2);         # 20 X 1 Matrix
  
  A2 = [1;A2];             # 21 X 1 Matrix 
  
  #Output Layer
  
  Z3 = (A2' * Theta2')'; #12 X 1 Matrix
  Predicted_Y  = sigmoid(Z3); #12 x 1
  
  #Cost Accumulator
  
  RunCost = sum((-Y_Val.* log(Predicted_Y)) - ((1-Y_Val) .* log(1-Predicted_Y)));
  TotalCost = TotalCost + RunCost;
  
endfor
  
  #Computing total cost
  Reg = 1/(2*m) * lambda * ((sum(Theta1(:).^2)) + (sum(Theta2(:).^2)));
  Cost = (1/m * TotalCost) + Reg; 


endfunction
