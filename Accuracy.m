#Function calculates the accuracy of model using the test set
function [TestAcc] = Accuracy(X_test,y_test,theta1,theta2);


#Preparing the X_Val
bias  = ones(size(X_test,1),1);
X_Val = [bias X_test];  #7767 X 562

#Feedforward predictions

#Layer 1
Z2 = (X_Val * theta1')'; #20 X 7767
A2 = sigmoid(Z2);

#Output Layer
bias1 = ones(1,size(Z2,2)); # 1 X 7767
A2 = [bias1;A2];   #21 X 7767
Z3 = (A2' * theta2'); #7762 X 12
Predicted_Y = sigmoid(Z3);

#Finding the max value indices
Pred_Y = Predicted_Y';  #12 X 7762
[max_values indices] = max(Pred_Y);
final_predict = indices';  #7762 X 1

Match = (final_predict == y_test);
TestAcc = sum(Match)/(size(Match,1)); #Calculates accuracy
