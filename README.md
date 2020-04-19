# Neural Network from Scratch Activity Prediction
Smartphone-Based Recognition of Human Activities and Postural Transitions

## Data
A group of volunteers were asked to carry out a range of activites while wearing a waist-mounted smartphone, which captured accelerometer and gyroscope data. These data points were thereafter labelled to include the type of activity performed.
Source data can be found [here](https://data.world/uci/human-activity-recognition) 

The type of activites in the dataset were as follows: 
- 1 WALKING           
- 2 WALKING_UPSTAIRS  
- 3 WALKING_DOWNSTAIRS
- 4 SITTING           
- 5 STANDING          
- 6 LAYING            
- 7 STAND_TO_SIT      
- 8 SIT_TO_STAND      
- 9 SIT_TO_LIE        
- 10 LIE_TO_SIT        
- 11 STAND_TO_LIE      
- 12 LIE_TO_STAND

## Task
The objective was to create a Neural Network using the sensor data to predict the type activity performed, using a neural network which was developed from scratch in Octave using backpropagation and gradient descent to train the model.

## Model
A feedfoward three layer neural network (one hidden layer) architecture was used, with 20 units in the hidden layer. 

The neural network, backpropagation algorithm and gradient descent optimisation algorithm was developed from scratch implemented in Octave. 
The training/test set was split 70/30. The batch gradient descent technique was used, as well as regularized cost function to prevent model overfitting.

## Results
Accuracy of 65+% on the test set was obtained using this simple one hidden layer 20 node network. The cost function for the network does appear to have many local optima as test accuracy as high as 75% was seen, and with hyperparameter tuning, the accuracy could be higher.

A deeper neural network may also yield higher accuracy due to temporal nature of some of the activities, which may not be fully captured in a one hidden layer network.






