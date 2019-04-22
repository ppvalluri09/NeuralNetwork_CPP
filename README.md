# Neural_Network_C++
A Neural Network in C++, with just one hidden layer... Make your own training data and train your Neural Network to get the desired output... The applications are endless, from using QT to make a replica of Quick Draw to implementing Neuroevolution...

# How do I use it???
Just import the library into your main.cpp file and follow the instructions below:

Create an instance of your Neural Network:
        
        NeuralNetwork nn(input_nodes, hidden_nodes, output_nodes);
        
        Example:-  
            
                   NeuralNetwork nn(2, 10, 1);        // 2 input_nodes, 10 hidden_nodes, 1 output_nodes
        
The input_nodes, hidden_nodes and the output_nodes are the number of nodes you wanna have in your neural network...
The input and output nodes depends on the training data and the target... But hidden_nodes is the User's choice... More the number of nodes in the Hidden Layer more the efficiency/accuracy of the model but it'll take more time to train the model too..

Now get your training data ready, for now we are going to test it with simple XOR problem...
        
        long double inputs[] = {1, 0};
        long double target[] = {1};
        
Now use the following to train your model with the training data:
        
        nn.train(inputs, 2, targets, 1);
        
The 2 and 1 represent the number of elements or input_nodes and output_nodes respectively... 
Here's the hitch there is something called training cycles for a model... More the number of times you train with the data the better it becomes, so you might wanna do something like:
        
        for (int i = 0; i < cycles; i++)
                nn.train(inputs, 2, target, 1);
                
Once your network is trained it's time to predict/test your model...
So prepare your testing data, in this case the training and testing data is the same so simply:-

        Matrix output = nn.predict(inputs, 2);       /* inputs array and 2 represents the number of input_nodes or elements
                                                        in the input array... predict() returns a matrix*/
        
        output.print();        /* This is to display the predicted output, output is of type Matrix which has a function defined in it's header to print the matrix in it's form... */
        
There you go... Your very own Neural Network which can predict when data is fed in... You can do a whole lot with it... Get better data, implement Quick Draw using GTK, classify images ( that requires a convolutional network but u get results )... 
Modify it, try it use it... 

Do Support and Contribute... You want more updates on this repo do add it to your watch and star it... Cheers...

Wanna learn the mathematics??? Go through the library code... :)

# nn.h
This header file contains all the neural network operations like randomizing weights and tweaking them using backpropagation and gradiel descent which are implemented as matrices... (Even vectors could've been used, but matrices were simpler to use)...

# matrix.h
Contains all the functions for all matix operations including the activation functions ( sigmoid was used here )...

# main.cpp
Driver program to test and use the Neural Network Library....
 
# Contribution by: Valluri Pavan Preetham... 
Modify it and use it to your will!!!

