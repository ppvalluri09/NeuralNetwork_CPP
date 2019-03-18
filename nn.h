
// Neural Network header file for training and classification of data. . .

#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<time.h>

using namespace std;

class NeuralNetwork {			// Neural Network class...
	private:
		int input_nodes;		// Input Layer...
		int hidden_nodes;		// Hidden Layer...
		int output_nodes;		// Output Layer...
		long double learning_rate;		// Learning Rate...
		Matrix weights_ih;		// Weights from input to hidden layer...
		Matrix weights_ho;		// Weights from hidden to output layer...
		Matrix bias_h;			// Hidden Bias...
		Matrix bias_o;			// Output Bias...


	public:
		NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) {		// Constructor to get number of input, hidden, output nodes to generate weight and bias matrices. . .
			this->input_nodes = input_nodes;
			this->hidden_nodes = hidden_nodes;
			this->output_nodes = output_nodes;

			weights_ih.setSize(this->hidden_nodes, this->input_nodes);
			weights_ho.setSize(this->output_nodes, this->hidden_nodes);
			weights_ih.randomize();
			weights_ho.randomize();


			bias_h.setSize(this->hidden_nodes, 1);
			bias_o.setSize(this->output_nodes, 1);
			bias_h.randomize();
			bias_o.randomize();

			this->learning_rate = 0.1;
		}

		long double* predict(long double*, int);
		void train(long double*, int, long double*, int);
};

long double* NeuralNetwork::predict(long double* input_array, int length) {
	Matrix inputs = Matrix::fromArray(input_array, length);
	Matrix hidden = (this->weights_ih) * inputs;
	hidden = hidden + (this->bias_h);
	// Activation Function
	hidden = Matrix::map(hidden, 1);

	Matrix output = (this->weights_ho) * hidden;
	//output = Matrix::add(output, this->bias_o);
	output = output + (this->bias_o);
	// Activation Function
	output = Matrix::map(output, 1);
	output.print();

	//return output;
}

void NeuralNetwork::train(long double* input_array, int input_length, long double* target_array, int target_length) {
	Matrix inputs = Matrix::fromArray(input_array, input_length);
	Matrix hidden = (this->weights_ih) * inputs;
	hidden = hidden + (this->bias_h);
	// Activation Function
	hidden = Matrix::map(hidden, 1);

	Matrix outputs = (this->weights_ho) * hidden;
	outputs = outputs + (this->bias_o);
	// Activation Function
	outputs = Matrix::map(outputs, 1);

	Matrix targets = Matrix::fromArray(target_array, target_length);

	Matrix output_errors = targets - outputs;

	// gradient = outputs * (1 - outputs);
	// Calculate gradient

	Matrix gradients = Matrix::map(outputs, 2);
	gradients.dot(output_errors);
	gradients.multiply(this->learning_rate);

	// Calculate deltas
	Matrix hidden_t = Matrix::transpose(hidden);
	Matrix weight_ho_deltas = Matrix::multiply(gradients, hidden_t);
	// Adjust the bais by its deltas
	this->weights_ho = (this->weights_ho) + weight_ho_deltas;
	this->bias_o = (this->bias_o) + gradients;

	// Calculate hidden layer errors...
	Matrix who_t = Matrix::transpose(this->weights_ho);
	Matrix hidden_errors = who_t * output_errors;

	// Calculate hidden layer gradient...
	Matrix hidden_gradient = Matrix::map(hidden, 2);
	hidden_gradient.dot(hidden_errors);
	hidden_gradient.multiply(this->learning_rate);

	// Calculate input->hidden deltas...
	Matrix inputs_t = Matrix::transpose(inputs);
	Matrix weight_ih_deltas = hidden_gradient * inputs_t;

	// Adjust weights and bias...
	this->weights_ih = (this->weights_ih) + weight_ih_deltas;
	this->bias_h = (this->bias_h) + hidden_gradient;
}
