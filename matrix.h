
// Header file for matrix operations associated with linear algebra and Neural Networks. . .

#include<iostream>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<iomanip>
#include<time.h>
#define MAX_ROWS 5000
#define PRECISION 6

using namespace std;

long double sigmoid(long double x) {                      // Sigmoid Activation Function
        return 1.0/(1.0 + (long double)exp(-x));
}

long double softplus(long double x) {                     // SoftPlus Activation Function
        return log(1 + exp(x)); 
}

long double elu(long double x) {                          // ELU (Exponential Linear Unit) Activation Function
        long double a = 0.1;
        if (x > 0.0) {
                return x;
        } else {
                return a * (exp(x) - 1);
        }
}


class Matrix {                          // A matrix class containing the matrix, number of rows and columns. . .
        private:
                int rows;
                int cols;
                long double *data[MAX_ROWS];
	
        public:
                Matrix(int rows, int cols) {            // Matrix constructor with the number of rows and columns. . .
                        this->rows = rows;
                        this->cols = cols;
                        for (int i=0; i < this->rows; i++) {
                                data[i] = (long double*)malloc(this->cols * sizeof(long double));
                                for (int j=0; j < this->cols; j++) {
                                        data[i][j] = 0.0;
                                }
                        }
                }

                Matrix() {
                        this->rows = 0;
                        this->cols = 0;
                }

                void setSize(int, int);
                void printSize() {
                        printf("%d, %d\n", rows, cols);
                }

                void print();                           // Function to print the matrix...
                void multiply(long double);                  // Scalar Multiplication...
                void randomize();                       // Function to assign randomly generated values to the matrices ...
                void add(long double);                       // Scalar addition...
                void subtract(long double);                  // Scalar subtraction...
                static Matrix transpose(Matrix);        // Static member function to find the transpose of the matrix...
                static Matrix add(Matrix, Matrix);      // Static member function to add two matices and return the result...
                static Matrix multiply(Matrix, Matrix); // Static member function to multiply two matices and return the result...
                static Matrix subtract(Matrix, Matrix); // Static member function to subtract two matices and return the result...
                void scale(long double n);                   // Function to scale the matrix, i.e multiply all elements by n...
                void square();                          // Function to square all the elements in the matrix...
                static Matrix map(Matrix, int);              // Function to calculate the value of the Activation Function...
                static Matrix fromArray(long double*, int);       // Function to convert an array to an object of type "Matrix"...
                long double* toArray();                      // Function to return the array associated with the object...
                void dot(Matrix);
                
                Matrix operator + (Matrix const &m);
                Matrix operator - (Matrix const &m);
                Matrix operator * (Matrix const &m);

                long double getValue() {
                        return this->data[0][0];
                }
};

Matrix Matrix::operator +(Matrix const &m)  {
        Matrix x(this->rows, this->cols);
        if (this->rows != m.rows || this->cols != m.cols) {
                cout<<"Number of rows and columns must be equal!!!"<<endl;
        } else {
                for (int i = 0; i < x.rows; i++) {
                        for (int j = 0; j < x.cols; j++) {
                                x.data[i][j] = this->data[i][j] + m.data[i][j];
                        }
                }
        }
        return x;
}

Matrix Matrix::operator -(Matrix const &m)  {
        Matrix x(this->rows, this->cols);
        if (this->rows != m.rows || this->cols != m.cols) {
                cout<<"Number of rows and columns must be equal!!!"<<endl;
        } else {
                for (int i = 0; i < x.rows; i++) {
                        for (int j = 0; j < x.cols; j++) {
                                x.data[i][j] = this->data[i][j] - m.data[i][j];
                        }
                }
        }
        return x;
}


Matrix Matrix::operator * (Matrix const &m) {
        if (this->cols != m.rows) {
                printf("The colums of the first matrix must be equal to the rows of the second matrix\n");
        } else {
                Matrix result(this->rows, m.cols);
                for (int i=0; i<result.rows; i++) {
                        for (int j=0; j<result.cols; j++) {
                                result.data[i][j] = 0.0;
                                for (int k = 0; k < this->cols; k++) {
                                        result.data[i][j] += this->data[i][k] * m.data[k][j];
                                }
                        }
                }
                return result;
        }
}


void Matrix::setSize(int rows, int cols) {
        this->rows = rows; 
        this->cols = cols;
        for (int i=0; i < this->rows; i++) {
                data[i] = (long double*)malloc(this->cols * sizeof(long double));
                for (int j=0; j < this->cols; j++) {
                        data[i][j] = 0.0;
                }
        }
}

long double* Matrix::toArray() {
        long double *a = (long double*)malloc(sizeof(long double) * (this->rows * this->cols));
        int k=0;
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        a[k++] = data[i][j];
                }
        }
        return a;
}

void Matrix::square() {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        data[i][j] = (pow(data[i][j], 2));
                }
        }
}

void Matrix::scale(long double n) {
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			data[i][j] *= n;
		}	
	}
}

Matrix Matrix::fromArray(long double *a, int length) {
        Matrix result(length, 1);
        for (int i=0; i<length; i++) {
                result.data[i][0] = a[i];
        }
        return result;
}

Matrix Matrix::map(Matrix m, int mode) {
        Matrix result(m.rows, m.cols);
        if (mode == 1) {
                for (int i=0; i<result.rows; i++) {
                        for (int j=0; j<result.cols; j++) {
                                long double val = m.data[i][j];
                                result.data[i][j] = sigmoid(val);
                        }
                }
        } else if (mode == 2) {
                for (int i=0; i<result.rows; i++) {
                        for (int j=0; j<result.cols; j++) {
                                long double val = m.data[i][j];
                                result.data[i][j] = sigmoid(val) * (1 - sigmoid(val));
                        }
                }
        }
        return result;
}

Matrix Matrix::multiply(Matrix m1, Matrix m2) {
        if (m1.cols != m2.rows) {
                printf("The colums of the first matrix must be equal to the rows of the second matrix\n");
        } else {
                Matrix result(m1.rows, m2.cols);
                for (int i=0; i<result.rows; i++) {
                        for (int j=0; j<result.cols; j++) {
                                result.data[i][j] = 0.0;
                                for (int k=0; k<m1.cols; k++) {
                                        result.data[i][j] += m1.data[i][k] * m2.data[k][j];
                                }
                        }
                }
                return result;
        }
}

Matrix Matrix::transpose(Matrix m) {
        Matrix result(m.cols, m.rows);
        for (int i=0; i<result.rows; i++) {
                for (int j=0; j<result.cols; j++) {
                        result.data[i][j] = m.data[j][i];
                }
        }
        return result;
}

Matrix Matrix::subtract(Matrix m1, Matrix m2) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
                printf("Number of Rows and Columns must be equal in both matrices\n");
        } else {
                Matrix result(m1.rows, m1.cols);
                for (int i=0; i<result.rows; i++) {
                        for (int j=0; j<result.cols; j++) {
                                result.data[i][j] = m1.data[i][j] - m2.data[i][j];
                        }
                }
                return result;
        }
}

Matrix Matrix::add(Matrix m1, Matrix m2) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
                printf("Number of Rows and Columns must be equal in both matrices\n");
        } else {
                Matrix result(m1.rows, m1.cols);
                for (int i=0; i<result.rows; i++) {
                        for (int j=0; j<result.cols; j++) {
                                result.data[i][j] = m1.data[i][j] + m2.data[i][j];
                        }
                }
                return result;
        }
}

void Matrix::print() {
        for (int i=0; i < rows; i++) {
                for (int j=0; j < cols; j++) {
                        cout<<fixed<<showpoint;
                        cout<<setprecision(PRECISION);                  // Setting the precision of data
                        cout<<data[i][j]<<"  ";
                }
                cout<<"\n";
        }
        cout<<endl;
}

void Matrix::multiply(long double n) {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        data[i][j] *= n;
                }
        }
}

void Matrix::randomize() {                      // Randomize generates the random values between 0 and 1. . .
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        data[i][j] = ((long double) rand() / (RAND_MAX));
                }
        }
}

void Matrix::add(long double n) {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        data[i][j] += n;
                }
        }
}

void Matrix::subtract(long double n) {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        data[i][j] -= n;
                }
        }
}

void Matrix::dot(Matrix m) {
        for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                        this->data[i][j] = this->data[i][j] * m.data[i][j];
                }
        }
}
