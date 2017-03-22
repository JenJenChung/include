#include "NeuralNet.h"

// Constructor: Initialises NN given layer sizes, also initialises NN activation function, currently has hardcoded mutation rates, mutation value std and bias node value
NeuralNet::NeuralNet(size_t numIn, size_t numOut, size_t numHidden){
  bias = 1.0 ;
  MatrixXd A(numIn, numHidden) ;
  weightsA = A ;
  MatrixXd B(numHidden+1, numOut) ;
  weightsB = B ;
  mutationRate = 0.5 ;
  mutationStd = 1.0 ;
  
  ActivationFunction = &NeuralNet::HyperbolicTangent ;
  InitialiseWeights(weightsA) ;
  InitialiseWeights(weightsB) ;
}

// Evaluate NN output given input vector
VectorXd NeuralNet::EvaluateNN(VectorXd inputs){
  VectorXd hiddenLayer = (this->*ActivationFunction)(inputs, 0) ;
  VectorXd outputs = (this->*ActivationFunction)(hiddenLayer, 1) ;
  return outputs ;
}

// Mutate the weights of the NN according to the mutation rate and mutation value std
void NeuralNet::MutateWeights(){
  double fan_in = weightsA.rows() ;
  for (int i = 0; i < weightsA.rows(); i++)
    for (int j = 0; j < weightsA.cols(); j++)
      weightsA(i,j) += RandomMutation(fan_in) ;
  
  fan_in = weightsB.rows() ;
  for (int i = 0; i < weightsB.rows(); i++)
    for (int j = 0; j < weightsB.cols(); j++)
      weightsB(i,j) += RandomMutation(fan_in) ;
}

// Migrated from rebhuhnc/libraries/SingleAgent/NeuralNet/NeuralNet.cpp
double NeuralNet::RandomMutation(double fan_in) {
  // Adds random amount mutationRate% of the time,
  // amount based on fan_in and mutstd
  if (rand_interval(0, 1) > mutationRate)
    return 0.0;
  else {
    // FOR MUTATION
    std::default_random_engine generator;
    generator.seed(static_cast<size_t>(time(NULL)));
    std::normal_distribution<double> distribution(0.0, mutationStd);
    return distribution(generator);
  }
}

// Assign weight matrices
void NeuralNet::SetWeights(MatrixXd A, MatrixXd B){
  weightsA = A ;
  weightsB = B ;
}

// Wrapper for writing NN weight matrices to specified files
void NeuralNet::OutputNN(const char * A, const char * B){
  // Write NN weights to txt files
  // File names stored in A and B
	std::stringstream NNFileNameA ;
	NNFileNameA << A ;
	std::stringstream NNFileNameB ;
	NNFileNameB << B ;

  WriteNN(weightsA, NNFileNameA) ;
  WriteNN(weightsB, NNFileNameB) ;
}

// Write weight matrix values to file
void NeuralNet::WriteNN(MatrixXd A, std::stringstream &fileName){
  std::ofstream NNFile ;
  NNFile.open(fileName.str().c_str()) ;
  for (int i = 0; i < A.rows(); i++){
	  for (int j = 0; j < A.cols(); j++)
	    NNFile << A(i,j) << "," ;
    NNFile << "\n" ;
	}
	NNFile.close() ;
}

// Initialise NN weight matrices to random values
void NeuralNet::InitialiseWeights(MatrixXd & A){
  double fan_in = A.rows() ;
  for (int i = 0; i < A.rows(); i++){
    for (int j = 0; j< A.cols(); j++){
      // For initialization of the neural net weights
      double rand_neg1to1 = rand_interval(-1, 1)*0.1;
      double scale_factor = 100.0;
      A(i,j) = scale_factor*rand_neg1to1 / sqrt(fan_in);
    }
  }
}

// Hyperbolic tan activation function
VectorXd NeuralNet::HyperbolicTangent(VectorXd input, size_t layer){
  VectorXd output ;
  if (layer == 0){
    output = input.transpose()*weightsA ;
  }
  else if (layer == 1){
    VectorXd hidden(input.size()+1) ;
    hidden.head(input.size()) = input ;
    hidden(input.size()) = bias ;
    output = hidden.transpose()*weightsB ;
    for (int i = 0; i < output.size(); i++)
      output(i) = tanh(output(i)) ;
  }
  else{
    std::printf("Error: second argument must be in {0,1}!\n") ;
  }
  
  return output ;
}

// Logistic function activation function
VectorXd NeuralNet::LogisticFunction(VectorXd input, size_t layer){
  VectorXd output ;
  if (layer == 0){
    output = weightsA*input ;
  }
  else if (layer == 1){
    VectorXd hidden(input.size()+1) ;
    hidden.head(input.size()) = input ;
    hidden(input.size()) = bias ;
    output = weightsB*hidden ;
    for (int i = 0; i < output.size(); i++)
      output(i) = 1/(1+exp(-output(i))) ;
  }
  else{
    std::printf("Error: second argument must be in {0,1}!\n") ;
  }
  
  return output ;
}
