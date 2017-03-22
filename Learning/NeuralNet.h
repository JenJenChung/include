// Some code migrated from rebhuhnc/libraries/SingleAgent/NeuralNet/NeuralNet.h
#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <Eigen/Eigen>
#include "Utilities/Utilities.h"

using namespace Eigen ;
using easymath::rand_interval ;

class NeuralNet{
  public:
    NeuralNet(size_t numIn, size_t numOut, size_t numHidden) ; // single hidden layer
    ~NeuralNet(){}
    
    VectorXd EvaluateNN(VectorXd inputs) ;
    void MutateWeights() ;
    void SetWeights(MatrixXd, MatrixXd) ;
    MatrixXd GetWeightsA() {return weightsA ;}
    MatrixXd GetWeightsB() {return weightsB ;}
    void OutputNN(const char *, const char *) ; // write NN weights to file
    double GetEvaluation() {return evaluation ;}
    void SetEvaluation(double eval) {evaluation = eval ;}
  private:
    double bias ;
    MatrixXd weightsA ;
    MatrixXd weightsB ;
    double mutationRate ;
    double mutationStd ;
    double evaluation ;
    
    void InitialiseWeights(MatrixXd &) ;
    VectorXd (NeuralNet::*ActivationFunction)(VectorXd, size_t) ;
    VectorXd HyperbolicTangent(VectorXd, size_t) ; // outputs between [-1,1]
    VectorXd LogisticFunction(VectorXd, size_t) ; // outputs between [0,1]
    double RandomMutation(double) ;
    void WriteNN(MatrixXd, std::stringstream &) ;
} ;
#endif // NEURAL_NET_H_
