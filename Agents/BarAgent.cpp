#include "BarAgent.h"

BarAgent::BarAgent(size_t nPop, string evalFunc, size_t nAct, actFun afType): popSize(nPop), numActions(nAct){
  numIn = 1 ; // hard coded for 1 element input = 1 (since agent is stateless, always pass in 1)
  numOut = 1 ; // hard coded for 1 element output k (night to visit bar)
  numHidden = 16 ;
  AgentNE = new NeuroEvo(numIn, numOut, numHidden, nPop, afType) ;
  
  if (evalFunc.compare("D") == 0)
    isD = true ;
  else if (evalFunc.compare("G") == 0)
    isD = false ;
  else{
    std::cout << "ERROR: Unknown evaluation function type [" << evalFunc << "], setting to global evaluation!\n" ;
    isD = false ;
  }
}

BarAgent::~BarAgent(){
  delete(AgentNE) ;
  AgentNE = 0 ;
}
  
void BarAgent::ResetEpochEvals(){
  // Re-initialise size of evaluations vector
  vector<double> evals(2*popSize,0) ;
  epochEvals = evals ;
}

void BarAgent::InitialiseNewLearningEpoch(vector<Bar> b){
  barNights.clear() ;
  
  for (size_t i = 0; i < b.size(); i++){
    barNights.push_back(b[i]) ;
  }
}

int BarAgent::ExecuteNNControlPolicy(size_t i){
  VectorXd s(1) ;
  s(0) = 1.0 ; // since problem is stateless, always pass in 1
  
  // Calculate action, NN returns a value between 0 and 1
  double a = AgentNE->GetNNIndex(i)->EvaluateNN(s)(0) ;
  
  // Compute discrete action
  int k = 0 ;
  bool indexFound = false ;
  for (size_t j = 0; j < numActions; j++){
    if (a <= (double)(j+1)/(double)numActions){
      k = (int)j ;
      indexFound = true ;
      break ;
    }
  }
  
  if (!indexFound){
    std::cout << "ERROR: action index not found! Performing default action 0.\n" ;
  }
  
  curAction = k ;
  return k ;
}

void BarAgent::ComputeEval(vector<int> jointState, size_t ind, double G){
  DifferenceEvaluationFunction(jointState, ind, G) ;
}

void BarAgent::SetEpochPerformance(double G, size_t i){
  if (isD)
    epochEvals[i] = D ;
  else
    epochEvals[i] = G ;
}

void BarAgent::EvolvePolicies(bool init){
  if (!init)
    AgentNE->EvolvePopulation(epochEvals) ;
  AgentNE->MutatePopulation() ;
}

void BarAgent::OutputNNs(char * A){
  // Filename to write to stored in A
  std::stringstream fileName ;
  fileName << A ;
  std::ofstream NNFile ;
  NNFile.open(fileName.str().c_str(),std::ios::app) ;
  
  // Only write in non-mutated (competitive) policies
  for (size_t i = 0; i < popSize; i++){
    NeuralNet * NN = AgentNE->GetNNIndex(i) ;
    MatrixXd NNA = NN->GetWeightsA() ;
    for (int j = 0; j < NNA.rows(); j++){
      for (int k = 0; k < NNA.cols(); k++)
        NNFile << NNA(j,k) << "," ;
      NNFile << "\n" ;
    }
    
    MatrixXd NNB = NN->GetWeightsB() ;
    for (int j = 0; j < NNB.rows(); j++){
      for (int k = 0; k < NNB.cols(); k++)
        NNFile << NNB(j,k) << "," ;
      NNFile << "\n" ;
    }
  }
  NNFile.close() ;
}

void BarAgent::DifferenceEvaluationFunction(vector<int> jointAction, size_t ind, double G){
  // Remove agent from joint action
  vector<size_t> attendance(barNights.size(), 0) ;
  for (size_t i = 0; i < jointAction.size(); i++){
    if (i != ind)
      attendance[jointAction[i]]++ ;
  }
  
  double G_hat = 0.0 ;
  for (size_t i = 0; i < barNights.size(); i++){
    G_hat += barNights[i].GetReward(attendance[i]) ;
  }
  
  D = G - G_hat ;
}
