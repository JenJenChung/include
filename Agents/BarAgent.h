#ifndef BAR_AGENT_H_
#define BAR_AGENT_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"
#include "Domains/Bar.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

class BarAgent{
  public:
    BarAgent(size_t nPop, string evalFunc, size_t nNights, actFun afType=LOGISTIC) ;
    ~BarAgent() ;
    
    void ResetEpochEvals() ;
    void InitialiseNewLearningEpoch(vector<Bar>) ;
    
    int ExecuteNNControlPolicy(size_t) ; // executes NN_i, outputs index of action (assumes discrete set of actions)
    void ComputeEval(vector<int>, size_t, double) ;
    void SetEpochPerformance(double G, size_t i) ;
    
    void EvolvePolicies(bool init = false) ;
    
    void OutputNNs(char *) ;
    NeuroEvo * GetNEPopulation(){return AgentNE ;}
  private:
    size_t popSize ;
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    size_t numNights ;
    
    int curAction ;
    vector<Bar> barNights ;
    bool isD ;
    double D ;
    vector<double> epochEvals ;
    NeuroEvo * AgentNE ;
    
    void DifferenceEvaluationFunction(vector<int>, size_t, double) ;
} ;

#endif // BAR_AGENT_H_
