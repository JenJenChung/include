#ifndef INTERSECTION_H_
#define INTERSECTION_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"

using std::vector ;
using std::string ;

class Intersection{
  public:
    Intersection(size_t nPop, size_t nIn, size_t nOut, size_t nHidden) ;
    ~Intersection() ;
    
    void ResetEpochEvals() ;
    
    VectorXd ExecuteNNControlPolicy(size_t, VectorXd) ;
    void SetEpochPerformance(double G, size_t i) ;
    vector<double> GetEpochEvals(){return epochEvals ;}
    
    void EvolvePolicies(bool init = false) ;
    
    void OutputNNs(string) ;
    NeuroEvo * GetNEPopulation(){return IntersectionNE ;}
    
    size_t GetNumIn(){return numIn ;}
    size_t GetNumHidden(){return numHidden ;}
    size_t GetNumOut(){return numOut ;}
    
  private:
    size_t popSize ;
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    
    vector<double> epochEvals ;
    NeuroEvo * IntersectionNE ;
};

#endif // INTERSECTION_H_
