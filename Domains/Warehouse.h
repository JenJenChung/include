#ifndef WAREHOUSE_H_
#define WAREHOUSE_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <chrono>
#include <random>
#include <float.h>
#include <Eigen/Eigen>
#include <yaml-cpp/yaml.h>
#include "Agents/Intersection.h"
#include "Planning/Graph.h"
#include "Planning/Edge.h"
#include "AGV.h"

using std::vector ;
using std::list ;
using std::string ;
using std::ifstream ;
using std::stringstream ;

class Warehouse{
  public:
    Warehouse(YAML::Node) ;
    ~Warehouse() ;
    
    void SimulateEpoch(bool train = true) ;
    void EvolvePolicies(bool init = false) ;
    void ResetEpochEvals() ;
    
    void OutputPerformance(string) ;
    void OutputControlPolicies(string) ;
//    void OutputTrajectories(char *, char *) ;
//    
    void ExecutePolicies(YAML::Node) ;
    
  private:
    size_t nSteps ;
    size_t nPop ;
    size_t nAgents ;
    size_t nAGVs ;
    vector<double> baseCosts ;
    vector<size_t> capacities ;
    
    struct iAgent{
      size_t vID ;          // graph vertex ID associated with agent
      vector<size_t> eIDs ; // graph edge IDs associated with incoming edges to agent vertex
      list<size_t> agvIDs ; // agv IDs waiting to cross intersection
    } ;
    
    vector<Intersection *> maTeam ; // manage agent NE routines
    vector<iAgent *> whAgents ; // manage agent vertex and edge lookups from graph
    Graph * whGraph ; // vertex and edge definitions, access to change edge costs at each step
    vector<AGV *> whAGVs ; // manage AGV A* search and movement through graph
    
    void InitialiseGraph(string, string, string, YAML::Node) ; // read in configuration files and construct Graph
    void InitialiseMATeam() ; // create agents for each vertex in graph
    void InitialiseAGVs(YAML::Node) ; // create AGVs to move in graph
    void InitialiseNewEpoch() ; // reset simulation for each episode/epoch
    
    vector< vector<size_t> > RandomiseTeams(size_t) ; // shuffle agent populations
    
    void QueryMATeam(vector<size_t>, vector<double>&, vector<size_t>&) ; // get current graph costs
    void GetJointState(vector<Edge *> e, vector<size_t> &eNum, vector<double> &eTime) ;
    
    void UpdateGraphCosts(vector<double>) ;
    
    size_t GetAgentID(int) ;
    
    bool outputEvals ;
    bool outputNNs ;
    bool outputTrajs ;
    
    std::ofstream evalFile ;
    std::ofstream NNFile ;
    std::ofstream trajFile ;
    
};

#endif // WAREHOUSE_H_
