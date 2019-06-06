#ifndef WAREHOUSE_LINKS_H_
#define WAREHOUSE_LINKS_H_

#include <vector>
#include <list>
#include <Eigen/Eigen>
#include "Agents/Link.h"
#include "Warehouse.h"

using std::vector ;
using std::list ;

class WarehouseLinks : public Warehouse {
  public:
    WarehouseLinks(YAML::Node configs) : Warehouse(configs){}
    ~WarehouseLinks(void) ;
    
    void SimulateEpoch(bool fail = false) ;
    void SimulateEpoch(vector<size_t> team) ;
    
    void InitialiseMATeam() ; // create agents for each vertex in graph
    
  private:
    void QueryMATeam(vector<size_t>, vector<double>&, vector<size_t>&, vector<size_t>) ; // get current graph costs, final input contains IDs of failed agents
    void GetJointState(vector<Edge *> e, vector<size_t> &eNum) ;
    
};

#endif // WAREHOUSE_LINKS_NO_TIME_H_
