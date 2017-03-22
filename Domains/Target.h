#ifndef TARGET_H_
#define TARGET_H_

#include <float.h>
#include <Eigen/Eigen>

using namespace Eigen ;

class Target{
  public:
    Target(Vector2d xy, double v): loc(xy), val(v), obsRadius(4.0), nearestObs(DBL_MAX), observed(false){}
    ~Target(){}
    
    Vector2d GetLocation(){return loc ;}
    double GetValue(){return val ;}
    double GetNearestObs(){return nearestObs ;}
    bool IsObserved(){return observed ;}
    
    void ObserveTarget(Vector2d xy){
      Vector2d diff = xy - loc ;
      double d = diff.norm() ;
      if (observed && d < nearestObs)
        nearestObs = d ;
      else if (!observed && d <= obsRadius){
        nearestObs = d ;
        observed = true ;
      }
    }
    
    void ResetTarget(){
      nearestObs = DBL_MAX ;
      observed = false ;
    }
  private:
    Vector2d loc ;
    double val ;
    double obsRadius ;
    double nearestObs ;
    bool observed ;
} ;
#endif // TARGET_H_
