#include "aliases.hpp"
#include <iostream>

vvd_t fit_par_jackknife(const vvd_t &coord, const int n_par, vd_t &error, const vvd_t &y, const int range_min, const int range_max)
{
    int njacks = y.size();
    
    MatrixXd S(n_par,n_par);
    valarray<VectorXd> Sy(VectorXd(n_par),njacks);
    valarray<VectorXd> jpars(VectorXd(n_par),njacks);
    
    vvd_t jvpars(vd_t(0.0,n_par),njacks);
    
    //initialization
    S=MatrixXd::Zero(n_par,n_par);
    for(int ijack=0; ijack<njacks; ijack++)
    {
        Sy[ijack]=VectorXd::Zero(n_par);
        jpars[ijack]=VectorXd::Zero(n_par);
    }
    
    //definition
    for(int i=range_min; i<=range_max; i++)
    {
        if(error[i]<1.0e-20) error[i]+=1.0e-20;
        
        for(int j=0; j<n_par; j++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
        
        for(int ijack=0; ijack<njacks; ijack++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]);
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
    {
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
        
        for(int ipar=0;ipar<n_par;ipar++) jvpars[ijack][ipar]=jpars[ijack](ipar);
    }
    
    for(int i=range_min; i<=range_max; i++)
        cout<<"(x,y) [ijack=0] = "<<coord[1][i]<<" "<<y[0][i]<<" "<<error[i]<<endl;
    cout<<"Extrapolation: "<<jpars[0](0)<<endl;
    
    
    return jvpars;
    
}
