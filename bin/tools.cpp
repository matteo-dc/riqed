#include "aliases.hpp"

//! return the symmetric
vd_t symmetric(const vd_t &data)
{
    size_t s=data.size();
    vd_t out(s);
    for(size_t it=0;it<s;it++) out[(s-it)%s]=data[it];
    
    return out;
}

//! return the averaged
vd_t symmetrize(const vd_t &data, int par=1)
{
    size_t T=data.size();
    size_t TH=T/2;
    
    if(abs(par)!=1 and par!=0)
    {printf("Unknown parity %d",par); exit(1);}
    
    if(T%2)
    {printf("Size %zu odd",T); exit(1);}
    
    vd_t data_symm=symmetric(data);
    vd_t out(TH+1);
    
    for(size_t it=0;it<TH+1;it++)  out[it]=(data[it]+par*data_symm[it])/(1.0+abs(par));
    
    return out;
}
