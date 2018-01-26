#include "global.hpp"
#include "aliases.hpp"
#include <tuple>
#include <omp.h>

// average bilinears and Z
tuple<vvvvd_t,vvvvd_t> ave_err(vector<jproj_t> jG)
{
    int _moms=(int)jG.size();
    int _nbil=nbil;
    int _njacks=njacks;
    int _nmr=(int)jG[0][0][0].size();
    
    vvvvd_t G_ave(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_moms);
    vvvvd_t sqr_G_ave(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_moms);
    vvvvd_t G_err(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_moms);
    
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<_moms;imom++)
        for(int ibil=0;ibil<_nbil;ibil++)
            for(int mrA=0;mrA<_nmr;mrA++)
                for(int mrB=0;mrB<_nmr;mrB++)
                    for(int ijack=0;ijack<_njacks;ijack++)
                    {
                        G_ave[imom][ibil][mrA][mrB]+=jG[imom][ibil][ijack][mrA][mrB]/njacks;
                        sqr_G_ave[imom][ibil][mrA][mrB]+=jG[imom][ibil][ijack][mrA][mrB]*jG[imom][ibil][ijack][mrA][mrB]/njacks;
                    }
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<_moms;imom++)
        for(int ibil=0;ibil<_nbil;ibil++)
            for(int mrA=0;mrA<_nmr;mrA++)
                for(int mrB=0;mrB<_nmr;mrB++)
                    for(int ijack=0;ijack<_njacks;ijack++)
                    {
                        G_err[imom][ibil][mrA][mrB]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_G_ave[imom][ibil][mrA][mrB]-G_ave[imom][ibil][mrA][mrB]*G_ave[imom][ibil][mrA][mrB]));
                    }
                                                                                  
    tuple<vvvvd_t,vvvvd_t> tuple_ave_err(G_ave,G_err);
    
    return tuple_ave_err;
}

// average Zq
tuple<vvd_t,vvd_t> ave_err(vector<vvd_t> jZq)
{
    int _moms=(int)jZq.size();
    int _njacks=njacks;
    int _nmr=(int)jZq[0][0].size();
    
    vvd_t Zq_ave(vd_t(0.0,_nmr),_moms);
    vvd_t sqr_Zq_ave(vd_t(0.0,_nmr),_moms);
    vvd_t Zq_err(vd_t(0.0,_nmr),_moms);
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<_moms;imom++)
        for(int mr=0;mr<_nmr;mr++)
            for(int ijack=0;ijack<_njacks;ijack++)
            {
                Zq_ave[imom][mr]+=jZq[imom][ijack][mr]/njacks;
                sqr_Zq_ave[imom][mr]+=jZq[imom][ijack][mr]*jZq[imom][ijack][mr]/njacks;
            }
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<_moms;imom++)
        for(int mr=0;mr<_nmr;mr++)
            Zq_err[imom][mr]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[imom][mr]-Zq_ave[imom][mr]*Zq_ave[imom][mr]));
    
    tuple<vvd_t,vvd_t> tuple_ave_err(Zq_ave,Zq_err);
    
    return tuple_ave_err;
}

// average effective mass
tuple<vvd_t,vvd_t> ave_err(vvvd_t jM)
{
    int _njacks=njacks;
    int _nmr=(int)jM[0].size();
    
    vvd_t M_ave(vd_t(0.0,_nmr),_nmr);
    vvd_t sqr_M_ave(vd_t(0.0,_nmr),_nmr);
    vvd_t M_err(vd_t(0.0,_nmr),_nmr);
    
    for(int mrA=0;mrA<_nmr;mrA++)
        for(int mrB=0;mrB<_nmr;mrB++)
            for(int ijack=0;ijack<_njacks;ijack++)
            {
                M_ave[mrA][mrB]+=jM[ijack][mrA][mrB]/njacks;
                sqr_M_ave[mrA][mrB]+=jM[ijack][mrA][mrB]*jM[ijack][mrA][mrB]/njacks;
            }
    
    for(int mrA=0;mrA<_nmr;mrA++)
        for(int mrB=0;mrB<_nmr;mrB++)
            M_err[mrA][mrB]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_M_ave[mrA][mrB]-M_ave[mrA][mrB]*M_ave[mrA][mrB]));
    
    tuple<vvd_t,vvd_t> tuple_ave_err(M_ave,M_err);

    return tuple_ave_err;
}
