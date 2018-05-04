#include "aliases.hpp"
#include "global.hpp"
#include <omp.h>

// clusterize propagator
jprop_t clusterize(jprop_t &jS,vvprop_t &S)
{
#pragma omp parallel for collapse (2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
            jS[ijack][mr] += S[ijack][mr];
    
    return jS;
}

// jackknife double
vvd_t jackknife(vvd_t &jd)
{
    size_t size=jd[0].size();
    vd_t jSum(0.0,size);
    
    for(size_t j=0;j<jd.size();j++) jSum+= jd[j];

    for(size_t j=0;j<jd.size();j++)
    {
        jd[j]=jSum-jd[j];
        for(auto &it : jd[j])
            it/=(nconfs-clust_size);
    }
    
    return jd;
}

// jackknife Propagator
vvprop_t jackknife(vvprop_t &jS)
{
    valarray<prop_t> jSum(prop_t::Zero(),nmr);
    
    for(int mr=0;mr<nmr;mr++)
        for(int ijack=0;ijack<njacks;ijack++)
            jSum[mr]+= jS[ijack][mr];
    
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
            jS[ijack][mr]=(jSum[mr]-jS[ijack][mr])/((nconfs-clust_size)/*/nhits*/);
    
    return jS;
}

// jackknife Vertex
jvert_t jackknife(jvert_t &jVert)
{
    vert_t jSum(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr);
    
#pragma omp parallel for collapse(3)
    for(int mrA=0;mrA<nmr;mrA++)
        for(int mrB=0;mrB<nmr;mrB++)
            for(int igam=0;igam<16;igam++)
                for(int ijack=0;ijack<njacks;ijack++)
                    jSum[mrA][mrB][igam] += jVert[ijack][mrA][mrB][igam];

#pragma omp parallel for collapse(4)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mrA=0;mrA<nmr;mrA++)
            for(int mrB=0;mrB<nmr;mrB++)
                for(int igam=0;igam<16;igam++)
                    jVert[ijack][mrA][mrB][igam] = (jSum[mrA][mrB][igam]-jVert[ijack][mrA][mrB][igam])/((nconfs-clust_size)/**nhits*/);
    
    return jVert;
}

// jackknife meslep
jmeslep_t jackknife(jmeslep_t &jmeslep)
{
    jvert_t jSum(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),11),5),nmr),nmr);
    
#pragma omp parallel for collapse(4)
    for(int mrA=0;mrA<nmr;mrA++)
        for(int mrB=0;mrB<nmr;mrB++)
            for(int iop=0;iop<5;iop++)
                for(int iproj=0;iproj<11;iproj++)
                    for(int ijack=0;ijack<njacks;ijack++)
                        jSum[mrA][mrB][iop][iproj] += jmeslep[ijack][mrA][mrB][iop][iproj];
    
#pragma omp parallel for collapse(5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mrA=0;mrA<nmr;mrA++)
            for(int mrB=0;mrB<nmr;mrB++)
                for(int iop=0;iop<5;iop++)
                    for(int iproj=0;iproj<11;iproj++)
                        jmeslep[ijack][mrA][mrB][iop][iproj] = (jSum[mrA][mrB][iop][iproj]-jmeslep[ijack][mrA][mrB][iop][iproj])/((nconfs-clust_size)/**nhits*/);
    
    return jmeslep;
}

