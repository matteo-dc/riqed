#include "global.hpp"
#include "aliases.hpp"
#include "allocate.hpp"
#include <tuple>
#include <omp.h>
#include <iostream>

// average bilinears
tuple<vvvvvd_t,vvvvvd_t> ave_err(vector<jproj_t> jG)
{
    int _bilmoms=(int)jG.size();
    int _nins=(int)jG[0].size();
    int _nbil=(int)jG[0][0].size();
    int _njacks=(int)jG[0][0][0].size();
    int _nmr=(int)jG[0][0][0][0].size();
    
    vvvvvd_t G_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_nins),_bilmoms);
    vvvvvd_t sqr_G_ave=G_ave;
    vvvvvd_t G_err=G_ave;
    
    for(int imom=0;imom<_bilmoms;imom++)
    {
#pragma omp parallel for collapse(4)
        for(int ins=0;ins<_nins;ins++)
            for(int ibil=0;ibil<_nbil;ibil++)
                for(int mrA=0;mrA<_nmr;mrA++)
                    for(int mrB=0;mrB<_nmr;mrB++)
                        for(int ijack=0;ijack<_njacks;ijack++)
                        {
                            G_ave[imom][ins][ibil][mrA][mrB]+=jG[imom][ins][ibil][ijack][mrA][mrB]/_njacks;
                            sqr_G_ave[imom][ins][ibil][mrA][mrB]+=(jG[imom][ins][ibil][ijack][mrA][mrB]*jG[imom][ins][ibil][ijack][mrA][mrB])/_njacks;
                        }
#pragma omp parallel for collapse(4)
        for(int ins=0;ins<_nins;ins++)
            for(int ibil=0;ibil<_nbil;ibil++)
                for(int mrA=0;mrA<_nmr;mrA++)
                    for(int mrB=0;mrB<_nmr;mrB++)
                    {
                        G_err[imom][ins][ibil][mrA][mrB]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_G_ave[imom][ins][ibil][mrA][mrB]-G_ave[imom][ins][ibil][mrA][mrB]*G_ave[imom][ins][ibil][mrA][mrB]));
                    }
    }
    
    tuple<vvvvvd_t,vvvvvd_t> tuple_ave_err(G_ave,G_err);
    
    return tuple_ave_err;
}

// average Zbil
tuple<vvvvd_t,vvvvd_t> ave_err_Z(vector<jZbil_t> jZ)
{
    int _bilmoms=(int)jZ.size();
    int _nbil=(int)jZ[0].size();
    int _njacks=(int)jZ[0][0].size();
    int _nmr=(int)jZ[0][0][0].size();
        
    vvvvd_t Z_ave(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_bilmoms);
    vvvvd_t sqr_Z_ave=Z_ave;
    vvvvd_t Z_err=Z_ave;
    
    for(int imom=0;imom<_bilmoms;imom++)
    {
#pragma omp parallel for collapse(3)
        for(int ibil=0;ibil<_nbil;ibil++)
            for(int mrA=0;mrA<_nmr;mrA++)
                for(int mrB=0;mrB<_nmr;mrB++)
                    for(int ijack=0;ijack<_njacks;ijack++)
                    {
                        Z_ave[imom][ibil][mrA][mrB]+=jZ[imom][ibil][ijack][mrA][mrB]/_njacks;
                        sqr_Z_ave[imom][ibil][mrA][mrB]+=(jZ[imom][ibil][ijack][mrA][mrB]*jZ[imom][ibil][ijack][mrA][mrB])/_njacks;
                    }
#pragma omp parallel for collapse(3)
        for(int ibil=0;ibil<_nbil;ibil++)
            for(int mrA=0;mrA<_nmr;mrA++)
                for(int mrB=0;mrB<_nmr;mrB++)
                {
                    Z_err[imom][ibil][mrA][mrB]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_ave[imom][ibil][mrA][mrB]-Z_ave[imom][ibil][mrA][mrB]*Z_ave[imom][ibil][mrA][mrB]));
                }
    }
    
    tuple<vvvvd_t,vvvvd_t> tuple_ave_err(Z_ave,Z_err);
    
    return tuple_ave_err;
}

// average sigma
tuple<vvvvd_t,vvvvd_t> ave_err(vector<vvvvd_t> sig)
{
    int _linmoms=(int)sig.size();
    int nproj=(int)sig[0].size();
    int nins=(int)sig[0][0].size();
    int _njacks=(int)sig[0][0][0].size();
    int _nmr=(int)sig[0][0][0][0].size();
    
    vvvvd_t sig_ave(vvvd_t(vvd_t(vd_t(0.0,_nmr),nins),nproj),_linmoms);
    vvvvd_t sqr_sig_ave=sig_ave;
    vvvvd_t sig_err=sig_ave;
    
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<_linmoms;imom++)
        for(int iproj=0;iproj<nproj;iproj++)
            for(int ins=0;ins<nins;ins++)
                for(int mr=0;mr<_nmr;mr++)
                    for(int ijack=0;ijack<_njacks;ijack++)
                    {
                        sig_ave[imom][iproj][ins][mr] += sig[imom][iproj][ins][ijack][mr]/_njacks;
                        sqr_sig_ave[imom][iproj][ins][mr] += sig[imom][iproj][ins][ijack][mr]*
                        sig[imom][iproj][ins][ijack][mr]/_njacks;
                    }
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<_linmoms;imom++)
        for(int iproj=0;iproj<nproj;iproj++)
            for(int ins=0;ins<nins;ins++)
                for(int mr=0;mr<_nmr;mr++)
                    sig_err[imom][iproj][ins][mr]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_sig_ave[imom][iproj][ins][mr]-sig_ave[imom][iproj][ins][mr]*sig_ave[imom][iproj][ins][mr]));
    
    tuple<vvvvd_t,vvvvd_t> tuple_ave_err(sig_ave,sig_err);
    
    return tuple_ave_err;
}

// average Zq
tuple<vvd_t,vvd_t> ave_err_Zq(vector<vvd_t> jZq)
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

// average of valence effective mass and mPCAC
tuple<vvd_t,vvd_t> ave_err(vvvd_t jM)
{
    int _njacks=jM.size();
    int _nm=(int)jM[0].size();
    
    vvd_t M_ave(vd_t(0.0,_nm),_nm);
    vvd_t sqr_M_ave(vd_t(0.0,_nm),_nm);
    vvd_t M_err(vd_t(0.0,_nm),_nm);
    
    for(int mA=0;mA<_nm;mA++)
        for(int mB=0;mB<_nm;mB++)
            for(int ijack=0;ijack<_njacks;ijack++)
            {
                M_ave[mA][mB]+=jM[ijack][mA][mB]/_njacks;
                sqr_M_ave[mA][mB]+=jM[ijack][mA][mB]*jM[ijack][mA][mB]/_njacks;
            }
    
    for(int mA=0;mA<_nm;mA++)
        for(int mB=0;mB<_nm;mB++)
            M_err[mA][mB]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_M_ave[mA][mB]-M_ave[mA][mB]*M_ave[mA][mB]));
    
    tuple<vvd_t,vvd_t> tuple_ave_err(M_ave,M_err);
    
    return tuple_ave_err;
}

// average of sea effective mass
tuple<double,double> ave_err(vd_t jM)
{
    int _njacks=jM.size();
    
    double M_ave=0.0, sqr_M_ave=0.0, M_err=0.0;
    
    for(int ijack=0;ijack<_njacks;ijack++)
    {
        M_ave+=jM[ijack]/_njacks;
        sqr_M_ave+=jM[ijack]*jM[ijack]/_njacks;
    }
    M_err=sqrt((double)(njacks-1))*sqrt(fabs(sqr_M_ave-M_ave*M_ave));
    
    tuple<double,double> tuple_ave_err(M_ave,M_err);
    
    return tuple_ave_err;
}

// average deltam
tuple<vd_t,vd_t> ave_err(vvd_t jdeltam)
{
    int _njacks=njacks;
    int _nmr=(int)jdeltam[0].size();
    
    vd_t deltam_ave(0.0,_nmr);
    vd_t sqr_deltam_ave(0.0,_nmr);
    vd_t deltam_err(0.0,_nmr);
    
    for(int mr=0;mr<_nmr;mr++)
        for(int ijack=0;ijack<_njacks;ijack++)
        {
            deltam_ave[mr]+=jdeltam[ijack][mr]/njacks;
            sqr_deltam_ave[mr]+=jdeltam[ijack][mr]*jdeltam[ijack][mr]/njacks;
        }
    
    for(int mr=0;mr<_nmr;mr++)
        deltam_err[mr]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_deltam_ave[mr]-deltam_ave[mr]*deltam_ave[mr]));
    
    tuple<vd_t,vd_t> tuple_ave_err(deltam_ave,deltam_err);
    
    return tuple_ave_err;
}

// average meslep
tuple<vvvvvvd_t,vvvvvvd_t> ave_err(vector<jproj_meslep_t> jpr_meslep)
{
    int _bilmoms=(int)jpr_meslep.size();
    int _nins=(int)jpr_meslep[0].size();
    int _nbil=(int)jpr_meslep[0][0].size();
    int _njacks=(int)jpr_meslep[0][0][0][0].size();
    int _nmr=(int)jpr_meslep[0][0][0][0][0].size();
    
    vvvvvvd_t pr_meslep_ave(vvvvvd_t(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_nbil),_nins),_bilmoms);
    vvvvvvd_t sqr_pr_meslep_ave=pr_meslep_ave;
    vvvvvvd_t pr_meslep_err=pr_meslep_ave;
    
    for(int imom=0;imom<_bilmoms;imom++)
    {
#pragma omp parallel for collapse(5)
        for(int ins=0;ins<_nins;ins++)
            for(int iop1=0;iop1<_nbil;iop1++)
                for(int iop2=0;iop2<_nbil;iop2++)
                    for(int mrA=0;mrA<_nmr;mrA++)
                        for(int mrB=0;mrB<_nmr;mrB++)
                            for(int ijack=0;ijack<_njacks;ijack++)
                            {
                                pr_meslep_ave[imom][ins][iop1][iop2][mrA][mrB]+=
                                    jpr_meslep[imom][ins][iop1][iop2][ijack][mrA][mrB]/_njacks;
                                sqr_pr_meslep_ave[imom][ins][iop1][iop2][mrA][mrB]+=
                                    (jpr_meslep[imom][ins][iop1][iop2][ijack][mrA][mrB]*jpr_meslep[imom][ins][iop1][iop2][ijack][mrA][mrB])/_njacks;
                            }
        
#pragma omp parallel for collapse(5)
        for(int ins=0;ins<_nins;ins++)
            for(int iop1=0;iop1<_nbil;iop1++)
                for(int iop2=0;iop2<_nbil;iop2++)
                    for(int mrA=0;mrA<_nmr;mrA++)
                        for(int mrB=0;mrB<_nmr;mrB++)
                        {
                            pr_meslep_err[imom][ins][iop1][iop2][mrA][mrB]=
                                sqrt((double)(njacks-1))*sqrt(fabs(sqr_pr_meslep_ave[imom][ins][iop1][iop2][mrA][mrB]-pr_meslep_ave[imom][ins][iop1][iop2][mrA][mrB]*pr_meslep_ave[imom][ins][iop1][iop2][mrA][mrB]));
                        }
    }
        
    tuple<vvvvvvd_t,vvvvvvd_t> tuple_ave_err(pr_meslep_ave,pr_meslep_err);
    
    return tuple_ave_err;
}

// average Z4f
tuple<vvvvvd_t,vvvvvd_t> ave_err_Z4f(vector<jZ4f_t> jZ4f) //TO BE CHANGED!
{
    int _bilmoms=(int)jZ4f.size();
    int _nbil=(int)jZ4f[0].size();
    int _njacks=(int)jZ4f[0][0][0].size();
    int _nmr=(int)jZ4f[0][0][0][0].size();
    
    vvvvvd_t Z4f_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_nbil),_bilmoms);
    vvvvvd_t sqr_Z4f_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_nbil),_bilmoms);
    vvvvvd_t Z4f_err(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),_nbil),_nbil),_bilmoms);
    
    for(int imom=0;imom<_bilmoms;imom++)
    {
#pragma omp parallel for collapse(4)
        for(int iop1=0;iop1<_nbil;iop1++)
            for(int iop2=0;iop2<_nbil;iop2++)
                for(int mrA=0;mrA<_nmr;mrA++)
                    for(int mrB=0;mrB<_nmr;mrB++)
                        for(int ijack=0;ijack<_njacks;ijack++)
                        {
                            Z4f_ave[imom][iop1][iop2][mrA][mrB]+=jZ4f[imom][iop1][iop2][ijack][mrA][mrB]/_njacks;
                            sqr_Z4f_ave[imom][iop1][iop2][mrA][mrB]+=(jZ4f[imom][iop1][iop2][ijack][mrA][mrB]*jZ4f[imom][iop1][iop2][ijack][mrA][mrB])/_njacks;
                        }
#pragma omp parallel for collapse(4)
        for(int iop1=0;iop1<_nbil;iop1++)
            for(int iop2=0;iop2<_nbil;iop2++)
                for(int mrA=0;mrA<_nmr;mrA++)
                    for(int mrB=0;mrB<_nmr;mrB++)
                    {
                        Z4f_err[imom][iop1][iop2][mrA][mrB]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z4f_ave[imom][iop1][iop2][mrA][mrB]-Z4f_ave[imom][iop1][iop2][mrA][mrB]*Z4f_ave[imom][iop1][iop2][mrA][mrB]));
                    }
    }
    
    tuple<vvvvvd_t,vvvvvd_t> tuple_ave_err(Z4f_ave,Z4f_err);
    
    return tuple_ave_err;
}

