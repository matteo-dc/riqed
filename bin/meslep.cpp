#include "global.hpp"
#include "aliases.hpp"
#include "Dirac.hpp"
#include <iostream>
#include "meslep.hpp"
#include "contractions.hpp"

vvvvdcompl_t build_mesloop(const vvprop_t &L)
{
    vvvvdcompl_t mesloop(vvvdcompl_t(vvdcompl_t(vdcompl_t(0.0,11),11),njacks),2); // nGamma*nProj=11*11=121 for LO and EM
    
    using namespace meslep;
        
#pragma omp parallel for collapse (3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int igam=0;igam<11;igam++)
            for(int iproj=0;iproj<11;iproj++)
            {
                prop_t op=GAMMA[iG[igam]]*(GAMMA[0]+g5L_sign[igam]*GAMMA[5]);
                prop_t pr=(GAMMA[iG[iproj]]*(GAMMA[0]+g5L_sign[iproj]*GAMMA[5])).adjoint()/2.0;
                prop_t pF=GAMMA[5]*L[ijack][EM].adjoint()*GAMMA[5];
                prop_t amp=GAMMA[5]*L[ijack][LO].inverse().adjoint()*GAMMA[5];
                
                // In the LO mesloop the external leptonic propagator is fully amputated (must be 1 if igam==iproj)
                mesloop[LO][ijack][igam][iproj] = (op*pr).trace()/12.0;
                // Multiplying for V to compensate the 1/V deriving from fft
                mesloop[EM][ijack][igam][iproj] = (op*pF*amp*pr).trace()/12.0;
            }
    
    return mesloop;
}


prop_t make_meslep(const prop_t &propOUT, const prop_t &propIN, const dcompl &lloop, const int iop, const int igam)
{
    using namespace meslep;
    
    return propOUT*GAMMA[iG[igam]]*(GAMMA[0]+g5_sign[iop]*GAMMA[5])*GAMMA[5]*propIN.adjoint()*GAMMA[5]*lloop;
}


void build_meslep(const vvvprop_t &SOUT,const vvvprop_t &SIN, const vvprop_t &L, valarray<jmeslep_t> &jmeslep)
{
    using namespace meslep;
    
    vvvvdcompl_t mesloop=build_mesloop(L);
    
    const int im1[nmeslep]={_LO,_F ,_LO,_FF,_LO,_F ,_P ,_LO,_S ,_LO};
    const int im2[nmeslep]={_LO,_LO,_F ,_LO,_FF,_F ,_LO,_P ,_LO,_S };
    const int imL[nmeslep]={_LO,_F ,_F ,_LO,_LO,_LO,_LO,_LO,_LO,_LO};
    
#pragma omp parallel for collapse (6)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int iop=0;iop<5;iop++)
                    for(int iproj=0;iproj<11;iproj++)
                        for(int ikind=0;ikind<nmeslep;ikind++)
                        {
                            vector<size_t> igam = iG_of_iop[iop];
                            
                            for(auto &ig : igam)
                            {
                                jmeslep[ikind][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][im1[ikind]][mr_fw],SIN[ijack][im2[ikind]][mr_bw],
                                            mesloop[imL[ikind]][ijack][ig][iproj],iop,ig);
                                
                                if(ikind==M11)
                                    jmeslep[M11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                        make_meslep(SOUT[ijack][_T ][mr_fw],SIN[ijack][_LO][mr_bw],
                                                    mesloop[_LO][ijack][ig][iproj],iop,ig);
                                if(ikind==M22)
                                    jmeslep[M22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                        make_meslep(SOUT[ijack][_LO][mr_fw],SIN[ijack][_T ][mr_bw],
                                                    mesloop[_LO][ijack][ig][iproj],iop,ig);
                            }
                        }
    
}

jvproj_meslep_t compute_pr_meslep(vvvprop_t &jpropOUT_inv, valarray<jmeslep_t> &jmeslep, vvvprop_t  &jpropIN_inv, const double qIN, const double qOUT, const double ql)
{
    using namespace meslep;
    
    jvproj_meslep_t jG_op(vvvvvd_t(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nOp),nOp),nprmeslep);    // 5*5=25
    // nprmeslep=8 : {QCD,IN,OUT,M11,M22,M12,A11,A22}
  
//#warning putting charges to 1
//    double Q[nprmeslep]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}; // charge factors
    double Q[nprmeslep]={1.0,ql*qIN,ql*qOUT,qIN*qIN,qOUT*qOUT,qIN*qOUT,qIN*qIN,qOUT*qOUT}; // charge factors
    const int i1[nprmeslep]={LO ,LO,LO ,LO ,LO ,LO ,EM ,LO };
    const int iv[nprmeslep]={QCD,IN,OUT,M11,M22,M12,QCD,QCD};
    const int i2[nprmeslep]={LO, LO,LO ,LO ,LO ,LO ,LO ,EM };
    
    //    cout<<"---------  projected meslep 5x5 (QCD,IN,OUT) without charges  ----------"<<endl;
#pragma omp parallel for collapse(6)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int iop1=0;iop1<nOp;iop1++)
                    for(int iop2=0;iop2<nOp;iop2++)
                        for(int k=0;k<nprmeslep;k++)
                        {
                            vector<size_t> iproj = iG_of_iop[iop2];
                            
                            for(auto &ip : iproj)
                            {
                                prop_t jLambda = Q[k]*jpropOUT_inv[i1[k]][ijack][mr_fw]*
                                                 jmeslep[iv[k]][ijack][mr_fw][mr_bw][iop1][ip]*
                                                 GAMMA[5]*(jpropIN_inv[i2[k]][ijack][mr_bw].adjoint())*GAMMA[5];
                                
                                double jGamma = (jLambda*(GAMMA[iG[ip]]*(GAMMA[0]+g5_sign[iop2]*GAMMA[5])).adjoint()).trace().real()/12.0/2.0;
                                // the factor 2.0 is to normalize the projector with (1+-g5)
                                
                                jG_op[k][iop1][iop2][ijack][mr_fw][mr_bw] += jGamma*op_norm[iop1]/proj_norm[iop2];
                            }
                        }
    
    return jG_op;
}
