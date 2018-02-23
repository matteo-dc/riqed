#include "global.hpp"
#include "aliases.hpp"
#include "Dirac.hpp"
#include <iostream>
#include "meslep.hpp"

vvvvdcompl_t build_mesloop(const vvprop_t &L)
{
    vvvvdcompl_t mesloop(vvvdcompl_t(vvdcompl_t(vdcompl_t(0.0,16),16),njacks),2); // nGamma*nProj=16*16=256 for LO and EM
    
    using namespace meslep;
    
#pragma omp parallel for collapse (3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int igam=0;igam<16;igam++)
            for(int iproj=0;iproj<16;iproj++)
            {
                prop_t op=GAMMA[iG[igam]]*(GAMMA[0]+g5L_sign[igam]*GAMMA[5]);
                prop_t pr=(GAMMA[iG[iproj]]*(GAMMA[0]+g5L_sign[iproj]*GAMMA[5])).adjoint();///2.0;
                prop_t pF=GAMMA[5]*L[ijack][EM].adjoint()*GAMMA[5];
                
                // In the LO mesloop the external leptonic propagator is fully amputated
                mesloop[LO][ijack][igam][iproj] = (op*pr).trace()/12.0;
                mesloop[EM][ijack][igam][iproj] = (op*pF*pr).trace()/12.0;
            }
    
    return mesloop;
}


prop_t make_meslep(const prop_t &prop1, const prop_t &prop2, const dcompl &lloop, const int igam)
{
    using namespace meslep;
    
    return prop1*GAMMA[iG[igam]]*(GAMMA[0]+g5_sign[igam]*GAMMA[5])*GAMMA[5]*prop2.adjoint()*GAMMA[5]*lloop;
}


void build_meslep(const vvvprop_t &S1,const vvvprop_t &S2, const vvprop_t &L, valarray<jmeslep_t> &jmeslep)
{
    using namespace meslep;
    
    vvvvdcompl_t mesloop=build_mesloop(L);
    
#pragma omp parallel for collapse (5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int igam=0;igam<16;igam++)
                    for(int iproj=0;iproj<16;iproj++)
                    {
                        jmeslep[QCD][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][LO][mr_fw],S2[ijack][LO][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        
                        jmeslep[IN][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][EM][mr_fw],S2[ijack][LO][mr_bw],mesloop[EM][ijack][igam][iproj],igam);
                        
                        jmeslep[OUT][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][LO][mr_fw],S2[ijack][EM][mr_bw],mesloop[EM][ijack][igam][iproj],igam);
                        
                        jmeslep[M11][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][2][mr_fw] + S1[ijack][3][mr_fw],S2[ijack][LO][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        
                        jmeslep[M22][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][LO][mr_fw],S2[ijack][2][mr_bw] + S2[ijack][3][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        
                        jmeslep[M12][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][EM][mr_fw],S2[ijack][EM][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        
                        int r_fw = mr_fw % nr;
                        int r_bw = mr_bw % nr;
                        
                        if(r_fw==0) jmeslep[P11][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(+1.0*S1[ijack][4][mr_fw],S2[ijack][0][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        if(r_fw==1) jmeslep[P11][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(-1.0*S1[ijack][4][mr_fw],S2[ijack][0][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        
                        if(r_bw==0) jmeslep[P22][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][0][mr_fw],+1.0*S2[ijack][4][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        if(r_bw==1) jmeslep[P22][ijack][mr_fw][mr_bw][igam][iproj] += make_meslep(S1[ijack][0][mr_fw],-1.0*S2[ijack][4][mr_bw],mesloop[LO][ijack][igam][iproj],igam);
                        
//                        if(mr_bw==0 and mr_fw==0)
//                        {
//                            cout<<"----------------------------"<<endl;
//                            cout<<" gam "<<igam<<" iproj "<<iproj<<" ijack "<<ijack<<" mr1 "<<mr_fw<<" mr2 "<<mr_bw<<endl;
//                            cout<<"----------------------------"<<endl;
//                            cout<<"S1 "<<S1[ijack][LO][mr_fw](0,0)<<" S1em "<<S1[ijack][EM][mr_fw](0,0)<<endl;
//                            cout<<"S2 "<<S2[ijack][LO][mr_bw](0,0)<<" S2em "<<S2[ijack][EM][mr_bw](0,0)<<endl;
//                            cout<<"mesloop QCD "<<mesloop[LO][ijack][igam][iproj]<<endl;
//                            cout<<"mesloop EM "<<mesloop[EM][ijack][igam][iproj]<<endl;
//                            cout<<"meslep QCD "<<make_meslep(S1[ijack][LO][mr_fw],S2[ijack][LO][mr_bw],mesloop[LO][ijack][igam][iproj],igam)(0,0)<<endl;
//                            cout<<"meslep IN "<<make_meslep(S1[ijack][EM][mr_fw],S2[ijack][LO][mr_bw],mesloop[EM][ijack][igam][iproj],igam)(0,0)<<endl;
//                            cout<<"meslep OUT "<<make_meslep(S1[ijack][LO][mr_fw],S2[ijack][EM][mr_bw],mesloop[EM][ijack][igam][iproj],igam)(0,0)<<endl;
//                        }
                    }
    

    
}

jvproj_meslep_t compute_pr_meslep(vvvprop_t &jprop1_inv, valarray<jmeslep_t> &jmeslep, vvvprop_t  &jprop2_inv, const double q1, const double q2, const double ql)
{
    using namespace meslep;

    valarray<jmeslep_t> jLambda(jmeslep_t(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),16),nmr),nmr),njacks),8);
    
    jvproj_meslep_t jG_g(vvvvvd_t(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),16),16),8);      // 16*16=256
    jvproj_meslep_t jG_op(vvvvvd_t(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil),nbil),8); // nbil*nbil=5*5=25
    
    // size=8 : {QCD,IN,OUT,M11,M22,M12,A11,A22}
    //  QCD: operator in pure QCD
    //  IN:  photon exchange between quark q1 and charged lepton
    //  OUT: photon exchange between quark q2 and charged lepton
    //  M11: em correction to the quark q1 propagator
    //  M22: em correction to the quark q2 propagator
    //  M12: photon exchanged between the two quarks
    //  A11: LO operator amputated with the em correction to the inverse propagator of q1
    //  A22: LO operator amputated with the em correction to the inverse propagator of q2

//    double Q[8]={1.0,ql*q1,ql*q2,q1*q1,q2*q2,q1*q2,q1*q1,q2*q2}; // charge factors
    double Q[8]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}; // charge factors
    const int i1[8]={LO,LO,LO,LO,LO,LO,EM,LO};
    const int iv[8]={QCD,IN,OUT,M11,M22,M12,QCD,QCD};
    const int i2[8]={LO,LO,LO,LO,LO,LO,LO,EM};
    
//    cout<<"---------  projected meslep 16x16 (QCD,IN,OUT) without charges  ----------"<<endl;
#pragma omp parallel for collapse(5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int igam=0;igam<16;igam++)
                    for(int iproj=0;iproj<16;iproj++)
                    {
//                        if(mr_fw==0 and mr_bw==0)
//                        {
//                            printf("igam[%d-%d] ijack[%d] - jpr_meslep:  ",igam,iproj,ijack);
//                        }
                        
                        for(int k=0; k<8; k++)
                        {
                            jLambda[k][ijack][mr_fw][mr_bw][igam][iproj] = Q[k]*jprop1_inv[i1[k]][ijack][mr_fw]*jmeslep[iv[k]][ijack][mr_fw][mr_bw][igam][iproj]*GAMMA[5]*jprop2_inv[i2[k]][ijack][mr_bw].adjoint()*GAMMA[5];
                            
                            jG_g[k][igam][iproj][ijack][mr_fw][mr_bw] = (jLambda[k][ijack][mr_fw][mr_bw][igam][iproj]*(GAMMA[iG[iproj]]*(GAMMA[0]+g5_sign[iproj]*GAMMA[5])).adjoint()).trace().real()/12.0;///2.0;
                            // the factor 2.0 is to normalize the projector with (1+-g5)
                            
//                            if(mr_fw==0 and mr_bw==0)
//                            {
//                                printf("  %lg  ",jG_g[k][igam][iproj][ijack][mr_bw][mr_fw]/Q[k]);
//                            }
                        }
//                        if(mr_fw==0 and mr_bw==0) printf("\n");
                    }
    
//    cout<<"---------  projected meslep 5x5 (QCD,IN,OUT) without charges  ----------"<<endl;
#pragma omp parallel for collapse(5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int iop1=0;iop1<nbil;iop1++)
                    for(int iop2=0;iop2<nbil;iop2++)
                    {
//                        if(mr_fw==0 and mr_bw==0)
//                        {
//                            printf("iop[%d-%d] ijack[%d] - jpr_meslep_5x5:  ",iop1,iop2,ijack);
//                        }
                        
                        for(int k=0; k<8; k++)
                        {
                            vector<size_t> igam = iG_of_iop[iop1];
                            vector<size_t> iproj = iG_of_iop[iop2];
                            
                            for(auto &ig : igam)
                                for(auto &ip : iproj)
                                {
                                    jG_op[k][iop1][iop2][ijack][mr_fw][mr_bw] += jG_g[k][ig][ip][ijack][mr_fw][mr_bw]/norm_factor[iop2];
                                }

//                            if(mr_fw==0 and mr_bw==0)
//                            {
//                                printf("  %lg  ",jG_op[k][iop1][iop2][ijack][0][0]/Q[k]);
//                            }
                        }
//                        if(mr_fw==0 and mr_bw==0) printf("\n");
                    }
    
    return jG_op;
}
