#include "global.hpp"
#include "aliases.hpp"
#include "Dirac.hpp"
#include <iostream>
#include "operations.hpp"
#include "prop.hpp"
#include "rotate.hpp"
#include "jack.hpp"
#include "print.hpp"
#include <chrono>

#define EXTERN_MESLEP
 #include "meslep.hpp"

namespace pr_meslep
{
    void set_ins()
    {
        ins_list={LO,IN,OUT,M11,M22,M12,P11,P22,S11,S22};
        nins=ins_list.size();
    }
}

using namespace std::chrono;

vvvvdcompl_t build_mesloop(const vvprop_t &L)
{
    vvvvdcompl_t mesloop(vvvdcompl_t(vvdcompl_t(vdcompl_t(0.0,meslep::nGamma),meslep::nGamma),njacks),lprop::nins);
    // nGamma*nProj=11*11=121 for LO and EM
    
    using namespace meslep;
        
#pragma omp parallel for collapse (3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int igam=0;igam<meslep::nGamma;igam++)
            for(int iproj=0;iproj<meslep::nGamma;iproj++)
            {
                prop_t op=GAMMA[iG[igam]]*(GAMMA[0]+g5L_sign[igam]*GAMMA[5]);
                prop_t pr=(GAMMA[iG[iproj]]*(GAMMA[0]+g5L_sign[iproj]*GAMMA[5])).adjoint()/2.0;
                prop_t pF=GAMMA[5]*L[ijack][lprop::F].adjoint()*GAMMA[5];
                prop_t amp=GAMMA[5]*L[ijack][lprop::LO].inverse().adjoint()*GAMMA[5];
                
                // In the LO mesloop the external leptonic propagator is fully amputated (must be 1 if igam==iproj)
                mesloop[lprop::LO][ijack][igam][iproj] = (op*pr).trace()/12.0;
                mesloop[lprop::F ][ijack][igam][iproj] = (op*pF*amp*pr).trace()/12.0;
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
    
#pragma omp parallel for collapse (5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int iop=0;iop<nOp;iop++)
                    for(int iproj=0;iproj<nGamma;iproj++)
                    {
                        vector<size_t> igam = iG_of_iop[iop];
                        
                        for(auto &ig : igam)
                        {
                            jmeslep[LO][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::LO][mr_fw],
                                            SIN[ijack][qprop::LO][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);

                            jmeslep[IN][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::LO][mr_fw],
                                            SIN[ijack][qprop::F][mr_bw],
                                            mesloop[lprop::F][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[OUT][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::F][mr_fw],
                                            SIN[ijack][qprop::LO][mr_bw],
                                            mesloop[lprop::F][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[M11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::LO][mr_fw],
                                            SIN[ijack][qprop::FF][mr_bw]+SIN[ijack][qprop::T][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[M22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::FF][mr_fw]+SOUT[ijack][qprop::T][mr_fw],
                                            SIN[ijack][qprop::LO][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[M12][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::F][mr_fw],
                                            SIN[ijack][qprop::F][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[P11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::LO][mr_fw],
                                            SIN[ijack][qprop::P][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[P22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::P][mr_fw],
                                            SIN[ijack][qprop::LO][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[S11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::LO][mr_fw],
                                            SIN[ijack][qprop::S][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[S22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][qprop::S][mr_fw],
                                            SIN[ijack][qprop::LO][mr_bw],
                                            mesloop[lprop::LO][ijack][ig][iproj],iop,ig);
                        }
                    }
}

void oper_t::compute_meslep()
{
    ifstream jpr_meslep_data(path_print+"jpr_meslep");
    
    if(jpr_meslep_data.good())
    {
        cout<<"Reading meslep from files: \""<<path_print<<"jpr_meslep\""<<endl<<endl;
        
        read_vec_bin(jpr_meslep,path_print+"jpr_meslep");
    }
    else
    {
        cout<<"Creating the meslep -- ";
        
        //these are the charges in the lagrangian
        const double ql  = -1.0;     //!< the program simulates positive muon *antiparticle*
        const double qIN = +2.0/3.0; //!< charge of the quark1 (up)
        const double qOUT= -1.0/3.0; //!< charge of the quark2 (down)
        
        // array of input files to be read in a given conf
        FILE* input_q[combo];
        FILE* input_l[combo_lep];
        
        const vector<string> v_path_q = setup_read_qprop(input_q);
        const vector<string> v_path_l = setup_read_lprop(input_l);
        
        for(int imeslepmom=0;imeslepmom<_meslepmoms;imeslepmom++)
        {
            high_resolution_clock::time_point t0=high_resolution_clock::now();
            
            cout<<endl;
            cout<<"\r\t meslepmom = "<<imeslepmom+1<<"/"<<_meslepmoms<<endl;
            
            const int imom1=meslepmoms[imeslepmom][1]; // p1
            const int imom2=meslepmoms[imeslepmom][2]; // p2
            const bool read2=(imom1!=imom2);
            
            // definition of jackknifed propagators
            /* prop1 */
            jprop_t jS1(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            /* prop2 */
            jprop_t jS2(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            
            // definition of jackknifed meslep ("in" & "out" diagrams)
            valarray<jmeslep_t> jmeslep(jmeslep_t(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),meslep::nGamma),meslep::nOp),_nmr),_nmr),njacks),meslep::nmeslep);
            
            cout<<"- Building meslep"<<endl;
            
            double t_span1=0.0, t_span2=0.0, t_span4=0.0;
            
            for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
                for(int ihit=0;ihit<nhits;ihit++)
                {
                    const int mom1=linmoms[imom1][0];
                    const int mom2=linmoms[imom2][0];
                    const int momlep=mom1;
                    
                    high_resolution_clock::time_point ta=high_resolution_clock::now();
                    
                    vvvprop_t S1=read_qprop_mom(input_q,v_path_q,i_in_clust,ihit,mom1);
                    vvvprop_t S2=(read2)?read_qprop_mom(input_q,v_path_q,i_in_clust,ihit,mom2):S1;
                    vvprop_t L=read_lprop_mom(input_l,v_path_l,i_in_clust,ihit,momlep);
                    
                    S1=rotate(S1);
                    S2=(read2)?rotate(S2):S1;
                    L=rotate(L);
                    
                    high_resolution_clock::time_point tb=high_resolution_clock::now();
                    t_span1 += (duration_cast<duration<double>>(tb-ta)).count();
                    
                    ta=high_resolution_clock::now();
                    
                    build_prop(S1,jS1);
                    if(read2) build_prop(S2,jS2);
                    else {jS2=jS1;}
                    
                    tb=high_resolution_clock::now();
                    t_span2 += (duration_cast<duration<double>>(tb-ta)).count();
                    
                    ta=high_resolution_clock::now();
                    
                    build_meslep(S1,S2,L,jmeslep);
                    
                    tb=high_resolution_clock::now();
                    t_span4 += (duration_cast<duration<double>>(tb-ta)).count();
                    
                }
            cout<<"\t read: "<<t_span1<<" s"<<endl;
            cout<<"\t build prop: "<<t_span2<<" s"<<endl;
            cout<<"\t build meslep: "<<t_span4<<" s"<<endl;
            
            
            cout<<"- Jackknife of propagators and meslep"<<endl;
            
            // jackknife averages
            /* prop1 */
            for(auto &prop1 : jS1) prop1 = jackknife(prop1);
            /* prop2 */
            if(read2)
                for(auto &prop2 : jS2) prop2 = jackknife(prop2);
            else
                jS2=jS1;
            /* meslep */
            for(int ins=0;ins<meslep::nmeslep;ins++)
                jmeslep[ins]=jackknife(jmeslep[ins]);
            
            cout<<"- Inverting propagators"<<endl;
            
            // definition of inverse propagators
            jprop_t jS1_inv(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            jprop_t jS2_inv(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            
            // invert propagators
            /* prop1 */
            jS1_inv[jprop::LO] = invert_jprop(jS1[jprop::LO]);
            for(int i=1;i<jprop::nins;i++)
                jS1_inv[i] = - jS1_inv[jprop::LO]*jS1[i]*jS1_inv[jprop::LO];
            /* prop2 */
            if(read2)
            {
                jS2_inv[jprop::LO] = invert_jprop(jS2[jprop::LO]);
                for(int i=1;i<jprop::nins;i++)
                    jS2_inv[i] = - jS2_inv[jprop::LO]*jS2[i]*jS2_inv[jprop::LO];
            }
            else
                jS2_inv=jS1_inv;
            
            cout<<"- Computing projected meslep"<<endl;
            
            jpr_meslep[imeslepmom] = compute_pr_meslep(jS1_inv,jmeslep,jS2_inv,qIN,qOUT,ql);
            
//            jpr_meslep_LO[imeslepmom] = jpr_meslep[QCD];
//            jpr_meslep_EM[imeslepmom] = jpr_meslep[M11] + jpr_meslep[M22] + jpr_meslep[M12] - jpr_meslep[6] - jpr_meslep[7];
//            jpr_meslep_nasty[imeslepmom] = jpr_meslep[IN] + jpr_meslep[OUT];
            
            ////////////////////////////////////////////////////////////
            //
            //#pragma omp parallel for collapse(3)
            //        for(int ijack=0;ijack<njacks;ijack++)
            //            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            //                for(int iop1=0;iop1<nbil;iop1++)
            //                {
            //                    jpr_meslep_EM[imeslepmom][iop1][iop1][ijack][mr_fw][mr_fw] = jZq_EM[imeslepmom][ijack][mr_fw] - jpr_meslep_EM[imeslepmom][iop1][iop1][ijack][mr_fw][mr_fw];
            //                }
            //
            ////////////////////////////////////////////////////////////
            
            high_resolution_clock::time_point t1=high_resolution_clock::now();
            duration<double> t_span = duration_cast<duration<double>>(t1-t0);
            cout<<"\t\t time: "<<t_span.count()<<" s"<<endl;
            
        } // close mom loop
        cout<<endl<<endl;
        
        print_vec_bin(jpr_meslep,path_print+"jpr_meslep");
    }
}


jvproj_meslep_t compute_pr_meslep(jprop_t &jpropOUT_inv, valarray<jmeslep_t> &jmeslep, jprop_t  &jpropIN_inv, const double qIN, const double qOUT, const double ql)
{
    using namespace meslep;
    
    jvproj_meslep_t jG_op(vvvvvd_t(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nOp),nOp),nmeslep);
    
    int nloop = nmeslep + nampQED;
      
//#warning putting charges to 1
//    double Q[nprmeslep]={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    
//    charge factors
    double Q[nloop]={1.0,ql*qIN,ql*qOUT,qIN*qIN,qOUT*qOUT,qIN*qOUT,qIN*qIN,qOUT*qOUT,qIN*qIN,qOUT*qOUT,
                                        qIN*qIN,qOUT*qOUT,         qIN*qIN,qOUT*qOUT,qIN*qIN,qOUT*qOUT};
    
    const int i1[nloop]={jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,
                                                       jprop::LO,jprop::PH,          jprop::LO,jprop::P ,jprop::LO,jprop::S }; //propOUT_inv
    const int iv[nloop]={LO ,IN ,OUT,M11,M22,M12,P11,P22,S11,S22,
                                     LO ,LO ,    LO ,LO ,LO ,LO }; //meslep
    const int i2[nloop]={jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,
                                                       jprop::PH,jprop::LO,          jprop::P ,jprop::LO,jprop::S ,jprop::LO}; //propIN_inv
    
    const int im[nloop]={LO ,IN ,OUT,M11,M22,M12,P11,P22,S11,S22,
                                     M11,M22,    P11,P22,S11,S22};
    
    //    cout<<"---------  projected meslep 5x5 (QCD,IN,OUT) without charges  ----------"<<endl;
#pragma omp parallel for collapse(6)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int iop1=0;iop1<nOp;iop1++)
                    for(int iop2=0;iop2<nOp;iop2++)
                        for(int k=0;k<nloop;k++)
                        {
                            vector<size_t> iproj = iG_of_iop[iop2];
                            
                            for(auto &ip : iproj)
                            {
                                prop_t jLambda = Q[k]*jpropOUT_inv[i1[k]][ijack][mr_fw]*
                                                 jmeslep[iv[k]][ijack][mr_fw][mr_bw][iop1][ip]*
                                                 GAMMA[5]*(jpropIN_inv[i2[k]][ijack][mr_bw].adjoint())*GAMMA[5];
                                
                                double jGamma = (jLambda*(GAMMA[iG[ip]]*(GAMMA[0]+g5_sign[iop2]*GAMMA[5])).adjoint()).trace().real()/12.0/2.0;
                                // the factor 2.0 is to normalize the projector with (1+-g5)
                                
                                jG_op[im[k]][iop1][iop2][ijack][mr_fw][mr_bw] += jGamma*op_norm[iop1]/proj_norm[iop2];
                            }
                        }
    
    return jG_op;
}


void oper_t::compute_Z4f()
{
    cout<<"Computing Z4f"<<endl;
    
    //#warning putting charges to 1
    //these are the charges in the lagrangian
    const double qIN=+2.0/3.0; //!< charge of the quark (in)
    const double qOU=-1.0/3.0; //!< charge of the quark (out)
    //    const double qIN=1.0;
    //    const double qOU=1.0;
    
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        //compute Z's according to 'riqed.pdf', one for each momentum
#pragma omp parallel for collapse(3)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
                {
                    O4f_t G4f_LO(O4f_t::Zero()), G4f_EM(O4f_t::Zero());
                    
                    for(int iop1=0;iop1<nbil;iop1++)
                        for(int iop2=0;iop2<nbil;iop2++)
                        {
                            // LO
                            G4f_LO(iop1,iop2) =
                                jpr_meslep[ibilmom][meslep::LO][iop1][iop2][ijack][mr_fw][mr_bw];
                            
                            // EM
                            G4f_EM(iop1,iop2) =
                                jpr_meslep[ibilmom][meslep::IN ][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::OUT][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::M11][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::M22][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::M12][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::P11][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::P22][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::S11][iop1][iop2][ijack][mr_fw][mr_bw] +
                                jpr_meslep[ibilmom][meslep::S22][iop1][iop2][ijack][mr_fw][mr_bw];
                        }
                    
                    O4f_t G4f_LO_inv = G4f_LO.inverse();
                    
                    O4f_t G4f_EM_rel = G4f_EM*G4f_LO_inv;
                    
                    O4f_t Z4f_LO = sqrt(jZq[imom1][ijack][mr_fw]*jZq[imom2][ijack][mr_bw])*G4f_LO_inv;
                    
                    O4f_t Z4f_EM =
                        0.5*(qOU*qOU*jZq_EM[imom1][ijack][mr_fw]+
                             qIN*qIN*jZq_EM[imom2][ijack][mr_bw])*O4f_t::Identity() -
                        G4f_EM_rel;
                    
                    for(int iop1=0;iop1<nbil;iop1++)
                        for(int iop2=0;iop2<nbil;iop2++)
                        {
                            jZ_4f[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw] = Z4f_LO(iop1,iop2);
                            jZ_4f_EM[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw] = Z4f_EM(iop1,iop2);
                        }
                }
        
    }// close mom loop
    
}

