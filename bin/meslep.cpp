#include "global.hpp"
#include "aliases.hpp"
#include "Dirac.hpp"
#include <iostream>
#include "meslep.hpp"
#include "operations.hpp"
#include "prop.hpp"
#include "rotate.hpp"
#include "jack.hpp"
#include "print.hpp"
#include <chrono>

using namespace std::chrono;

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
    //       jmeslep_types={QCD, IN,OUT,M11,M22,M12,P11,P22,S11,S22};
//    const int im1[nmeslep]={_LO,_LO,_F ,_LO,_FF,_F ,_LO,_P ,_LO,_S }; //out
//    const int im2[nmeslep]={_LO,_F ,_LO,_FF,_LO,_F ,_P ,_LO,_S ,_LO}; //in
//    const int imL[nmeslep]={_LO,_F ,_F ,_LO,_LO,_LO,_LO,_LO,_LO,_LO}; //lepton
    
#pragma omp parallel for collapse (5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int iop=0;iop<5;iop++)
                    for(int iproj=0;iproj<11;iproj++)
                    {
                        vector<size_t> igam = iG_of_iop[iop];
                        
                        for(auto &ig : igam)
                        {
                            jmeslep[QCD][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_LO][mr_fw],SIN[ijack][_LO][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);

                            jmeslep[IN][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_LO][mr_fw],SIN[ijack][_F][mr_bw],
                                            mesloop[_F][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[OUT][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_F][mr_fw],SIN[ijack][_LO][mr_bw],
                                            mesloop[_F][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[M11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_LO][mr_fw],SIN[ijack][_FF][mr_bw]+SIN[ijack][_T][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[M22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_FF][mr_fw]+SOUT[ijack][_T][mr_fw],SIN[ijack][_LO][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[M12][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_F][mr_fw],SIN[ijack][_F][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[P11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_LO][mr_fw],SIN[ijack][_P][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[P22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_P][mr_fw],SIN[ijack][_LO][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[S11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_LO][mr_fw],SIN[ijack][_S][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);
                            
                            jmeslep[S22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                make_meslep(SOUT[ijack][_S][mr_fw],SIN[ijack][_LO][mr_bw],
                                            mesloop[_LO][ijack][ig][iproj],iop,ig);
                            
//                            jmeslep[ikind][ijack][mr_fw][mr_bw][iop][iproj] +=
//                            make_meslep(SOUT[ijack][im1[ikind]][mr_fw],SIN[ijack][im2[ikind]][mr_bw],
//                                        mesloop[imL[ikind]][ijack][ig][iproj],iop,ig);
                        }
                    }
    
}

void oper_t::compute_meslep()
{
    //    ifstream jG_LO_4f_data(path_print+"jG_LO_4f");
    //    ifstream jG_EM_4f_data(path_print+"jG_EM_4f");
    ifstream jpr_meslep_LO_data(path_print+"jpr_meslep_LO");
    ifstream jpr_meslep_EM_data(path_print+"jpr_meslep_EM");
    ifstream jpr_meslep_nasty_data(path_print+"jpr_meslep_nasty");
    
    if(/*jG_LO_4f_data.good() and jG_EM_4f_data.good() and*/ jpr_meslep_LO_data.good() and jpr_meslep_EM_data.good() and jpr_meslep_nasty_data.good())
    {
        cout<<"Reading meslep from files"<<endl<<endl;
        
        //        read_vec_bin(jG_LO_4f,path_print+"jG_LO_4f");
        //        read_vec_bin(jG_EM_4f,path_print+"jG_EM_4f");
        read_vec_bin(jpr_meslep_LO,path_print+"jpr_meslep_LO");
        read_vec_bin(jpr_meslep_EM,path_print+"jpr_meslep_EM");
        read_vec_bin(jpr_meslep_nasty,path_print+"jpr_meslep_nasty");
    }
    else
    {
        cout<<"Creating the vertices -- ";
        
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
            jprop_t jS1_LO(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS1_PH(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS1_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS1_S(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS1_EM(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            /* prop2 */
            jprop_t jS2_LO(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS2_PH(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS2_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS2_S(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS2_EM(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            
            // definition of jackknifed meslep ("in" & "out" diagrams)
            valarray<jmeslep_t> jmeslep(jmeslep_t(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),meslep::nGamma),meslep::nOp),_nmr),_nmr),njacks),meslep::nmeslep);
            
            cout<<"- Building vertices"<<endl;
            
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
                    
                    build_prop(S1,jS1_LO,jS1_PH,jS1_P,jS1_S);
                    if(read2) build_prop(S2,jS2_LO,jS2_PH,jS2_P,jS2_S);
                    else {jS2_LO=jS1_LO; jS2_PH=jS1_PH ; jS2_P=jS1_P; jS2_S=jS1_S;}
                    
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
            
            
            cout<<"- Jackknife of propagators, vertices and meslep"<<endl;
            
            // jackknife averages
            jS1_LO        = jackknife(jS1_LO);
            jS1_PH = jackknife(jS1_PH);
            jS1_P        = jackknife(jS1_P);
            jS1_S        = jackknife(jS1_S);
            
            jS2_LO        = (read2)?jackknife(jS2_LO):jS1_LO;
            jS2_PH = (read2)?jackknife(jS2_PH):jS1_PH;
            jS2_P        = (read2)?jackknife(jS2_P):jS1_P;
            jS2_S        = (read2)?jackknife(jS2_S):jS1_S;
            
            for(int i=0;i<meslep::nmeslep;i++)
                jmeslep[i]=jackknife(jmeslep[i]);
            
            // build the complete electromagnetic correction to the propagator
#pragma omp parallel for collapse(3)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int m=0;m<nm;m++)
                    for(int r=0;r<nr;r++)
                    {
                        int mr = r + nr*m;
                        
                        jS1_EM[ijack][mr] = jS1_PH[ijack][mr] +
                        deltam_cr[ijack][m][m][r]*jS1_P[ijack][mr] +
                        deltamu[ijack][m][m][r]  *jS1_S[ijack][mr];
                        
                        (read2)?jS2_EM[ijack][mr] = jS2_PH[ijack][mr] +
                        deltam_cr[ijack][m][m][r]*jS2_P[ijack][mr] +
                        deltamu[ijack][m][m][r]  *jS2_S[ijack][mr]
                        :jS2_EM[ijack][mr]=jS1_EM[ijack][mr];
                    }
            
            // build the complete electromagnetic correction to the vertex
#pragma omp parallel for collapse (5)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                        for(int iop=0;iop<5;iop++)
                            for(int iproj=0;iproj<11;iproj++)
                            {
                                int r_fw = mr_fw%nr;
                                int m_fw = (mr_fw-r_fw)/nr;
                                int r_bw = mr_bw%nr;
                                int m_bw = (mr_bw-r_bw)/nr;
                                
                                jmeslep[M11][ijack][mr_fw][mr_bw][iop][iproj] +=
                                deltam_cr[ijack][m_bw][m_bw][r_bw]*jmeslep[P11][ijack][mr_fw][mr_bw][iop][iproj] +
                                deltamu[ijack][m_bw][m_bw][r_bw]  *jmeslep[S11][ijack][mr_fw][mr_bw][iop][iproj];
                                jmeslep[M22][ijack][mr_fw][mr_bw][iop][iproj] +=
                                deltam_cr[ijack][m_fw][m_fw][r_fw]*jmeslep[P22][ijack][mr_fw][mr_bw][iop][iproj] +
                                deltamu[ijack][m_fw][m_fw][r_fw]  *jmeslep[S22][ijack][mr_fw][mr_bw][iop][iproj];
                                //N.B.: fw(bw) corresponds to OUT(IN)!
                            }
            
            cout<<"- Inverting propagators"<<endl;
            
            // invert propagators
            vvvprop_t jS1_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
            vvvprop_t jS2_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
            
            jS1_inv_LO_and_EM[LO] = invert_jprop(jS1_LO);
            jS1_inv_LO_and_EM[EM] = jS1_inv_LO_and_EM[LO]*jS1_EM*jS1_inv_LO_and_EM[LO];
            jS2_inv_LO_and_EM[LO] = (read2)?invert_jprop(jS2_LO):jS1_inv_LO_and_EM[LO];
            jS2_inv_LO_and_EM[EM] = (read2)?jS2_inv_LO_and_EM[LO]*jS2_EM*jS2_inv_LO_and_EM[LO]:jS1_inv_LO_and_EM[EM];
            
            cout<<"- Computing projected meslep"<<endl;
            
            jvproj_meslep_t jpr_meslep = compute_pr_meslep(jS1_inv_LO_and_EM,jmeslep,jS2_inv_LO_and_EM,qIN,qOUT,ql);
            
            jpr_meslep_LO[imeslepmom] = jpr_meslep[QCD];
            jpr_meslep_EM[imeslepmom] = jpr_meslep[M11] + jpr_meslep[M22] + jpr_meslep[M12] - jpr_meslep[6] - jpr_meslep[7];
            jpr_meslep_nasty[imeslepmom] = jpr_meslep[IN] + jpr_meslep[OUT];
            
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
        
        //        print_vec_bin(jG_LO_4f,path_print+"jG_LO_4f");
        //        print_vec_bin(jG_EM_4f,path_print+"jG_EM_4f");
        print_vec_bin(jpr_meslep_LO,path_print+"jpr_meslep_LO");
        print_vec_bin(jpr_meslep_EM,path_print+"jpr_meslep_EM");
        print_vec_bin(jpr_meslep_nasty,path_print+"jpr_meslep_nasty");
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
    const int i1[nprmeslep]={LO ,LO,LO ,LO ,LO ,LO ,LO ,EM }; //propOUT_inv
    const int iv[nprmeslep]={QCD,IN,OUT,M11,M22,M12,QCD,QCD}; //meslep
    const int i2[nprmeslep]={LO, LO,LO ,LO ,LO ,LO ,EM ,LO }; //propIN_inv
    
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


void oper_t::compute_Z4f()
{
    
    //#warning putting charges to 1
    //these are the charges in the lagrangian
    const double qIN=+2.0/3.0; //!< charge of the quark (in)
    const double qOUT=-1.0/3.0; //!< charge of the quark (out)
    //    const double qIN=1.0;
    //    const double qOUT=1.0;
    
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
                            G4f_LO(iop1,iop2)=jpr_meslep_LO[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw];
                            G4f_EM(iop1,iop2)=jpr_meslep_EM[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw]+jpr_meslep_nasty[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw];
                        }
                    
                    O4f_t G4f_LO_inv = G4f_LO.inverse();
                    
                    O4f_t G4f_EM_rel = G4f_EM*G4f_LO_inv;
                    
                    O4f_t Z4f_LO = sqrt(jZq[imom1][ijack][mr_fw]*jZq[imom2][ijack][mr_bw])*G4f_LO_inv;
                    
                    O4f_t Z4f_EM = //Z4f_LO*
                    (0.5*(qOUT*qOUT*jZq_EM[imom1][ijack][mr_fw] + qIN*qIN*jZq_EM[imom2][ijack][mr_bw])*O4f_t::Identity() -G4f_EM_rel);
                    
                    for(int iop1=0;iop1<nbil;iop1++)
                        for(int iop2=0;iop2<nbil;iop2++)
                        {
                            jZ_4f[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw] = Z4f_LO(iop1,iop2);
                            jZ_EM_4f[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw] = Z4f_EM(iop1,iop2);
                        }
                }
        
    }// close mom loop
    
}

