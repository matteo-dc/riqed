#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <stdio.h>
#include <omp.h>
#include "operations.hpp"
#include <chrono>
#include "rotate.hpp"
#include "vertices.hpp"
#include "jack.hpp"
#include "print.hpp"
#include "prop.hpp"

using namespace std::chrono;

//project the amputated green function
vvvvvd_t compute_pr_bil( vvvprop_t &jpropOUT_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jpropIN_inv)
{
   
    const int i1[4]={LO,LO,EM,LO};
    const int iv[4]={LO,EM,LO,LO};
    const int i2[4]={LO,LO,LO,EM};
    
    int npr_bil = 4;
    
    valarray<jproj_t> jG(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil),4);
    valarray<jproj_t> jG_LO_and_EM(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil),2);
    
    const int ibil_of_igam[16]={0,1,1,1,1,2,3,3,3,3,4,4,4,4,4,4};
    
#pragma omp parallel for collapse(5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int igam=0;igam<16;igam++)
                    for(int k=0;k<npr_bil;k++)
                    {
                        prop_t jLambda_igam = jpropOUT_inv[i1[k]][ijack][mr_fw]*
                                              jVert[iv[k]][ijack][mr_fw][mr_bw][igam]*
                                              GAMMA[5]*(jpropIN_inv[i2[k]][ijack][mr_bw]).adjoint()*GAMMA[5];
                        
                        jG[k][ibil_of_igam[igam]][ijack][mr_fw][mr_bw] +=
                            (jLambda_igam*Proj[igam]).trace().real()/12.0;
                    }
    
    jG_LO_and_EM[LO] = jG[0];
    jG_LO_and_EM[EM] = (jG[1]-jG[2]-jG[3])/jG[0];  // jG_em = -jG_1+jG_a+jG_b;
    
    return jG_LO_and_EM;
}

//vvvvvd_t compute_pr_bil_4f( vvvprop_t &jprop1_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jprop2_inv, const double q1, const double q2)
//{
//    
//    const int i1[4]={LO,LO,EM,LO};
//    const int iv[4]={LO,EM,LO,LO};
//    const int i2[4]={LO,LO,LO,EM};
//    
//    // add electric charge of the quarks when one amputates with the EM corrected propagators
//    const double Q[4]={1.0,1.0,q1*q1,q2*q2};
//    
//    valarray<jvert_t> jLambda(vvvvprop_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),22),nmr),nmr),njacks),4);
//    
//    valarray<jproj_t> jG(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil+1),4);
//    valarray<jproj_t> jG_LO_and_EM(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil+1),2);
//    
//#pragma omp parallel for collapse(5)
//    for(int ijack=0;ijack<njacks;ijack++)
//        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
//            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
//                for(int igam=0;igam<22;igam++)
//                    for(int k=0;k<4;k++)
//                    {
//                        jLambda[k][ijack][mr_fw][mr_bw][igam] = Q[k]*jprop1_inv[i1[k]][ijack][mr_fw]*jVert[iv[k]][ijack][mr_fw][mr_bw][igam]*GAMMA_4f[5]*(jprop2_inv[i2[k]][ijack][mr_bw]).adjoint()*GAMMA_4f[5];
//                        
//                        if(igam==0)              jG[k][0][ijack][mr_fw][mr_bw] =(jLambda[k][ijack][mr_fw][mr_bw][0]*Proj_4f[0]).trace().real()/12.0;
//                        if(igam>0 and igam<5)    jG[k][1][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*Proj_4f[igam]).trace().real()/12.0;
//                        if(igam==5)              jG[k][2][ijack][mr_fw][mr_bw] =(jLambda[k][ijack][mr_fw][mr_bw][5]*Proj_4f[5]).trace().real()/12.0;
//                        if(igam>5 and igam<10)   jG[k][3][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*Proj_4f[igam]).trace().real()/12.0;
////                        if(igam>=10 and igam<16) jG[k][4][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*Proj_4f[igam]).trace().real()/12.0;
////                        if(igam>=16 and igam<22) jG[k][5][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*Proj_4f[igam]).trace().real()/12.0;
//                        if(igam>=10 and igam<13)
//                            jG[k][4][ijack][mr_fw][mr_bw]+=((jLambda[k][ijack][mr_fw][mr_bw][igam]+jLambda[k][ijack][mr_fw][mr_bw][igam+6])*(Proj_4f[igam]+Proj_4f[igam+6])).trace().real()/12.0;
//
//                    }
//    
//    jG_LO_and_EM[LO] = jG[0];
//    jG_LO_and_EM[EM] = jG[1]-jG[2]-jG[3];  // jG_em = -jG_1+jG_a+jG_b;
//    
//    return jG_LO_and_EM;
//}

void oper_t::compute_bil()
{
    ifstream jG_LO_data(path_print+"jG_LO");
    ifstream jG_EM_data(path_print+"jG_EM");
    
    if(jG_LO_data.good() and jG_EM_data.good())
    {
        cout<<"Reading bilinears from files"<<endl<<endl;
        
        read_vec_bin(jG_LO,path_print+"jG_LO");
        read_vec_bin(jG_EM,path_print+"jG_EM");
    }
    else
    {
        cout<<"Creating the vertices -- ";
        
        // array of input files to be read in a given conf
        FILE* input[combo];
        
        const vector<string> v_path = setup_read_qprop(input);
        
        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        {
            high_resolution_clock::time_point t0=high_resolution_clock::now();
            
            cout<<endl;
            cout<<"\r\t bilmom = "<<ibilmom+1<<"/"<<_bilmoms<<endl;
            
            const int imom1=bilmoms[ibilmom][1]; // p1
            const int imom2=bilmoms[ibilmom][2]; // p2
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
            
            // definition of jackknifed vertices
            valarray<jvert_t> jVert_LO_EM_P_S(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),_nmr),_nmr),njacks),6);
            // size=6 > {LO,self+tadpole,Pfw,Pbw,Sfw,Sbw}
            valarray<jvert_t> jVert_LO_and_EM(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),_nmr),_nmr),njacks),2);
            
            cout<<"- Building vertices"<<endl;
            
            double t_span1=0.0, t_span2=0.0, t_span3=0.0;
            
            for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
                for(int ihit=0;ihit<nhits;ihit++)
                {
                    const int mom1=linmoms[imom1][0];
                    const int mom2=linmoms[imom2][0];
                    
                    high_resolution_clock::time_point ta=high_resolution_clock::now();
                    
                    vvvprop_t S1=read_qprop_mom(input,v_path,i_in_clust,ihit,mom1);
                    vvvprop_t S2=(read2)?read_qprop_mom(input,v_path,i_in_clust,ihit,mom2):S1;
                    
                    S1=rotate(S1);
                    S2=(read2)?rotate(S2):S1;
                    
                    high_resolution_clock::time_point tb=high_resolution_clock::now();
                    t_span1 += (duration_cast<duration<double>>(tb-ta)).count();
                    
                    ta=high_resolution_clock::now();
                    
                    build_prop(S1,jS1_LO,jS1_PH,jS1_P,jS1_S);
                    if(read2) build_prop(S2,jS2_LO,jS2_PH,jS2_P,jS2_S);
                    else {jS2_LO=jS1_LO; jS2_PH=jS1_PH ; jS2_P=jS1_P; jS2_S=jS1_S;}
                    
                    tb=high_resolution_clock::now();
                    t_span2 += (duration_cast<duration<double>>(tb-ta)).count();
                    
                    ta=high_resolution_clock::now();
                    
                    build_vert(S1,S2,jVert_LO_EM_P_S);
                    
                    tb=high_resolution_clock::now();
                    t_span3 += (duration_cast<duration<double>>(tb-ta)).count();
                }
            cout<<"\t read: "<<t_span1<<" s"<<endl;
            cout<<"\t build prop: "<<t_span2<<" s"<<endl;
            cout<<"\t build vert: "<<t_span3<<" s"<<endl;
            
            
            cout<<"- Jackknife of propagators and vertices"<<endl;
            
            // jackknife averages
            jS1_LO        = jackknife(jS1_LO);
            jS1_PH = jackknife(jS1_PH);
            jS1_P        = jackknife(jS1_P);
            jS1_S        = jackknife(jS1_S);
            
            jS2_LO        = (read2)?jackknife(jS2_LO):jS1_LO;
            jS2_PH = (read2)?jackknife(jS2_PH):jS1_PH;
            jS2_P        = (read2)?jackknife(jS2_P):jS1_P;
            jS2_S        = (read2)?jackknife(jS2_S):jS1_S;
            
            for(int vkind=0; vkind<6; vkind++)
                jVert_LO_EM_P_S[vkind] = jackknife(jVert_LO_EM_P_S[vkind]);
            
            // build the complete electromagnetic correction
#pragma omp parallel for collapse(2)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr=0;mr<nmr;mr++)
                    {
                        jS1_EM[ijack][mr] = jS1_PH[ijack][mr] +
                                            deltam_cr[ijack][mr]*jS1_P[ijack][mr] +
                                            deltamu[ijack][mr]  *jS1_S[ijack][mr];
                        
                        (read2)?jS2_EM[ijack][mr] = jS2_PH[ijack][mr] +
                                                    deltam_cr[ijack][mr]*jS2_P[ijack][mr] +
                                                    deltamu[ijack][mr]  *jS2_S[ijack][mr]
                               :jS2_EM[ijack][mr] = jS1_EM[ijack][mr];
                    }
            
#pragma omp parallel for collapse (4)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                        for(int igam=0;igam<16;igam++)
                        {
                            jVert_LO_and_EM[LO][ijack][mr_fw][mr_bw][igam] =
                            jVert_LO_EM_P_S[LO][ijack][mr_fw][mr_bw][igam];
                            
                            jVert_LO_and_EM[EM][ijack][mr_fw][mr_bw][igam] =
                                jVert_LO_EM_P_S[EM][ijack][mr_fw][mr_bw][igam] +
                                deltam_cr[ijack][mr_fw]*jVert_LO_EM_P_S[Pfw][ijack][mr_fw][mr_bw][igam] +
                                deltam_cr[ijack][mr_bw]*jVert_LO_EM_P_S[Pbw][ijack][mr_fw][mr_bw][igam] +
                                deltamu[ijack][mr_fw]*jVert_LO_EM_P_S[Sfw][ijack][mr_fw][mr_bw][igam] +
                                deltamu[ijack][mr_bw]*jVert_LO_EM_P_S[Sbw][ijack][mr_fw][mr_bw][igam];
                        }
            
            
            cout<<"- Inverting propagators"<<endl;
            
            // invert propagators
            vvvprop_t jS1_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
            vvvprop_t jS2_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
            
            jS1_inv_LO_and_EM[LO] = invert_jprop(jS1_LO);
            jS1_inv_LO_and_EM[EM] = jS1_inv_LO_and_EM[LO]*jS1_EM*jS1_inv_LO_and_EM[LO];
            jS2_inv_LO_and_EM[LO] = (read2)?invert_jprop(jS2_LO):jS1_inv_LO_and_EM[LO];
            jS2_inv_LO_and_EM[EM] = (read2)?jS2_inv_LO_and_EM[LO]*jS2_EM*jS2_inv_LO_and_EM[LO]:jS1_inv_LO_and_EM[EM];
            
            cout<<"- Computing bilinears"<<endl;
            
            // compute the projected green function (S,V,P,A,T)
            vvvvvd_t jG_LO_and_EM = compute_pr_bil(jS1_inv_LO_and_EM,jVert_LO_and_EM,jS2_inv_LO_and_EM);
            
            jG_LO[ibilmom] = jG_LO_and_EM[LO];
            jG_EM[ibilmom] = jG_LO_and_EM[EM];
            
            high_resolution_clock::time_point t1=high_resolution_clock::now();
            duration<double> t_span = duration_cast<duration<double>>(t1-t0);
            cout<<"\t\t time: "<<t_span.count()<<" s"<<endl;
            
        } // close mom loop
        cout<<endl<<endl;
        
        print_vec_bin(jG_LO,path_print+"jG_LO");
        print_vec_bin(jG_EM,path_print+"jG_EM");
    }
}

void oper_t::compute_Zbil()
{
    Zbil_computed=true;
    
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        //compute Z's according to 'riqed.pdf', one for each momentum
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
                    for(int ibil=0;ibil<nbil;ibil++)
                    {
                        jZ[ibilmom][ibil][ijack][mr_fw][mr_bw] = sqrt(jZq[imom1][ijack][mr_fw]*jZq[imom2][ijack][mr_bw])/jG_LO[ibilmom][ibil][ijack][mr_fw][mr_bw];
                        
                        jZ_EM[ibilmom][ibil][ijack][mr_fw][mr_bw] = - jG_EM[ibilmom][ibil][ijack][mr_fw][mr_bw] + 0.5*(jZq_EM[imom1][ijack][mr_fw] + jZq_EM[imom2][ijack][mr_bw]);
                    }
        
    }// close mom loop
}



