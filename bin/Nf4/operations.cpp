#include "aliases.hpp"
#include "global.hpp"
#include "read.hpp"
#include "jack.hpp"
#include "Zq.hpp"
#include "Zbil.hpp"
#include "Dirac.hpp" //useless
#include "vertices.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include "fit.hpp"
#include <stdio.h>
#include "subtraction.hpp"
#include "evolution.hpp"
#include "print.hpp"
#include "ave_err.hpp"
#include <chrono>

#define EXTERN_OPER

#include "operations.hpp"

#include "vertices.hpp"

using namespace std::chrono;

SCHEME_t get_scheme()
{
    if(scheme=="RI_MOM") return RI_MOM;
    if(scheme=="SMOM") return SMOM;
    
    return ERR;
}

void oper_t::set_moms()
{
    switch(get_scheme())
    {
        case RI_MOM:
            set_ri_mom_moms();
            break;
        case SMOM:
            cout<<"SMOM!"<<endl;
            set_smom_moms();
            break;
        case ERR:
            cout<<"Invalid scheme."<<endl;
            exit(0);
            break;
    }
}

void oper_t::set_ri_mom_moms()
{
    bilmoms.resize(moms);
    for(int imom=0;imom<moms;imom++)
    {
        bilmoms[imom]={imom,imom,imom};
    }
}

void oper_t::set_smom_moms()
{
    double eps=1e-10;
    for(int i=0;i<moms;i++)
        for(int j=0;j<moms;j++)
        {
            if(2.0*fabs(p2[i]-p2[j])<(p2[i]+p2[j])*eps)
            {
                coords_t momk;
                p_t k_array;
                double k2=0.0;
                for(size_t mu=0;mu<4;mu++)
                {
                    momk[mu]=mom_list[i][mu]+mom_list[j][mu];
                    k_array[mu]=2*M_PI*momk[mu]/size[mu];
                    k2+=k_array[mu]*k_array[mu];
                }
                
                if(2.0*fabs(p2[i]-k2)<(p2[i]+k2)*eps)
                {
                    //search in mom_list
                    auto posk = find(mom_list.begin(),mom_list.end(),momk);
                    
                    if(posk!=mom_list.end())
                    {
                        const int k=distance(mom_list.begin(),posk);
                        //inform and add to the list
//                        cout<<"Found smom pair: "<<i<<" ";
//                        for(auto &ip1 : mom_list[i]) cout<<ip1;
//                        cout<<" + "<<j<<" ";
//                        for(auto &ip2 : mom_list[j]) cout<<ip2;
//                        cout<<" = "<<k<<" ";
//                        for(auto &ik : momk)cout<<ik;
//                        cout<<endl;
                        bilmoms.push_back({k,i,j});
                    }
                    else
                        cout<<"Unable to find it!"<<endl;
                }
                else cout<<"p2-k2 != 0"<<endl;
                
            }else cout<<"p1^2-p2^2 != 0"<<endl;
            
        }
}

////////

void oper_t::create_basic()
{
    step = "basic";
    
    _nm=nm;
    _nr=nr;
    _nmr=_nm*_nr;
    
    set_moms();
    
    allocate();
    
    switch(get_scheme())
    {
        case RI_MOM:
            ri_mom();
            break;
        case SMOM:
            cout<<"SMOM!"<<endl;
            smom();
            break;
        case ERR:
            cout<<"Invalid scheme."<<endl;
            exit(0);
            break;
    }
    
    compute_Zbil();
}

void oper_t::ri_mom()
{
//    compute_prop();
    compute_bil();
}

void oper_t::smom()
{
    ri_mom();
}


//////////

void oper_t::allocate()
{
    jZq.resize(bilmoms.size());
    jZq_em.resize(bilmoms.size());
    
    jG_0.resize(bilmoms.size());
    jG_em.resize(bilmoms.size());
    
    jZ.resize(bilmoms.size());
    jZ_em.resize(bilmoms.size());
    
//    jZq_ave_r.resize(bilmoms.size());
//    jZq_em_ave_r.resize(bilmoms.size());
    
//    m_eff_equivalent_Zq.resize(neq2);
//    m_eff_equivalent.resize(neq);
    
//    jZq_ave_r.resize(bilmoms.size());
//    jZq_em_ave_r.resize(bilmoms.size());
//    
//    jG_0_ave_r.resize(bilmoms.size());
//    jG_em_ave_r.resize(bilmoms.size());
//    
//    jZq_chir.resize(bilmoms.size());
//    jZq_em_chir.resize(bilmoms.size());
//    
//    jG_0_chir.resize(bilmoms.size());
//    jG_em_chir.resize(bilmoms.size());
//    
//    jZ_chir.resize(bilmoms.size());
//    jZ_em_chir.resize(bilmoms.size());
    
//    cout<<"jZq "<<jZq.size()<<" "<<jZq_em.size()<<endl;
//    cout<<"jG "<<jG_0.size()<<" "<<jG_em.size()<<endl;
//    cout<<"jZ "<<jZ.size()<<" "<<jZ_em.size()<<endl;
//    cout<<"jG_ave_r "<<jG_0_ave_r.size()<<" "<<jG_em_ave_r.size()<<endl;
//    cout<<"jG_0_chir "<<jG_0_chir.size()<<" "<<jG_em_chir.size()<<endl;
    }


vvvprop_t build_prop(jprop_t &jS_0,jprop_t &jS_em,vvvprop_t &S)
{
//    vvvprop_t S_LO_or_EM(vprop_t(prop_t::Zero(),nmr),njacks);
    vvvprop_t S_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),2);
    
#pragma omp parallel for collapse(3)
    for(int m=0;m<nm;m++)
        for(int r=0;r<nr;r++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                int mr = r + nr*m;
                
                S_LO_and_EM[LO][ijack][mr] = S[ijack][0][mr];
                
                // Electromagnetic correction:  S_em = S_self + S_tad -+ deltam_cr*S_P
                if(r==0) S_LO_and_EM[EM][ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] + deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=0
                if(r==1) S_LO_and_EM[EM][ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] - deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=1
           
                jS_0[ijack][mr] += S_LO_and_EM[LO][ijack][mr];
                jS_em[ijack][mr] += S_LO_and_EM[EM][ijack][mr];
            }
    
//    jS_0=clusterize(jS_0,S_LO_and_EM[LO]);
//    jS_em=clusterize(jS_em,S_LO_and_EM[EM]);
    
    return S_LO_and_EM;
}

//void oper_t::compute_prop()
//{
//    cout<<"Creating the propagators -- ";
//    
//    // array of input files to be read in a given conf
//    ifstream input[combo];
//    vector<string> v_path = setup_read_prop(input);
//    
//    vvvd_t jZq_LO_and_EM(vvd_t(vd_t(0.0,nmr),njacks),2);
//    
//    for(int imom=0; imom<moms; imom++)
//    {
//        cout<<"\r\t mom = "<<imom+1<<"/"<<moms<<flush;
//        
//        // definition of jackknifed propagators
//        jprop_t jS_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
//        jprop_t jS_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
//
//        // initialize propagators
//        vvvprop_t S(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
//        vvprop_t S_0(vprop_t(prop_t::Zero(),nmr),njacks);
//        vvprop_t S_em(vprop_t(prop_t::Zero(),nmr),njacks);
//        vvvprop_t S_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),2);
//        
//        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
//            for(int ihit=0;ihit<nhits;ihit++)
//            {
//                S=read_prop_mom(input,v_path,i_in_clust,ihit,imom);
//                
//                S_LO_and_EM = build_prop(jS_0,jS_em,S);
//                
////                S_0 = build_prop(jS_0,S,LO);
////                S_em = build_prop(jS_em,S,EM);
//            }
//        
//        S_0 = S_LO_and_EM[LO];
//        S_em = S_LO_and_EM[EM];
//        
//        // jackknife average
//        jS_0=jackknife(jS_0);
//        jS_em=jackknife(jS_em);
//        
//        // invert propagator
//        vvvprop_t jS_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),2);
//        
//         jS_inv_LO_and_EM[LO] = invert_jprop(jS_0);
//         jS_inv_LO_and_EM[EM] = jS_inv_LO_and_EM[LO]*jS_em*jS_inv_LO_and_EM[LO];
//        
//        // compute quark field RCs (Zq or Sigma1 established from input file!) and store
//        jZq_LO_and_EM = compute_jZq(jS_inv_LO_and_EM,imom);
//        
//        jZq[imom] = jZq_LO_and_EM[LO];
//        jZq_em[imom] = - jZq_LO_and_EM[EM];
//        
////        jZq[imom] = compute_jZq(jS_0_inv,jS_em_inv,imom);
////        jZq_em[imom] = - compute_jZq(jS_em_inv,imom);
//        
//        // printf("%lf\n",jZq[0][0]);
//        
//    } // close mom loop
//    cout<<endl<<endl;
//}

void oper_t::compute_bil()
{
    cout<<"Creating the vertices -- ";
    
    // array of input files to be read in a given conf
    ifstream input[combo];
    const vector<string> v_path = setup_read_prop(input);
    
    int mom_size = (int)bilmoms.size();
    
    // initialize propagators
    vvvprop_t S1(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
    vvprop_t S1_0(vprop_t(prop_t::Zero(),nmr),njacks);
    vvprop_t S1_em(vprop_t(prop_t::Zero(),nmr),njacks);
    vvvprop_t S2(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
    vvprop_t S2_0(vprop_t(prop_t::Zero(),nmr),njacks);
    vvprop_t S2_em(vprop_t(prop_t::Zero(),nmr),njacks);
    
    vvvprop_t S1_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),2);
    vvvprop_t S2_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),2);

    
    for(int ibilmom=0;ibilmom<mom_size;ibilmom++)
    {
        high_resolution_clock::time_point t0=high_resolution_clock::now();

        cout<<endl;
        cout<<"\r\t mom = "<<ibilmom+1<<"/"<<mom_size<<endl;
        
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        const bool read2=(imom1!=imom2);
        
        // definition of jackknifed propagators
        jprop_t jS1_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jprop_t jS1_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jprop_t jS2_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jprop_t jS2_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        
        // definition of vertices
        valarray<jvert_t> jVert_LO_and_EM(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks),2);
        //        jvert_t jVert_0 (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        //        jvert_t jVert_em (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);

        
        cout<<"- Reading propagators and building vertices"<<endl;
        
        double t_span1=0.0, t_span2=0.0, t_span3=0.0;
        
        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
            for(int ihit=0;ihit<nhits;ihit++)
            {
                high_resolution_clock::time_point ta=high_resolution_clock::now();
                
                S1=read_prop_mom(input,v_path,i_in_clust,ihit,imom1);
                S2=(read2)?read_prop_mom(input,v_path,i_in_clust,ihit,imom2):S1;
                
                high_resolution_clock::time_point tb=high_resolution_clock::now();
                t_span1 += (duration_cast<duration<double>>(tb-ta)).count();
                
                ta=high_resolution_clock::now();
                
                S1_LO_and_EM = build_prop(jS1_0,jS1_em,S1);
                S2_LO_and_EM = (read2)?build_prop(jS2_0,jS2_em,S2):S1_LO_and_EM;
                
                tb=high_resolution_clock::now();
                t_span2 += (duration_cast<duration<double>>(tb-ta)).count();
                
                S1_em = S1_LO_and_EM[EM];
                S2_em = S2_LO_and_EM[EM];
                
                ta=high_resolution_clock::now();

                jVert_LO_and_EM = build_vert(S1,S2,S1_em,S2_em,jVert_LO_and_EM);
                
                tb=high_resolution_clock::now();
                t_span3 += (duration_cast<duration<double>>(tb-ta)).count();
            }
        cout<<"\t read: "<<t_span1<<" s"<<endl;
        cout<<"\t build prop: "<<t_span2<<" s"<<endl;
        cout<<"\t build vert: "<<t_span3<<" s"<<endl;

        
//        S1_0 = S1_LO_and_EM[LO];
//        S2_0 = S2_LO_and_EM[LO];
//        S1_em = S1_LO_and_EM[EM];
//        S2_em = S2_LO_and_EM[EM];
        
//        jvert_t jVert_0 = jVert_LO_and_EM[LO];
//        jvert_t jVert_em = jVert_LO_and_EM[EM];
        
        cout<<"- Jackknife of propagators and vertices"<<endl;
        
        // jackknife averages
        jS1_0=jackknife(jS1_0);
        jS1_em=jackknife(jS1_em);
        jS2_0=(read2)?jackknife(jS2_0):jS1_0;
        jS2_em=(read2)?jackknife(jS2_em):jS1_em;
        
        jVert_LO_and_EM[LO]=jackknife(jVert_LO_and_EM[LO]);
        jVert_LO_and_EM[EM]=jackknife(jVert_LO_and_EM[EM]);
        
        
        cout<<"- Inverting propagators"<<endl;

        // invert propagators
        vvvprop_t jS1_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),2);
        vvvprop_t jS2_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),2);
        
        jS1_inv_LO_and_EM[LO] = invert_jprop(jS1_0);
        jS1_inv_LO_and_EM[EM] = jS1_inv_LO_and_EM[LO]*jS1_em*jS1_inv_LO_and_EM[LO];
        jS2_inv_LO_and_EM[LO] = (read2)?invert_jprop(jS2_0):jS1_inv_LO_and_EM[LO];
        jS2_inv_LO_and_EM[EM] = (read2)?jS2_inv_LO_and_EM[LO]*jS2_em*jS2_inv_LO_and_EM[LO]:jS1_inv_LO_and_EM[EM];
        
        cout<<"- Computing Zq"<<endl;
        
        // compute Zq relative to imom1
        vvvd_t jZq_LO_and_EM = compute_jZq(jS1_inv_LO_and_EM,imom1);
        
        jZq[imom1] = jZq_LO_and_EM[LO];
        jZq_em[imom1] = - jZq_LO_and_EM[EM];
        
        cout<<"- Computing bilinears"<<endl;
        
        // compute the projected green function (S,V,P,A,T)
        vvvvvd_t jG_LO_and_EM = compute_pr_bil(jS1_inv_LO_and_EM,jVert_LO_and_EM,jS2_inv_LO_and_EM);
        
//        jproj_t jG_0_mom = compute_pr_bil(jS1_0_inv, jVert_0, jS2_0_inv);
//        jproj_t jG_1 = compute_pr_bil(jS1_0_inv, jVert_em, jS2_0_inv);
//        jproj_t jG_a = compute_pr_bil(jS1_em_inv, jVert_0, jS2_0_inv);
//        jproj_t jG_b = compute_pr_bil(jS1_0_inv, jVert_0, jS2_em_inv);
//        
//        jproj_t jG_em_mom = -jG_1+jG_a+jG_b;
        
//        jG_0[imom1]=jG_0_mom;
//        jG_em[imom1]=jG_em_mom;
        
        jG_0[imom1] = jG_LO_and_EM[LO];
        jG_em[imom1] = jG_LO_and_EM[EM];
        
        high_resolution_clock::time_point t1=high_resolution_clock::now();
        duration<double> t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"\t\t time: "<<t_span.count()<<" s"<<endl;
    
    } // close mom loop
    cout<<endl<<endl;
}

void oper_t::compute_Zbil()
{
    Zbil_computed=true;
    
    for(int ibilmom=0;ibilmom<(int)bilmoms.size();ibilmom++)
    {
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        // definition of the RCs estimators
        jZbil_t jZ_mom(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),njacks),nbil);
        jZbil_t jZ_em_mom(vvvd_t(vvd_t(vd_t(0.0,_nmr),_nmr),njacks),nbil);
        
        //compute Z's according to 'riqed.pdf', one for each momentum
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
                    for(int ibil=0;ibil<nbil;ibil++)
                    {
                        jZ_mom[ibil][ijack][mr_fw][mr_bw] = sqrt(jZq[imom1][ijack][mr_fw])*sqrt(jZq[imom2][ijack][mr_bw])/jG_0[imom1][ibil][ijack][mr_fw][mr_bw];
                        
                        jZ_em_mom[ibil][ijack][mr_fw][mr_bw] = jG_em[imom1][ibil][ijack][mr_fw][mr_bw]/jG_0[imom1][ibil][ijack][mr_fw][mr_bw] + 0.5*(jZq_em[imom1][ijack][mr_fw]/jZq[imom1][ijack][mr_fw] + jZq_em[imom2][ijack][mr_bw]/jZq[imom2][ijack][mr_bw]);
                    }
        
        jZ[imom1]=jZ_mom;
        jZ_em[imom1]=jZ_em_mom;
        
    }// close mom loop
}

void oper_t::resize_vectors(oper_t out)
{
    for(auto &ijack : out.jZq)
        for(auto &mr : ijack)
            mr.resize(out._nmr);
    
    for(auto &ijack : out.jZq_em)
        for(auto &mr : ijack)
            mr.resize(out._nmr);
    
    for(auto &ibil : out.jG_0)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }

    for(auto &ibil : out.jG_em)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }
    
    for(auto &ibil : out.jZ)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }
    
    for(auto &ibil : out.jZ_em)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }
}

oper_t oper_t::average_r(/*const bool recompute_Zbil*/)
{
    cout<<"Averaging over r"<<endl<<endl;
    
    oper_t out=(*this);
    
    out._nr=1;
    out._nm=_nm;
    out._nmr=(out._nm)*(out._nr);
    
//    out.bilmoms=bilmoms;
//    out.Zbil_computed=Zbil_computed;
    
    resize_vectors(out);

//    out.jZq_em=jZq_em;
//    out.jG_0=jG_0;
//    out.jG_em=jG_em;
//    out.jZ=jZ;
//    out.jZ_em=jZ_em;
    

//    // to be cancelled
//    out.jG_0_chir=jG_0_chir;
//    out.jG_em_chir=jG_em_chir;
//    out.jZq_chir=jZq_chir;
//    out.jZq_em_chir=jZq_chir;
//    out.jZ_chir=jZ_chir;
//    out.jZ_em_chir=jZ_em_chir;
  
//    for(int ieq=0;ieq<neq2;ieq++) m_eff_equivalent_Zq[ieq]=0.0;
//    for(int ieq=0;ieq<neq;ieq++)  m_eff_equivalent[ieq]=0.0;
    
    if(UseEffMass==1)
    {
        vvd_t eff_mass_temp(vd_t(0.0,out._nmr),out._nmr);
        
        for(int mA=0; mA<_nm; mA++)
            for(int mB=0; mB<_nm; mB++)
                for(int r=0; r<_nr; r++)
                {
                    // masses
                    eff_mass_temp[mA][mB] += eff_mass[r+_nr*mA][r+_nr*mB]/_nr;
                }
        eff_mass=eff_mass_temp;
    }
    
    for(size_t ibilmom=0;ibilmom<bilmoms.size();ibilmom++)
    {
        const int imom1=bilmoms[ibilmom][1]; // p1
//        const int imom2=bilmoms[ibilmom][2]; // p2
        
        vvd_t jZq_mom_temp(vd_t(0.0,out._nmr),njacks);
        vvd_t jZq_em_mom_temp(vd_t(0.0,out._nmr),njacks);
        
        for(int m=0; m<_nm; m++)
            for(int r=0; r<_nr; r++)
            {
                //LO
                for(int ijack=0;ijack<njacks;ijack++) jZq_mom_temp[ijack][m] += jZq[imom1][ijack][r+_nr*m]/_nr;
                //EM
                for(int ijack=0;ijack<njacks;ijack++) jZq_em_mom_temp[ijack][m] += jZq_em[imom1][ijack][r+_nr*m]/_nr;
            }
        
        (out.jZq)[imom1] = jZq_mom_temp;
        (out.jZq_em)[imom1] = jZq_em_mom_temp;
        
//        vvvd_t jG_0_ave_r_mom(vvd_t(vd_t(0.0,neq),njacks),5);
//        vvvd_t jG_em_ave_r_mom(vvd_t(vd_t(0.0,neq),njacks),5);
        
        jproj_t jG_0_mom_temp(vvvd_t(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks),nbil);
        jproj_t jG_em_mom_temp(vvvd_t(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks),nbil);

        
        for(int mA=0; mA<_nm; mA++)
            for(int mB=0; mB<_nm; mB++)
                for(int r=0; r<_nr; r++)
                {
//                    // masses
//                    if(UseEffMass==1 and ibilmom==0)
//                        m_eff_equivalent[ieq] += (eff_mass[r+nr*mA][r+nr*mB]+eff_mass[r+nr*mB][r+nr*mA])/(2.0*nr);
//                    else if(UseEffMass==0 and ibilmom==0 and r==0)
//                        m_eff_equivalent[ieq] = mass_val[mA] + mass_val[mB];
//                    
//                    //LO
//                    for(int ijack=0;ijack<njacks;ijack++)
//                        for(int ibil=0; ibil<5; ibil++)
//                            jG_0_ave_r_mom[ibil][ijack][ieq] += (jG_0_mom[ijack][r+nr*mA][r+nr*mB][ibil]+jG_0_mom[ijack][r+nr*mB][r+nr*mA][ibil])/(2.0*nr);
//                    //EM
//                    for(int ijack=0;ijack<njacks;ijack++)
//                        for(int ibil=0; ibil<5; ibil++)
//                            jG_em_ave_r_mom[ibil][ijack][ieq] += (jG_em_mom[ijack][r+nr*mA][r+nr*mB][ibil]+jG_em_mom[ijack][r+nr*mB][r+nr*mA][ibil])/(2.0*nr);
//                    
                    //LO
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int ibil=0; ibil<5; ibil++)
                            jG_0_mom_temp[ibil][ijack][mA][mB] += jG_0[imom1][ibil][ijack][r+_nr*mA][r+_nr*mB]/_nr;
                    //EM
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int ibil=0; ibil<5; ibil++)
                            jG_em_mom_temp[ibil][ijack][mA][mB] += jG_em[imom1][ibil][ijack][r+nr*mA][r+nr*mB]/nr;
                    
                    
                }
        
        (out.jG_0)[imom1]=jG_0_mom_temp;
        (out.jG_em)[imom1]=jG_em_mom_temp;
    }
    
    out.compute_Zbil();
    
    return out;
}
    

oper_t oper_t::chiral_extr()
{
    cout<<"Chiral extrapolation"<<endl<<endl;
    
    oper_t out=(*this);
    
    out._nr=_nr;
    out._nm=1;
    out._nmr=(out._nm)*(out._nr);
    
    resize_vectors(out);
    
    vvvvd_t G_0_err = get<1>(ave_err(jG_0));    //[imom][ibil][mr1][mr2]
    vvvvd_t G_em_err = get<1>(ave_err(jG_em));
    
    vvd_t Zq_err = get<1>(ave_err(jZq));        //[imom][mr]
    vvd_t Zq_em_err = get<1>(ave_err(jZq_em));
    
    //Sum of quark masses for the extrapolation
//    vd_t mass_sum(0.0,10);
//    int i_sum = 0;
//    for (int i=0; i<nm; i++)
//        for(int j=i;j<nm;j++)
//        {
//            mass_sum[i_sum] = mass_val[i]+mass_val[j];
//            i_sum++;
//        }

    
    //range for fit Zq
    int x_min_q=0;
    int x_max_q=_nm-1;
    
    // range for fit bilinears
    int x_min=0;
    int x_max=_nm*(_nm+1)/2-1;
    
    // number of fit parameters for bilinears
    int npar[5]={3,2,3,2,2};
    
    //extrapolate Zq
    for(size_t imom=0;imom<bilmoms.size();imom++)
    {
        for(int r=0; r<_nr; r++)
        {
            vvd_t coord_q(vd_t(0.0,_nm),2); // coords at fixed r
            
            vvvd_t jZq_r(vvd_t(vd_t(0.0,_nm),njacks),bilmoms.size());
            vvvd_t jZq_em_r(vvd_t(vd_t(0.0,_nm),njacks),bilmoms.size());
            
            vvd_t Zq_err_r(vd_t(0.0,_nm),bilmoms.size());
            vvd_t Zq_em_err_r(vd_t(0.0,_nm),bilmoms.size());
            
            for(int m=0; m<_nm; m++)
            {
                int mr = r + _nr*m;
                
                coord_q[0][m] = 1.0;
                if(UseEffMass==0)
                    coord_q[1][m]= mass_val[m];
                else if(UseEffMass==0)
                    coord_q[1][m] = pow(eff_mass[mr][mr],2.0);
                
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    jZq_r[imom][ijack][m]=jZq[imom][ijack][mr];
                    jZq_em_r[imom][ijack][m]=jZq[imom][ijack][mr];
                }
                
                Zq_err_r[imom][m]=Zq_err[imom][mr];
                Zq_em_err_r[imom][m]=Zq_em_err[imom][mr];
            }
            
            vvd_t jZq_pars_mom_r = polyfit(coord_q,2,Zq_err_r[imom],jZq_r[imom],x_min_q,x_max_q);
            vvd_t jZq_em_pars_mom_r = polyfit(coord_q,2,Zq_em_err_r[imom],jZq_em_r[imom],x_min_q,x_max_q);
            
            for(int ijack=0; ijack<njacks; ijack++)
            {
                (out.jZq)[imom][ijack][r]=jZq_pars_mom_r[ijack][0];
                (out.jZq_em)[imom][ijack][r]=jZq_em_pars_mom_r[ijack][0];
            }
        }
    }

    //extrapolate bilinears
    for(size_t imom=0;imom<bilmoms.size();imom++)
    {
        for(int r1=0; r1<_nr; r1++)
            for(int r2=0; r2<_nr; r2++)
            {
                vvd_t coord_bil(vd_t(0.0,_nm*(_nm+1)/2),3); // coords at fixed r1 and r2
                
                vvvvd_t jG_0_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),bilmoms.size());
                vvvvd_t jG_em_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),bilmoms.size());
                
                vvvd_t G_0_err_r1_r2(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),bilmoms.size());
                vvvd_t G_em_err_r1_r2(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),bilmoms.size());

                int ieq=0;
                for(int m1=0; m1<_nm; m1++)
                    for(int m2=m1; m2<_nm; m2++)
                    {
                        int mr1 = r1 + _nr*m1;
                        int mr2 = r2 + _nr*m2;

                        coord_bil[0][ieq] = 1.0;
                        if(UseEffMass==0)
                        {
                            coord_bil[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
                            coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];    // 1/(am1+am2)
                        }
                        else if(UseEffMass==1)
                        {
                            coord_bil[1][ieq] = pow((eff_mass[mr1][mr2]+eff_mass[mr2][mr1])/2.0,2.0);   //M^2 (averaged over equivalent combinations)
                            coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];  //1/M^2
                        }
                    
                        for(int ibil=0;ibil<nbil;ibil++)
                        {
                            for(int ijack=0;ijack<njacks;ijack++)
                            {
                                jG_0_r1_r2[imom][ibil][ijack][ieq] = (jG_0[imom][ibil][ijack][mr1][mr2]+jG_0[imom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0;
                                jG_em_r1_r2[imom][ibil][ijack][ieq] = (jG_em[imom][ibil][ijack][mr1][mr2]+jG_em[imom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0;
                            }
                            
                            G_0_err_r1_r2[imom][ibil][ieq] = (G_0_err[imom][ibil][mr1][mr2] + G_0_err[imom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0;
                            G_em_err_r1_r2[imom][ibil][ieq] = (G_em_err[imom][ibil][mr1][mr2] + G_em_err[imom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0;
                        }
                        
                        ieq++;
                    }
                
                for(int ibil=0;ibil<nbil;ibil++)
                {
                    vvd_t jG_0_pars_mom_ibil_r1_r2 = polyfit(coord_bil,npar[ibil],G_0_err_r1_r2[imom][ibil],jG_0_r1_r2[imom][ibil],x_min,x_max);
                    vvd_t jG_em_pars_mom_ibil_r1_r2 = polyfit(coord_bil,npar[ibil],G_em_err_r1_r2[imom][ibil],jG_em_r1_r2[imom][ibil],x_min,x_max);
                    
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
//                        if(ibil==0 or ibil==2)
//                            for(int ieq=0;ieq<neq;ieq++)
//                            {
//                                // Goldstone pole subtraction from bilinears
//                                jG_0_ave_r[imom][ibil][ijack][ieq] -= jG_0_pars_mom[ibil][ijack][2];
//                                jG_em_ave_r[imom][ibil][ijack][ieq] -= jG_em_pars_mom[ibil][ijack][2];
//                            }
                        
                        // extrapolated value
                        (out.jG_0)[imom][ibil][ijack][r1][r2] = jG_0_pars_mom_ibil_r1_r2[ijack][0];
                        (out.jG_em)[imom][ibil][ijack][r1][r2] = jG_em_pars_mom_ibil_r1_r2[ijack][0];
                    }
                }
            }
    }
    
    out.compute_Zbil();
    
    return out;
}

oper_t oper_t::subtract()
{
    cout<<"Subtracting the O(a2) effects"<<endl<<endl;
    
    oper_t out=(*this);
    
//    out.bilmoms=bilmoms;
//    out.Zbil_computed=Zbil_computed;
//    out.jZq=jZq;
//    out.jZq_em=jZq_em;
//    out.jG_0=jG_0;
//    out.jG_em=jG_em;
//    out.jZ=jZ;
//    out.jZ_em=jZ_em;
//    out.m_eff_equivalent_Zq=m_eff_equivalent_Zq;
//    out.m_eff_equivalent=m_eff_equivalent;
//    out.jZq_ave_r=jZq_ave_r;
//    out.jZq_em_ave_r=jZq_em_ave_r;
//    out.jZq_chir = jZq_chir;
//    out.jZq_em_chir = jZq_em_chir;
//    out.jZ_chir = jZ_chir;
//    out.jZ_em_chir = jZ_em_chir;
//    
//    jG_0_sub = jG_0_chir;
//    jG_em_sub = jG_em_chir;
//    jZq_sub = jZq_chir;
//    jZq_em_sub = jZq_em_chir;
//    
//    jZ_sub = jZ_chir;
//    jZ_em_sub = jZ_em_chir;
    
#pragma omp parallel for collapse(3)
    for(size_t imom=0;imom<bilmoms.size();imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0; mr1<_nmr; mr1++)
            {
                (out.jZq)[imom][ijack][mr1] = jZq[imom][ijack][mr1] - subtraction_q(imom,LO);
                (out.jZq_em)[imom][ijack][mr1] = jZq_em[imom][ijack][mr1] + /*(!)*/ subtraction_q(imom,EM)*jZq[imom][ijack][mr1];
                // N.B.: the subtraction gets an extra minus sign due to the definition of the e.m. expansion!
                
                for(int ibil=0;ibil<5;ibil++)
                    for(int mr2=0; mr2<_nmr; mr2++)
                    {
                        (out.jG_0)[imom][ibil][ijack][mr1][mr2] = jG_0[imom][ibil][ijack][mr1][mr2] - subtraction(imom,ibil,LO);
                        (out.jG_em)[imom][ibil][ijack][mr1][mr2] = jG_em[imom][ibil][ijack][mr1][mr2] - subtraction(imom,ibil,EM)*jG_0[imom][ibil][ijack][mr1][mr2];
                        
                        //                // constructing Z_sub
                        //                jZ_sub[imom][ibil][ijack] = jZq_sub[imom][ijack]/jG_0_sub[imom][ibil][ijack];
                        //                jZ_em_sub[imom][ibil][ijack] = jG_em_sub[imom][ibil][ijack]/jG_0_sub[imom][ibil][ijack] + jZq_em_sub[imom][ijack]/jZq_sub[imom][ijack];
                    }
            }

//    out.jG_0_sub = jG_0_sub;
//    out.jG_em_sub = jG_em_sub;
//    out.jZq_sub = jZq_sub;
//    out.jZq_em_sub = jZq_em_sub;
//    
//    out.jZ_sub = jZ_sub;
//    out.jZ_em_sub = jZ_em_sub;

    out.compute_Zbil();
    
    return out;
}

oper_t oper_t::evolve()
{
    cout<<"Evolving the Z's to the scale 1/a"<<endl<<endl;
    
    oper_t out=(*this);
    
//    out.bilmoms=bilmoms;
//    out.Zbil_computed=Zbil_computed;
//    out.jZq_sub = jZq_sub;
//    out.jZq_em_sub = jZq_em_sub;
//    out.jZ_sub = jZ_sub;
//    out.jZ_em_sub = jZ_em_sub;
    
//    jZq_evo = jZq_sub;
//    jZq_em_evo = jZq_em_sub;
//    jZ_evo = jZ_sub;
//    jZ_em_evo = jZ_em_sub;
    
//    vector<vd_t> jZq_evo_tmp(jZq_sub.size(),vd_t(0.0,njacks)), jZq_em_evo_tmp(jZq_sub.size(),vd_t(0.0,njacks));
//    vector<vvd_t> jZ_evo_tmp(jZ_sub.size(),vvd_t(vd_t(0.0,njacks),nbil)), jZ_em_evo_tmp(jZ_sub.size(),vvd_t(vd_t(0.0,njacks),nbil));

    double cq=0.0;
    vd_t cO(0.0,5);
    
    for(size_t imom=0;imom<bilmoms.size();imom++)
    {
        // Note that ZV  ZA are RGI because they're protected by the WIs
        cq=q_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);
        cO[0]=S_evolution_to_RIp_ainv(Nf,ainv,p2[imom]); //S
        cO[1]=1.0;                                       //A
        cO[2]=P_evolution_to_RIp_ainv(Nf,ainv,p2[imom]); //P
        cO[3]=1.0;                                       //V
        cO[4]=T_evolution_to_RIp_ainv(Nf,ainv,p2[imom]); //T
        
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0; mr1<_nmr; mr1++)
            {
                (out.jZq)[imom][ijack][mr1] = jZq[imom][ijack][mr1]/cq;
                (out.jZq_em)[imom][ijack][mr1] = jZq_em[imom][ijack][mr1]/cq;
                
                for(int ibil=0;ibil<5;ibil++)
                    for(int mr2=0; mr2<_nmr; mr2++)
                    {
                        (out.jZ)[imom][ibil][ijack][mr1][mr2] = jZ[imom][ibil][ijack][mr1][mr2]/cO[ibil];
                        (out.jZ_em)[imom][ibil][ijack][mr1][mr2] = jZ_em[imom][ibil][ijack][mr1][mr2]/cO[ibil];
                    }
            }
    }
    
//    out.jZq_evo = jZq_evo_tmp;
//    out.jZq_em_evo = jZq_em_evo_tmp;
//    out.jZ_evo = jZ_evo_tmp;
//    out.jZ_em_evo = jZ_em_evo_tmp;
    
    return out;
}

double mom_xyz(size_t imom)
{
    return abs(mom_list[imom][1])*abs(mom_list[imom][2])*abs(mom_list[imom][3]);
}

oper_t oper_t::average_equiv_moms()
{
    cout<<"Averaging over the equivalent momenta -- ";
    
    oper_t out=(*this);
    
    int tag=0, tag_aux=0;
    double eps=1.0e-15;
    
    vector<int> tag_vector;
    tag_vector.push_back(0);
    
    //Tag assignment
    for(size_t imom=0;imom<bilmoms.size();imom++)
    {
        size_t count_no=0;
        
        for(size_t j=0;j<imom;j++)
        {
            if( abs(p2_tilde[j]-p2_tilde[imom])<eps*p2_tilde[j] && abs(mom_xyz(j)-mom_xyz(imom))<eps*mom_xyz(j) )
            {
                tag_aux=tag_vector[j];
            }else count_no++;
            
            if(count_no==imom)
            {
                tag++;
                tag_vector.push_back(tag);
            }else if(j==imom-1)
            {
                tag_vector.push_back(tag_aux);
            }
        }
    }
    
    int neq_moms = tag+1;
    
    cout<<"found: "<<neq_moms<<" equivalent momenta."<<endl<<endl;
    
    vector<int> count_tag_vector(neq_moms);
    vector<double> p2_tilde_eqmoms(neq_moms);

    int count=0;
    for(int tag=0;tag<neq_moms;tag++)
    {
        count=0;
        for(size_t imom=0;imom<bilmoms.size();imom++)
        {
            if(tag_vector[imom]==tag) count++;
        }
        count_tag_vector[tag]=count;
    }
    
    for(int tag=0;tag<neq_moms;tag++)
        for(size_t imom=0;imom<bilmoms.size();imom++)
        {
            if(tag_vector[imom]==tag)  p2_tilde_eqmoms[tag] = p2_tilde[imom];
        }
    
    PRINT(p2_tilde_eqmoms);
    
//    out.jZq_evo = jZq_evo;
//    out.jZq_em_evo = jZq_em_evo;
//    out.jZ_evo = jZ_evo;
//    out.jZ_em_evo = jZ_em_evo;

    (out.jZq).resize(neq_moms);
    (out.jZq_em).resize(neq_moms);
    (out.jZ).resize(neq_moms);
    (out.jZ_em).resize(neq_moms);
    
    // initialize to zero
#pragma omp parallel for collapse(3)
    for(int tag=0;tag<neq_moms;tag++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0; mr1<_nmr; mr1++)
            {
                (out.jZq)[tag][ijack][mr1]=0.0;
                (out.jZq_em)[tag][ijack][mr1]=0.0;
                
                for(int ibil=0;ibil<5;ibil++)
                    for(int mr2=0; mr2<_nmr; mr2++)
                    {
                        (out.jZ)[tag][ibil][ijack][mr1][mr2]=0.0;
                        (out.jZ_em)[tag][ibil][ijack][mr1][mr2]=0.0;
                    }
            }
    
    // average over the equivalent momenta
//#pragma omp parallel for collapse(2)
    for(int tag=0;tag<neq_moms;tag++)
        for(size_t imom=0;imom<bilmoms.size();imom++)
        {
            if(tag_vector[imom]==tag)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr1=0; mr1<_nmr; mr1++)
                {
                    (out.jZq)[tag][ijack][mr1]+=jZq[imom][ijack][mr1]/count_tag_vector[tag];
                    (out.jZq_em)[tag][ijack][mr1]+=jZq_em[imom][ijack][mr1]/count_tag_vector[tag];
                    
                    for(int ibil=0;ibil<5;ibil++)
                        for(int mr2=0; mr2<_nmr; mr2++)
                    {
                        (out.jZ)[tag][ibil][ijack][mr1][mr2]+=jZ[imom][ibil][ijack][mr1][mr2]/count_tag_vector[tag];
                        (out.jZ_em)[tag][ibil][ijack][mr1][mr2]+=jZ_em[imom][ibil][ijack][mr1][mr2]/count_tag_vector[tag];
                    }
                }
            }
        }
    
    return out;
}

//! To be used after the average over the equivalent momenta! (*)
void continuum_limit(oper_t out, const int LO_or_EM)
{
    //! (*)
    int neq_moms = (out.jZq).size();
    vector<double> p2_tilde_eqmoms(neq_moms);
    READ(p2_tilde_eqmoms);
    
    vvd_t jZq_out(vd_t(0.0,neq_moms),njacks);
    vvvd_t jZ_out(vvd_t(vd_t(0.0,neq_moms),njacks),nbil);
    
    vd_t Zq_err(0.0,neq_moms);
    vvd_t Z_err(vd_t(0.0,neq_moms),nbil);
    
    if(LO_or_EM==0)
    {
        cout<<"-- Leading Order --"<<endl;
        
#pragma omp parallel for collapse(2)
        for(int imom=0; imom<neq_moms; imom++)
            for(int ijack=0; ijack<njacks; ijack++)
            {
                jZq_out[ijack][imom] = out.jZq[imom][ijack][0];
                for(int ibil=0; ibil<nbil; ibil++)
                    jZ_out[ibil][ijack][imom] = out.jZ[imom][ibil][ijack][0][0];
            }
        vvd_t Zq_err_tmp = get<1>(ave_err(out.jZq));
        vvvvd_t Z_err_tmp = get<1>(ave_err(out.jZ));
        
        for(int imom=0; imom<neq_moms; imom++)
        {
            Zq_err[imom] = Zq_err_tmp[imom][0];
            for(int ibil=0; ibil<nbil; ibil++)
                Z_err[ibil][imom] = Z_err_tmp[imom][ibil][0][0];
        }
        
    }
    else if(LO_or_EM==1)
    {
        cout<<"-- EM Correction --"<<endl;
        
#pragma omp parallel for collapse(2)
        for(int imom=0; imom<neq_moms; imom++)
            for(int ijack=0; ijack<njacks; ijack++)
            {
                jZq_out[ijack][imom] = out.jZq_em[imom][ijack][0];
                for(int ibil=0; ibil<nbil; ibil++)
                    jZ_out[ibil][ijack][imom] = out.jZ_em[imom][ibil][ijack][0][0];
            }
        vvd_t Zq_err_tmp = get<1>(ave_err(out.jZq_em));
        vvvvd_t Z_err_tmp = get<1>(ave_err(out.jZ_em));
        
        for(int imom=0; imom<neq_moms; imom++)
        {
            Zq_err[imom] = Zq_err_tmp[imom][0];
            for(int ibil=0; ibil<nbil; ibil++)
                Z_err[ibil][imom] = Z_err_tmp[imom][ibil][0][0];
        }
    }
    
    //linear fit
    int range_min=0;  //a2p2~1
    int range_max=neq_moms;
//    int p_min_value=0.9;
    double p_min_value=p2min;
    
    vvd_t coord_linear(vd_t(0.0,neq_moms),2);
    
    for(int i=0; i<range_max; i++)
    {
        coord_linear[0][i] = 1.0;  //costante
        coord_linear[1][i] = p2_tilde_eqmoms[i];   //p^2
    }
    
    vd_t jZq_out_par_ijack(0.0,2);
    vvd_t jZ_out_par_ijack(vd_t(0.0,2),nbil);
    
    double Zq_ave_cont=0.0, sqr_Zq_ave_cont=0.0, Zq_err_cont=0.0;
    vd_t Z_ave_cont(0.0,nbil), sqr_Z_ave_cont(0.0,nbil), Z_err_cont(0.0,nbil);
    
    for(int ijack=0; ijack<njacks; ijack++)
    {
        jZq_out_par_ijack=fit_continuum(coord_linear,Zq_err,jZq_out[ijack],range_min,range_max,p_min_value);
        
        Zq_ave_cont += jZq_out_par_ijack[0]/njacks;
        sqr_Zq_ave_cont += jZq_out_par_ijack[0]*jZq_out_par_ijack[0]/njacks;
        
        for(int ibil=0; ibil<nbil; ibil++)
        {
            jZ_out_par_ijack[ibil]=fit_continuum(coord_linear,Z_err[ibil],jZ_out[ibil][ijack],range_min,range_max,p_min_value);
        
            Z_ave_cont[ibil] += jZ_out_par_ijack[ibil][0]/njacks;
            sqr_Z_ave_cont[ibil] += jZ_out_par_ijack[ibil][0]*jZ_out_par_ijack[ibil][0]/njacks;
        }
    }
    
        Zq_err_cont=sqrt((double)(njacks-1))*sqrt(sqr_Zq_ave_cont-Zq_ave_cont*Zq_ave_cont);
        
        for(int ibil=0; ibil<nbil;ibil++)
            Z_err_cont[ibil]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_ave_cont[ibil]-Z_ave_cont[ibil]*Z_ave_cont[ibil]));
    
    cout<<"ZQ = "<<Zq_ave_cont<<" +/- "<<Zq_err_cont<<endl;
    
    vector<string> bil={"S","A","P","V","T"};
    
    for(int ibil=0; ibil<nbil;ibil++)
    {
        cout<<"Z"<<bil[ibil]<<" = "<<Z_ave_cont[ibil]<<" +/- "<<Z_err_cont[ibil]<<endl;
    }
    
//    vector<double> pert={-0.0695545,-0.100031,-0.118281,-0.130564,-0.108664};
//    
//    if(LO_or_EM==1)
//    {
//        cout<<"Z divided by the perturbative estimates (to be evolved in MSbar"
//    for(int ibil=0;i<nbil;ibil++)
//    {
//        cout<<"Z"<<bil[ibil]<<"(fact) = "<<A_bil[ibil]/pert[ibil]<<" +/- "<<A_err[ibil]/pert[ibil]<<endl;
//    }
//    }
    
    cout<<endl;
}

