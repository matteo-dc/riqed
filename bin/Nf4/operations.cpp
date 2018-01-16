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
#include <tuple>
#include "fit.hpp"
#include <stdio.h>
#include "subtraction.hpp"
#include "evolution.hpp"
#include "print.hpp"

#define EXTERN_OPER

#include "operations.hpp"

#include "vertices.hpp"

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
    
    set_moms();
    
    resize_vectors();
    
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
    compute_prop();
    compute_bil();
}

void oper_t::smom()
{
    ri_mom();
}


//////////

void oper_t::resize_vectors()
{
    jZq.resize(bilmoms.size());
    jZq_em.resize(bilmoms.size());
    
    jG_0.resize(bilmoms.size());
    jG_em.resize(bilmoms.size());
    
    jZ.resize(bilmoms.size());
    jZ_em.resize(bilmoms.size());
    
    jZq_ave_r.resize(bilmoms.size());
    jZq_em_ave_r.resize(bilmoms.size());
    
    m_eff_equivalent_Zq.resize(neq2);
    m_eff_equivalent.resize(neq);
    
    jZq_ave_r.resize(bilmoms.size());
    jZq_em_ave_r.resize(bilmoms.size());
    
    jG_0_ave_r.resize(bilmoms.size());
    jG_em_ave_r.resize(bilmoms.size());
    
    jZq_chir.resize(bilmoms.size());
    jZq_em_chir.resize(bilmoms.size());
    
    jG_0_chir.resize(bilmoms.size());
    jG_em_chir.resize(bilmoms.size());
    
    jZ_chir.resize(bilmoms.size());
    jZ_em_chir.resize(bilmoms.size());
    
//    cout<<"jZq "<<jZq.size()<<" "<<jZq_em.size()<<endl;
//    cout<<"jG "<<jG_0.size()<<" "<<jG_em.size()<<endl;
//    cout<<"jZ "<<jZ.size()<<" "<<jZ_em.size()<<endl;
//    cout<<"jG_ave_r "<<jG_0_ave_r.size()<<" "<<jG_em_ave_r.size()<<endl;
//    cout<<"jG_0_chir "<<jG_0_chir.size()<<" "<<jG_em_chir.size()<<endl;
    }


vvprop_t build_LO_prop(jprop_t &jS,vvvprop_t &S)
{
    vvprop_t S_0(vprop_t(prop_t::Zero(),nmr),njacks);
    
#pragma omp parallel for collapse(3)
    for(int m=0;m<nm;m++)
        for(int r=0;r<nr;r++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                int mr = r + nr*m;
                S_0[ijack][mr] = S[ijack][0][mr];
            }
    
    jS=clusterize(jS,S_0);
    
    return S_0;
}

vvprop_t build_EM_prop(jprop_t &jS,vvvprop_t &S)
{
    vvprop_t S_em(vprop_t(prop_t::Zero(),nmr),njacks);
    
#pragma omp parallel for collapse(3)
    for(int m=0;m<nm;m++)
        for(int r=0;r<nr;r++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                int mr = r + nr*m;
                
                // Electromagnetic correction:  S_em = S_self + S_tad -+ deltam_cr*S_P
                if(r==0) S_em[ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] + deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=0
                if(r==1) S_em[ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] - deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=1
            }
    
    jS=clusterize(jS,S_em);
    
    return S_em;
}

void oper_t::compute_prop()
{
    cout<<"Creating the propagators -- ";
    
    // array of input files to be read in a given conf
    ifstream input[combo];
    vector<string> v_path = setup_read_prop(input);
    
    for(int imom=0; imom<moms; imom++)
    {
        // definition of jackknifed propagators
        jprop_t jS_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jprop_t jS_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        
        // initialize propagators
        vvvprop_t S(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
        vvprop_t S_0(vprop_t(prop_t::Zero(),nmr),njacks);
        vvprop_t S_em(vprop_t(prop_t::Zero(),nmr),njacks);
        
//#pragma omp parallel for collapse(2)
        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
            for(int ihit=0;ihit<nhits;ihit++)
            {
                S=read_prop_mom(input,v_path,i_in_clust,ihit,imom);
                
                S_0 = build_LO_prop(jS_0,S);
                S_em = build_EM_prop(jS_em,S);
            }
        
        // jackknife average
        jS_0=jackknife(jS_0);
        jS_em=jackknife(jS_em);
        
        // invert propagator
        jprop_t jS_0_inv = invert_jprop(jS_0);
        jprop_t jS_em_inv = jS_0_inv*jS_em*jS_0_inv;
        
        // compute quark field RCs (Zq or Sigma1 established from input file!) and store
        jZq[imom] = compute_jZq(jS_0_inv,imom);
        jZq_em[imom] = - compute_jZq(jS_em_inv,imom);
        
        // printf("%lf\n",jZq[0][0]);
        
    } // close mom loop
}

void oper_t::compute_bil()
{
    cout<<"Creating the vertices -- ";
    
    // array of input files to be read in a given conf
    ifstream input[combo];
    vector<string> v_path = setup_read_prop(input);
    
    int mom_size = (int)bilmoms.size();
    
    for(int ibilmom=0;ibilmom<mom_size;ibilmom++)
    {
        cout<<"\r\t mom = "<<ibilmom+1<<"/"<<mom_size<<flush;
        
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        const bool read2=(imom1!=imom2);
        
        // definition of jackknifed propagators
        jprop_t jS1_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jprop_t jS1_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jprop_t jS2_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jprop_t jS2_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        
        // definition of vertices
        jvert_t jVert_0 (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        jvert_t jVert_em (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        
        // initialize propagators
        vvvprop_t S1(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
        vvprop_t S1_0(vprop_t(prop_t::Zero(),nmr),njacks);
        vvprop_t S1_em(vprop_t(prop_t::Zero(),nmr),njacks);
        vvvprop_t S2(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
        vvprop_t S2_0(vprop_t(prop_t::Zero(),nmr),njacks);
        vvprop_t S2_em(vprop_t(prop_t::Zero(),nmr),njacks);
        
#pragma omp parallel for collapse(2)
        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
            for(int ihit=0;ihit<nhits;ihit++)
            {
                S1=read_prop_mom(input,v_path,i_in_clust,ihit,imom1);
                S2=(read2)?read_prop_mom(input,v_path,i_in_clust,ihit,imom2):S1;
                
                S1_0 = build_LO_prop(jS1_0,S1);
                S1_em = build_EM_prop(jS1_em,S1);
                S2_0 = build_LO_prop(jS2_0,S2);
                S2_em = build_EM_prop(jS2_em,S2);
                               
                jVert_0 = build_LO_vert(S1,S2,jVert_0);
                jVert_em = build_EM_vert(S1,S2,S1_em,S2_em,jVert_em);
            }
        
        // jackknife averages
        jS1_0=jackknife(jS1_0);
        jS1_em=jackknife(jS1_em);
        jS2_0=jackknife(jS2_0);
        jS2_em=jackknife(jS2_em);
        
        jVert_0=jackknife(jVert_0);
        jVert_em=jackknife(jVert_em);
        
        // invert propagators
        jprop_t jS1_0_inv = invert_jprop(jS1_0);
        jprop_t jS1_em_inv = jS1_0_inv*jS1_em*jS1_0_inv;
        jprop_t jS2_0_inv = invert_jprop(jS2_0);
        jprop_t jS2_em_inv = jS2_0_inv*jS2_em*jS2_0_inv;
        
        // compute the projected green function (S,V,P,A,T)
        jproj_t jG_0_mom = compute_pr_bil(jS1_0_inv, jVert_0, jS2_0_inv);
        jproj_t jG_1 = compute_pr_bil(jS1_0_inv, jVert_em, jS2_0_inv);
        jproj_t jG_a = compute_pr_bil(jS1_em_inv, jVert_0, jS2_0_inv);
        jproj_t jG_b = compute_pr_bil(jS1_0_inv, jVert_0, jS2_em_inv);
        
        jproj_t jG_em_mom = -jG_1+jG_a+jG_b;
        
        jG_0[imom1]=jG_0_mom;
        jG_em[imom1]=jG_em_mom;
        
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
        jZbil_t jZ_mom(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
        jZbil_t jZ_em_mom(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
        
        //compute Z's according to 'riqed.pdf', one for each momentum
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                    for(int k=0;k<5;k++)
                    {
                        jZ_mom[ijack][mr_fw][mr_bw][k] = sqrt(jZq[imom1][ijack][mr_fw])*sqrt(jZq[imom2][ijack][mr_bw])/jG_0[imom1][ijack][mr_fw][mr_bw][k];
                        
                        jZ_em_mom[ijack][mr_fw][mr_bw][k] = jG_em[imom1][ijack][mr_fw][mr_bw][k]/jG_0[imom1][ijack][mr_fw][mr_bw][k] + 0.5*(jZq_em[imom1][ijack][mr_fw]/jZq[imom1][ijack][mr_fw] + jZq_em[imom2][ijack][mr_bw]/jZq[imom2][ijack][mr_bw]);
                    }
        
        jZ[imom1]=jZ_mom;
        jZ_em[imom1]=jZ_em_mom;
        
    }// close mom loop
}

oper_t oper_t::average_r(/*const bool recompute_Zbil*/)
{
    cout<<"Averaging over r"<<endl<<endl;
    
    oper_t out;
    
    out.bilmoms=bilmoms;
    out.Zbil_computed=Zbil_computed;
    out.jZq=jZq;
    out.jZq_em=jZq_em;
    out.jG_0=jG_0;
    out.jG_em=jG_em;
    out.jZ=jZ;
    out.jZ_em=jZ_em;
    out.jG_0_chir=jG_0_chir;
    out.jG_em_chir=jG_em_chir;
    out.jZq_chir=jZq_chir;
    out.jZq_em_chir=jZq_chir;
    out.jZ_chir=jZ_chir;
    out.jZ_em_chir=jZ_em_chir;
  
    for(int ieq=0;ieq<neq2;ieq++) m_eff_equivalent_Zq[ieq]=0.0;
    for(int ieq=0;ieq<neq;ieq++)  m_eff_equivalent[ieq]=0.0;

    for(size_t ibilmom=0;ibilmom<bilmoms.size();ibilmom++)
    {
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        vvd_t jZq_ave_r_mom(vd_t(0.0,neq2),njacks);
        vvd_t jZq_em_ave_r_mom(vd_t(0.0,neq2),njacks);
        
        int ieq=0;
        for(int m=0; m<nm; m++)
            for(int r=0; r<nr; r++)
            {
                ieq=m;
                
                // masses
                if(UseEffMass==1 and ibilmom==0)
                    m_eff_equivalent_Zq[ieq] += eff_mass[r+nr*m][r+nr*m]/nr;
                else if(UseEffMass==0 and ibilmom==0 and r==0)
                    m_eff_equivalent_Zq[ieq] = mass_val[ieq];
                
                //LO
                for(int ijack=0;ijack<njacks;ijack++) jZq_ave_r_mom[ijack][ieq] += jZq[imom1][ijack][r+nr*m]/nr;
                //EM
                for(int ijack=0;ijack<njacks;ijack++) jZq_em_ave_r_mom[ijack][ieq] += jZq_em[imom1][ijack][r+nr*m]/nr;
            }
        
        jZq_ave_r[imom1]=jZq_ave_r_mom;
        jZq_em_ave_r[imom1]=jZq_em_ave_r_mom;
        
        jproj_t jG_0_mom  = jG_0[imom1];
        jproj_t jG_em_mom = jG_em[imom1];
        
        vvvd_t jG_0_ave_r_mom(vvd_t(vd_t(0.0,neq),njacks),5);
        vvvd_t jG_em_ave_r_mom(vvd_t(vd_t(0.0,neq),njacks),5);
        
        ieq=0;
        
        for(int mA=0; mA<nm; mA++)
            for(int mB=mA; mB<nm; mB++)
            {
                for(int r=0; r<nr; r++)
                {
                    // masses
                    if(UseEffMass==1 and ibilmom==0)
                        m_eff_equivalent[ieq] += (eff_mass[r+nr*mA][r+nr*mB]+eff_mass[r+nr*mB][r+nr*mA])/(2.0*nr);
                    else if(UseEffMass==0 and ibilmom==0 and r==0)
                        m_eff_equivalent[ieq] = mass_val[mA] + mass_val[mB];
                    
                    //LO
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int ibil=0; ibil<5; ibil++)
                            jG_0_ave_r_mom[ibil][ijack][ieq] += (jG_0_mom[ijack][r+nr*mA][r+nr*mB][ibil]+jG_0_mom[ijack][r+nr*mB][r+nr*mA][ibil])/(2.0*nr);
                    //EM
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int ibil=0; ibil<5; ibil++)
                            jG_em_ave_r_mom[ibil][ijack][ieq] += (jG_em_mom[ijack][r+nr*mA][r+nr*mB][ibil]+jG_em_mom[ijack][r+nr*mB][r+nr*mA][ibil])/(2.0*nr);
                }
                ieq++;
            }
        
        jG_0_ave_r[imom1]=jG_0_ave_r_mom;
        jG_em_ave_r[imom1]=jG_em_ave_r_mom;
    }
    
    out.m_eff_equivalent_Zq=m_eff_equivalent_Zq;
    out.m_eff_equivalent=m_eff_equivalent;
    
    out.jZq_ave_r=jZq_ave_r;
    out.jZq_em_ave_r=jZq_em_ave_r;
    out.jG_0_ave_r=jG_0_ave_r;
    out.jG_em_ave_r=jG_em_ave_r;
    
    return out;
}
    
tuple<vvvd_t,vvvd_t> ave_err(vector<vvvd_t> jG)
{
    vvvd_t G_ave(vvd_t(vd_t(0.0,neq),5),jG.size());
    vvvd_t sqr_G_ave(vvd_t(vd_t(0.0,neq),5),jG.size());
    vvvd_t G_err(vvd_t(vd_t(0.0,neq),5),jG.size());
    
//#pragma omp parallel for collapse(4)
    for(size_t imom=0;imom<jG.size();imom++)
        for(int ieq=0;ieq<neq;ieq++)
            for(int ibil=0;ibil<5;ibil++)
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    G_ave[imom][ibil][ieq]+=jG[imom][ibil][ijack][ieq]/njacks;
                    sqr_G_ave[imom][ibil][ieq]+=jG[imom][ibil][ijack][ieq]*jG[imom][ibil][ijack][ieq]/njacks;
                }
#pragma omp parallel for collapse(3)
    for(size_t imom=0;imom<jG.size();imom++)
        for(int ieq=0;ieq<neq;ieq++)
            for(int ibil=0;ibil<5;ibil++)
                G_err[imom][ibil][ieq]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_G_ave[imom][ibil][ieq]-G_ave[imom][ibil][ieq]*G_ave[imom][ibil][ieq]));
    
    tuple<vvvd_t,vvvd_t> tuple_ave_err(G_ave,G_err);
    
    return tuple_ave_err;
}

tuple<vvd_t,vvd_t> ave_err(vector<vvd_t> jG)
{
    vvd_t G_ave(vd_t(0.0,5),jG.size());
    vvd_t sqr_G_ave(vd_t(0.0,5),jG.size());
    vvd_t G_err(vd_t(0.0,5),jG.size());
    
    //#pragma omp parallel for collapse(4)
    for(size_t imom=0;imom<jG.size();imom++)
        for(int ibil=0;ibil<5;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                G_ave[imom][ibil]+=jG[imom][ibil][ijack]/njacks;
                sqr_G_ave[imom][ibil]+=jG[imom][ibil][ijack]*jG[imom][ibil][ijack]/njacks;
            }
#pragma omp parallel for collapse(2)
    for(size_t imom=0;imom<jG.size();imom++)
        for(int ibil=0;ibil<5;ibil++)
            G_err[imom][ibil]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_G_ave[imom][ibil]-G_ave[imom][ibil]*G_ave[imom][ibil]));
    
    tuple<vvd_t,vvd_t> tuple_ave_err(G_ave,G_err);
    
    return tuple_ave_err;
}

tuple<vvd_t,vvd_t> ave_err_q(vector<vvd_t> jZq)
{
    vvd_t Zq_ave(vd_t(0.0,neq2),jZq.size());
    vvd_t sqr_Zq_ave(vd_t(0.0,neq2),jZq.size());
    vvd_t Zq_err(vd_t(0.0,neq2),jZq.size());
    
//#pragma omp parallel for collapse(3)
    for(size_t imom=0;imom<jZq.size();imom++)
        for(int ieq=0;ieq<neq2;ieq++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                Zq_ave[imom][ieq]+=jZq[imom][ijack][ieq]/njacks;
                sqr_Zq_ave[imom][ieq]+=jZq[imom][ijack][ieq]*jZq[imom][ijack][ieq]/njacks;
            }
#pragma omp parallel for collapse(2)
    for(size_t imom=0;imom<jZq.size();imom++)
        for(int ieq=0;ieq<neq2;ieq++)
            Zq_err[imom][ieq]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[imom][ieq]-Zq_ave[imom][ieq]*Zq_ave[imom][ieq]));
    
    tuple<vvd_t,vvd_t> tuple_ave_err(Zq_ave,Zq_err);
    
    return tuple_ave_err;
}

tuple<vd_t,vd_t> ave_err_q(vector<vd_t> jZq)
{
    vd_t Zq_ave(0.0,jZq.size());
    vd_t sqr_Zq_ave(0.0,jZq.size());
    vd_t Zq_err(0.0,jZq.size());
    
    //#pragma omp parallel for collapse(3)
    for(size_t imom=0;imom<jZq.size();imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            {
                Zq_ave[imom]+=jZq[imom][ijack]/njacks;
                sqr_Zq_ave[imom]+=jZq[imom][ijack]*jZq[imom][ijack]/njacks;
            }
#pragma omp parallel for
    for(size_t imom=0;imom<jZq.size();imom++)
            Zq_err[imom]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[imom]-Zq_ave[imom]*Zq_ave[imom]));
    
    tuple<vd_t,vd_t> tuple_ave_err(Zq_ave,Zq_err);
    
    return tuple_ave_err;
}

oper_t oper_t::chiral_extr()
{
    cout<<"Chiral extrapolation"<<endl<<endl;
    
    oper_t out;
    
    out.bilmoms=bilmoms;
    out.Zbil_computed=Zbil_computed;
    out.jZq=jZq;
    out.jZq_em=jZq_em;
    out.jG_0=jG_0;
    out.jG_em=jG_em;
    out.jZ=jZ;
    out.jZ_em=jZ_em;
    out.m_eff_equivalent_Zq=m_eff_equivalent_Zq;
    out.m_eff_equivalent=m_eff_equivalent;
    out.jZq_ave_r=jZq_ave_r;
    out.jZq_em_ave_r=jZq_em_ave_r;
    
    vvvd_t G_0_err = get<1>(ave_err(jG_0_ave_r));
    vvvd_t G_em_err = get<1>(ave_err(jG_em_ave_r));
    
    vvd_t Zq_err = get<1>(ave_err_q(jZq_ave_r));
    vvd_t Zq_em_err = get<1>(ave_err_q(jZq_em_ave_r));
    
    //Sum of quark masses for the extrapolation
//    vd_t mass_sum(0.0,10);
//    int i_sum = 0;
//    for (int i=0; i<nm; i++)
//        for(int j=i;j<nm;j++)
//        {
//            mass_sum[i_sum] = mass_val[i]+mass_val[j];
//            i_sum++;
//        }
    
    // coords for fit bilinears
    vvd_t coord_bil(vd_t(0.0,neq),3);
    vvd_t coord_q(vd_t(0.0,neq2),2);
    for(int i=0; i<neq; i++)
    {
        coord_bil[0][i] = 1.0;  //costante
        if(UseEffMass==1)
        {
            coord_bil[1][i] = m_eff_equivalent[i]*m_eff_equivalent[i];   //M^2
            coord_bil[2][i] = 1.0/(m_eff_equivalent[i]*m_eff_equivalent[i]);  //1/M^2
        }
        else if(UseEffMass==0)
        {
            coord_bil[1][i] = m_eff_equivalent[i];        // (am1+am2)
            coord_bil[2][i] = 1.0/m_eff_equivalent[i];    // 1/(am1+am2)
        }
    }
    for(int i=0; i<neq2; i++)
    {
        coord_q[0][i] = 1.0;  //costante
        if(UseEffMass==1)
            coord_q[1][i] = m_eff_equivalent_Zq[i]*m_eff_equivalent_Zq[i];   //M^2
        else if(UseEffMass==0)
            coord_q[1][i] = m_eff_equivalent_Zq[i];        // (am1+am2)
    }
    
    // range for fit bilinears
    int t_min=0;
    int t_max=neq-1;
    
    //range for fit Zq
    int t_min_q=0;
    int t_max_q=neq2-1;
    
    // number of fit parameters for bilinears
    int npar[5]={3,2,3,2,2};
    
    vvd_t jG_0_chir_mom(vd_t(0.0,njacks),5), jG_em_chir_mom(vd_t(0.0,njacks),5);
    vvvd_t jG_0_pars_mom(vvd_t(vd_t(0.0,3),njacks),5);
    vvvd_t jG_em_pars_mom(vvd_t(vd_t(0.0,3),njacks),5);
    
    vd_t jZq_chir_mom(0.0,njacks), jZq_em_chir_mom(0.0,njacks);
    vvd_t jZq_0_pars_mom(vd_t(0.0,2),njacks);
    vvd_t jZq_em_pars_mom(vd_t(0.0,2),njacks);
    
    vvd_t jZ_chir_mom(vd_t(0.0,njacks),5), jZ_em_chir_mom(vd_t(0.0,njacks),5);
    
    for(size_t imom=0;imom<bilmoms.size();imom++)
    {
        // Bilinears
        for(int ibil=0;ibil<5;ibil++)
        {
            jG_0_pars_mom[ibil]=fit_par_jackknife(coord_bil,npar[ibil],G_0_err[imom][ibil],jG_0_ave_r[imom][ibil],t_min,t_max);
            jG_em_pars_mom[ibil]=fit_par_jackknife(coord_bil,npar[ibil],G_em_err[imom][ibil],jG_em_ave_r[imom][ibil],t_min,t_max);
            
            for(int ijack=0;ijack<njacks;ijack++)
            {
                if(ibil==0 or ibil==2)
                    for(int ieq=0;ieq<neq;ieq++)
                    {
                        // Goldstone pole subtraction from bilinears
                        jG_0_ave_r[imom][ibil][ijack][ieq] -= jG_0_pars_mom[ibil][ijack][2];
                        jG_em_ave_r[imom][ibil][ijack][ieq] -= jG_em_pars_mom[ibil][ijack][2];
                    }
                
                // extrapolated value
                jG_0_chir_mom[ibil][ijack] = jG_0_pars_mom[ibil][ijack][0];
                jG_em_chir_mom[ibil][ijack] = jG_em_pars_mom[ibil][ijack][0];
            }
        }
        
        // Zq
        jZq_0_pars_mom=fit_par_jackknife(coord_q,2,Zq_err[imom],jZq_ave_r[imom],t_min_q,t_max_q);
        jZq_em_pars_mom=fit_par_jackknife(coord_q,2,Zq_em_err[imom],jZq_em_ave_r[imom],t_min_q,t_max_q);
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZq_chir_mom[ijack] = jZq_0_pars_mom[ijack][0];
            jZq_em_chir_mom[ijack] = jZq_em_pars_mom[ijack][0];
        }
        
        // pushback of chiral bils and Zq
        jG_0_chir[imom]=jG_0_chir_mom;
        jG_em_chir[imom]=jG_em_chir_mom;
        jZq_chir[imom]=jZq_chir_mom;
        jZq_em_chir[imom]=jZq_em_chir_mom;
        
        // constructing Z_chiral for each momentum
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int ibil=0;ibil<5;ibil++)
            {
                //LO
                jZ_chir_mom[ibil][ijack] = jZq_chir_mom[ijack]/jG_0_chir_mom[ibil][ijack];
                //EM
                jZ_em_chir_mom[ibil][ijack] = jG_em_chir_mom[ibil][ijack]/jG_0_chir_mom[ibil][ijack] + jZq_em_chir_mom[ijack]/jZq_chir_mom[ijack];
            }
        
        jZ_chir[imom]=jZ_chir_mom;
        jZ_em_chir[imom]=jZ_em_chir_mom;
    }
    
    out.jG_0_ave_r = jG_0_ave_r;
    out.jG_em_ave_r = jG_em_ave_r;
    
    out.jG_0_chir = jG_0_chir;
    out.jG_em_chir = jG_em_chir;
    out.jZq_chir = jZq_chir;
    out.jZq_em_chir = jZq_em_chir;
    out.jZ_chir = jZ_chir;
    out.jZ_em_chir = jZ_em_chir;
    
    return out;
}

oper_t oper_t::subtract()
{
    cout<<"Subtracting the O(a2) effects"<<endl<<endl;
    
    oper_t out;
    
    out.bilmoms=bilmoms;
    out.Zbil_computed=Zbil_computed;
    out.jZq=jZq;
    out.jZq_em=jZq_em;
    out.jG_0=jG_0;
    out.jG_em=jG_em;
    out.jZ=jZ;
    out.jZ_em=jZ_em;
    out.m_eff_equivalent_Zq=m_eff_equivalent_Zq;
    out.m_eff_equivalent=m_eff_equivalent;
    out.jZq_ave_r=jZq_ave_r;
    out.jZq_em_ave_r=jZq_em_ave_r;
    out.jZq_chir = jZq_chir;
    out.jZq_em_chir = jZq_em_chir;
    out.jZ_chir = jZ_chir;
    out.jZ_em_chir = jZ_em_chir;
    
    jG_0_sub = jG_0_chir;
    jG_em_sub = jG_em_chir;
    jZq_sub = jZq_chir;
    jZq_em_sub = jZq_em_chir;
    
    jZ_sub = jZ_chir;
    jZ_em_sub = jZ_em_chir;
    
#pragma omp parallel for collapse(2)
    for(size_t imom=0;imom<bilmoms.size();imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZq_sub[imom][ijack] = jZq_chir[imom][ijack] - subtraction_q(imom,LO);
            jZq_em_sub[imom][ijack] = jZq_em_chir[imom][ijack] + /*(!)*/ subtraction_q(imom,EM)*jZq_chir[imom][ijack];
            // N.B.: the subtraction gets an extra minus sign due to the definition of the e.m. expansion!
            
            for(int ibil=0;ibil<5;ibil++)
            {
                jG_0_sub[imom][ibil][ijack] = jG_0_chir[imom][ibil][ijack] - subtraction(imom,ibil,LO);
                jG_em_sub[imom][ibil][ijack] = jG_em_chir[imom][ibil][ijack] - subtraction(imom,ibil,EM)*jG_0_chir[imom][ibil][ijack];
                
                // constructing Z_sub
                jZ_sub[imom][ibil][ijack] = jZq_sub[imom][ijack]/jG_0_sub[imom][ibil][ijack];
                jZ_em_sub[imom][ibil][ijack] = jG_em_sub[imom][ibil][ijack]/jG_0_sub[imom][ibil][ijack] + jZq_em_sub[imom][ijack]/jZq_sub[imom][ijack];
            }
        }

    out.jG_0_sub = jG_0_sub;
    out.jG_em_sub = jG_em_sub;
    out.jZq_sub = jZq_sub;
    out.jZq_em_sub = jZq_em_sub;
    
    out.jZ_sub = jZ_sub;
    out.jZ_em_sub = jZ_em_sub;
    
    return out;
}

oper_t oper_t::evolve()
{
    cout<<"Evolving the Z's to the scale 1/a"<<endl<<endl;
    
    oper_t out;
    
    out.bilmoms=bilmoms;
    out.Zbil_computed=Zbil_computed;
    out.jZq_sub = jZq_sub;
    out.jZq_em_sub = jZq_em_sub;
    out.jZ_sub = jZ_sub;
    out.jZ_em_sub = jZ_em_sub;
    
//    jZq_evo = jZq_sub;
//    jZq_em_evo = jZq_em_sub;
//    jZ_evo = jZ_sub;
//    jZ_em_evo = jZ_em_sub;
    
    vector<vd_t> jZq_evo_tmp(jZq_sub.size(),vd_t(0.0,njacks)), jZq_em_evo_tmp(jZq_sub.size(),vd_t(0.0,njacks));
    vector<vvd_t> jZ_evo_tmp(jZ_sub.size(),vvd_t(vd_t(0.0,njacks),nbil)), jZ_em_evo_tmp(jZ_sub.size(),vvd_t(vd_t(0.0,njacks),nbil));

    double cq=0.0;
    vd_t cO(0.0,5);
    
//#pragma omp parallel for
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
        {
            jZq_evo_tmp[imom][ijack] = jZq_sub[imom][ijack]/cq;
            jZq_em_evo_tmp[imom][ijack] = jZq_em_sub[imom][ijack]/cq;
            
            for(int ibil=0;ibil<5;ibil++)
            {
                jZ_evo_tmp[imom][ibil][ijack] = jZ_sub[imom][ibil][ijack]/cO[ibil];
                jZ_em_evo_tmp[imom][ibil][ijack] = jZ_em_sub[imom][ibil][ijack]/cO[ibil];
            }
        }
    }
    
    out.jZq_evo = jZq_evo_tmp;
    out.jZq_em_evo = jZq_em_evo_tmp;
    out.jZ_evo = jZ_evo_tmp;
    out.jZ_em_evo = jZ_em_evo_tmp;
    
    return out;
}

double mom_xyz(size_t imom)
{
    return abs(mom_list[imom][1])*abs(mom_list[imom][2])*abs(mom_list[imom][3]);
}

oper_t oper_t::average_equiv_moms()
{
    cout<<"Averaging over the equivalent momenta -- "<<endl<<endl;
    
    oper_t out;
    
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
    
    out.jZq_evo = jZq_evo;
    out.jZq_em_evo = jZq_em_evo;
    out.jZ_evo = jZ_evo;
    out.jZ_em_evo = jZ_em_evo;

    (out.jZq_evo).resize(neq_moms);
    (out.jZq_em_evo).resize(neq_moms);
    (out.jZ_evo).resize(neq_moms);
    (out.jZ_em_evo).resize(neq_moms);
    
    // initialize to zero
#pragma omp parallel for collapse(2)
    for(int tag=0;tag<neq_moms;tag++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            (out.jZq_evo)[tag][ijack]=0.0;
            (out.jZq_em_evo)[tag][ijack]=0.0;
            for(int ibil=0;ibil<5;ibil++)
            {
                (out.jZ_evo)[tag][ibil][ijack]=0.0;
                (out.jZ_em_evo)[tag][ibil][ijack]=0.0;
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
                {
                    (out.jZq_evo)[tag][ijack]+=jZq_evo[imom][ijack]/count_tag_vector[tag];
                    (out.jZq_em_evo)[tag][ijack]+=jZq_em_evo[imom][ijack]/count_tag_vector[tag];
                    for(int ibil=0;ibil<5;ibil++)
                    {
                        (out.jZ_evo)[tag][ibil][ijack]+=jZ_evo[imom][ibil][ijack]/count_tag_vector[tag];
                        (out.jZ_em_evo)[tag][ibil][ijack]+=jZ_em_evo[imom][ibil][ijack]/count_tag_vector[tag];
                    }
                }
            }
        }
    
    return out;
}

//! To be used after the average over the equivalent momenta! (*)
void continuum_limit(oper_t out, const int LO_or_EM)
{
    vector<vd_t> jZq_out;
    vector<vvd_t> jZ_out;
    
    if(LO_or_EM==0)
    {
        cout<<"Continuum limit extrapolation:"<<endl<<endl;
        jZq_out = out.jZq_evo;
        jZ_out = out.jZ_evo;
        cout<<"-- Leading Order --"<<endl;
    }
    else if(LO_or_EM==1)
    {
        cout<<"Continuum limit extrapolation:"<<endl<<endl;
        jZq_out = out.jZq_em_evo;
        jZ_out = out.jZ_em_evo;
        cout<<"-- EM Correction --"<<endl;
    }
    
    //! (*)
    int neq_moms = jZq_out.size();
    vector<double> p2_tilde_eqmoms(neq_moms);
    READ(p2_tilde_eqmoms);
    
    vd_t Zq_err = get<1>(ave_err_q(jZq_out));
    vvd_t Z_err = get<1>(ave_err(jZ_out));
    
    //linear fit
    int p2_min=0;  //a2p2~1
    int p2_max=neq_moms;
    int p_min_value=0.9;
    
    vvd_t coord_linear(vd_t(0.0,neq_moms),2);
    
    for(int i=0; i<p2_max; i++)
    {
        coord_linear[0][i] = 1.0;  //costante
        coord_linear[1][i] = p2_tilde_eqmoms[i];   //p^2
    }
    
    vXd_t jZq_out_par=fit_chiral_jackknife(coord_linear,Zq_err,jZq_out,p2_min,p2_max,p_min_value);
    vvXd_t jZ_out_par=fit_chiral_jackknife(coord_linear,Z_err,jZ_out,p2_min,p2_max,p_min_value);
    
    int pars=jZq_out_par[0].size();
    
    vd_t Zq_par_ave(0.0,pars), sqr_Zq_par_ave(0.0,pars), Zq_par_err(0.0,pars);
    vvd_t Zq_par_ave_err(vd_t(0.0,pars),2);
    
    vvd_t Z_par_ave(vd_t(0.0,pars),nbil), sqr_Z_par_ave(vd_t(0.0,pars),nbil), Z_par_err(vd_t(0.0,pars),nbil);
    vvvd_t Z_par_ave_err(vvd_t(vd_t(0.0,pars),nbil),2);
    
    for(int ipar=0;ipar<pars;ipar++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            Zq_par_ave[ipar]+=jZq_out_par[ijack](ipar)/njacks;
            sqr_Zq_par_ave[ipar]+=jZq_out_par[ijack](ipar)*jZq_out_par[ijack](ipar)/njacks;
            
            for(int ibil=0; ibil<nbil;ibil++)
            {
                Z_par_ave[ibil][ipar]+=jZ_out_par[ibil][ijack](ipar)/njacks;
                sqr_Z_par_ave[ibil][ipar]+=jZ_out_par[ibil][ijack](ipar)*jZ_out_par[ibil][ijack](ipar)/njacks;
            }
        }
    
    for(int ipar=0;ipar<pars;ipar++)
    {
        Zq_par_err[ipar]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_par_ave[ipar]-Zq_par_ave[ipar]*Zq_par_ave[ipar]);
        
        for(int ibil=0; ibil<nbil;ibil++)
            Z_par_err[ibil][ipar]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_par_ave[ibil][ipar]-Z_par_ave[ibil][ipar]*Z_par_ave[ibil][ipar]));
    }
    
    Zq_par_ave_err[0]=Zq_par_ave; //Zq_par_ave_err[ave/err][par]
    Zq_par_ave_err[1]=Zq_par_err;
    Z_par_ave_err[0]=Z_par_ave; //Z_par_ave_err[ave/err][ibil][par]
    Z_par_ave_err[1]=Z_par_err;
    
    double A=Zq_par_ave_err[0][0];          //intercept
    double A_err=Zq_par_ave_err[1][0];
    //    double B=Zq_par_ave_err[0][1];          //slope
    //    double B_err=Zq_par_ave_err[1][1];
    
    cout<<"ZQ = "<<A<<" +/- "<<A_err<<endl;
    
    vd_t A_bil(0.0,nbil),A_bil_err(0.0,nbil);
    vd_t B_bil(0.0,nbil),B_bil_err(0.0,nbil);
    
    vector<string> bil={"S","A","P","V","T"};
    
    for(int ibil=0; ibil<nbil;ibil++)
    {
        A_bil[ibil]=Z_par_ave_err[0][ibil][0];
        A_bil_err[ibil]=Z_par_ave_err[1][ibil][0];
        B_bil[ibil]=Z_par_ave_err[0][ibil][1];
        B_bil_err[ibil]=Z_par_ave_err[1][ibil][1];
        
        cout<<"Z"<<bil[ibil]<<" = "<<A_bil[ibil]<<" +/- "<<A_bil_err[ibil]<<endl;
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

