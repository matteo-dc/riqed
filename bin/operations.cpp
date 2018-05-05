#include "aliases.hpp"
#include "global.hpp"
#include "read.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include "fit.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "subtraction.hpp"
#include "evolution.hpp"
#include "print.hpp"
#include "ave_err.hpp"
#include "meslep.hpp"
#include "vertices.hpp"
#include <chrono>
#include "prop.hpp"
#include "sigmas.hpp"

#define EXTERN_OPER
 #include "operations.hpp"

using namespace std::chrono;

SCHEME_t get_scheme()
{
    if(scheme=="RI_MOM") return RI_MOM;
    if(scheme=="SMOM") return SMOM;
    
    return ERR;
}

void oper_t::set_ins()
{
    sigma::set_ins();
    jprop::set_ins();
    qprop::set_ins();
    lprop::set_ins();
    gbil::set_ins();
    pr_meslep::set_ins();
}

void oper_t::set_moms()
{
    //read mom list
    read_mom_list(path_to_moms);
    moms=mom_list.size();
    cout<<"Read: "<<moms<<" momenta from \""<<mom_path<<"\" (BC: "<<BC<<")."<<endl<<endl;
    
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
    _linmoms=linmoms.size();
    _bilmoms=bilmoms.size();
    _meslepmoms=meslepmoms.size();
//    moms=_linmoms;
    
    print_vec(p2_tilde,path_print+"p2_tilde.txt");
    print_vec(p2,path_print+"p2.txt");
}

void oper_t::set_ri_mom_moms()
{
    linmoms.resize(moms);
    bilmoms.resize(moms);
    meslepmoms.resize(moms);
    
#pragma omp parallel for
    for(int imom=0;imom<moms;imom++)
        if(filt_moms[imom])
        {
            linmoms[imom]={imom};
            bilmoms[imom]={imom,imom,imom};
            meslepmoms[imom]=bilmoms[imom];
        }
}

void oper_t::set_smom_moms()
{
    // http://xxx.lanl.gov/pdf/0901.2599v2 (Sturm et al.)
    
    linmoms.clear();
    bilmoms.clear();
    
    double eps=1e-10;
    
    // SMOM not yet implemented for 4fermions
    if(compute_4f)
    {
        cout<<" meslepmoms not initialized for SMOM."<<endl;
        exit(0);
    }
    
#pragma omp parallel for
    for(int i=0;i<moms;i++)
        if(filt_moms[i])
            for(int j=0;j<moms;j++)
                if(filt_moms[j])
                {
                    if(2.0*fabs(p2[i]-p2[j])<(p2[i]+p2[j])*eps)
                    {
                        coords_t momk;
                        
                        p_t k_array, k_tilde_array;
                        double k_sqr=0.0, k_tilde_sqr=0.0;
                        double k_4=0.0, k_tilde_4=0.0;
                        
                        for(size_t mu=0;mu<4;mu++)
                        {
                            momk[mu]=mom_list[i][mu]-mom_list[j][mu];
                            
                            k_array[mu]=2*M_PI*momk[mu]/size[mu];
                            k_sqr+=k_array[mu]*k_array[mu];
                            k_4+=k_array[mu]*k_array[mu]*k_array[mu]*k_array[mu];

                            k_tilde_array[mu]=sin(k_array[mu]);
                            k_tilde_sqr+=k_tilde_array[mu]*k_tilde_array[mu];
                            k_tilde_4+=k_tilde_array[mu]*k_tilde_array[mu]*k_tilde_array[mu]*k_tilde_array[mu];
                        }
                        
                        if(2.0*fabs(p2[i]-k_sqr)<(p2[i]+k_sqr)*eps)
                        {
                            //search in mom_list
                            auto posk = find(mom_list.begin(),mom_list.end(),momk);
                            
                            //if not found, push into mom_list
                            if(posk==mom_list.end())
                            {
                                posk=mom_list.end();
                                
                                mom_list.push_back(momk);
                                p.push_back(k_array);
                                p_tilde.push_back(k_tilde_array);
                                p2.push_back(k_sqr);
                                p2_tilde.push_back(k_tilde_sqr);
                                p4.push_back(k_4);
                                p4_tilde.push_back(k_tilde_4);
                            }
                            
                            const int k=distance(mom_list.begin(),posk);
                            
                            vector<int> pos;
                            
                            //search in the linmoms: if found take the distance, otherwise add
                            for(const int ic : {i,j})
                            {
                                cout<<"searching for "<<ic<<endl;
                                auto pos_ic=find(linmoms.begin(),linmoms.end(),array<int,1>{ic});
                                size_t d;
                                if(pos_ic==linmoms.end())
                                {
                                    //the position will be the end
                                    d=linmoms.size();
                                    //include it
                                    linmoms.push_back({ic});
                                    
                                    cout<<" not found"<<endl;
                                }
                                else
                                {
                                    d=distance(linmoms.begin(),pos_ic);
                                    cout<<" found"<<endl;
                                }
                                
                                //add to the list
                                cout<<"Position: "<<d<<endl;
                                pos.push_back(d);
                            }
                            
                            //store
                            bilmoms.push_back({k,pos[0],pos[1]});
                            
                        } else cout<<"p2-k2 != 0"<<endl;
                    } else cout<<"p1^2-p2^2 != 0"<<endl;
                }
}

////////

void oper_t::create_basic(const int b, const int th, const int msea)
{
//    step = "basic";
    
    _beta=beta[b];
    _beta_label=beta_label[b];
    _nm_Sea=nm_Sea[b];
    _SeaMasses_label=to_string(SeaMasses_label[b][msea]);
    _theta_label=theta_label[th];
    
    if(strcmp(analysis.c_str(),"inte")==0)
    {
        path_to_beta = path_ensemble + _beta_label + "_b" + to_string_with_precision(_beta,2) + "/";
        ensemble_name = _beta_label + _SeaMasses_label + _theta_label;
        path_to_ens =  path_to_beta + ensemble_name + "/";
    }
    else if(strcmp(analysis.c_str(),"free")==0)
    {
        ensemble_name = _beta_label + _SeaMasses_label + _theta_label;
        path_to_ens = path_ensemble + ensemble_name + "/";
    }
    
    read_input(path_to_ens,ensemble_name);
    path_to_moms = path_to_ens + mom_path;
    
    path_print = path_to_ens+"print/";
   
    V=size[0]*size[1]*size[2]*size[3];
 
    _nm=nm;
    _nr=nr;
    _nmr=_nm*_nr;
    
    g2=6.0/_beta;
    g2_tilde=g2/plaquette;
    
    set_moms();
    
    set_ins();
    
    allocate();
    
    if(compute_mpcac) compute_mPCAC("");
    if(compute_mpcac) compute_mPCAC("sea");
    if(UseEffMass) eff_mass=read_eff_mass(path_to_ens+"eff_mass_array");
    if(UseEffMass) eff_mass_time=read_eff_mass_time(path_to_ens+"eff_mass_array_time");
    if(UseEffMass and _nm_Sea>0) eff_mass_sea=read_eff_mass_sea(path_to_ens+"eff_mass_sea_array");
    
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
    
    // read or compute deltam
    deltam_computed=false;
    compute_deltam_from_prop();
//    deltam_cr = read_deltam(path_to_ens,"deltam_cr_array");
//    deltamu   = read_deltam(path_to_ens,"deltamu_array");
    
    compute_Zq();
    compute_Zbil();
    if(compute_4f) compute_Z4f();
    
}

//////////

//oper_t oper_t::average_r(/*const bool recompute_Zbil*/)
//{
//    cout<<"Averaging over r"<<endl<<endl;
//    
//    oper_t out=(*this);
//    
//    out._nr=1;
//    out._nm=_nm;
//    out._nmr=(out._nm)*(out._nr);
//    
//    out.allocate();
//    
//    if(UseEffMass==1)
//    {
//        vvvd_t eff_mass_temp(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks);
//        
//        for(int ijack=0;ijack<njacks;ijack++)
//            for(int mA=0; mA<_nm; mA++)
//                for(int mB=0; mB<_nm; mB++)
//                    for(int r=0; r<_nr; r++)
//                    {
//                        eff_mass_temp[ijack][mA][mB] += eff_mass[ijack][r+_nr*mA][r+_nr*mB]/_nr;
//                    }
//        
//        out.eff_mass=eff_mass_temp;
//        
//        
//        if(_nm_Sea>1)
//        {
//            vvvd_t eff_mass_sea_temp(vvd_t(vd_t(0.0,out._nr),out._nr),njacks);
//            
//            for(int ijack=0;ijack<njacks;ijack++)
//                for(int r=0; r<_nr; r++)
//                    eff_mass_sea_temp[ijack][0][0] += eff_mass[ijack][r][r]/_nr;
//            
//            out.eff_mass_sea=eff_mass_sea_temp;
//        }
//    }
//    
//    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
//        for(int m=0; m<_nm; m++)
//            for(int r=0; r<_nr; r++)
//                for(int ijack=0;ijack<njacks;ijack++)
//                {
//                    //LO
//                    (out.sigma1_LO)[ilinmom][ijack][m] += sigma1_LO[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma2_LO)[ilinmom][ijack][m] += sigma2_LO[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma3_LO)[ilinmom][ijack][m] += sigma3_LO[ilinmom][ijack][r+_nr*m]/_nr;
//                    //EM
//                    (out.sigma1_PH)[ilinmom][ijack][m] += sigma1_PH[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma2_PH)[ilinmom][ijack][m] += sigma2_PH[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma3_PH)[ilinmom][ijack][m] += sigma3_PH[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma1_P)[ilinmom][ijack][m]  += sigma1_P[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma2_P)[ilinmom][ijack][m]  += sigma2_P[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma3_P)[ilinmom][ijack][m]  += sigma3_P[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma1_S)[ilinmom][ijack][m]  += sigma1_S[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma2_S)[ilinmom][ijack][m]  += sigma2_S[ilinmom][ijack][r+_nr*m]/_nr;
//                    (out.sigma3_S)[ilinmom][ijack][m]  += sigma3_S[ilinmom][ijack][r+_nr*m]/_nr;
//                }
//    
//    out.compute_Zq();
//    
//    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//    {
//        jproj_t jG_LO_mom_temp(vvvd_t(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks),nbil);
//        jproj_t jG_EM_mom_temp(vvvd_t(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks),nbil);
//        
//        for(int mA=0; mA<_nm; mA++)
//            for(int mB=0; mB<_nm; mB++)
//                for(int r=0; r<_nr; r++)
//                    for(int ijack=0;ijack<njacks;ijack++)
//                        for(int ibil=0; ibil<5; ibil++)
//                        {
//                            //LO
//                            jG_LO_mom_temp[ibil][ijack][mA][mB] +=
//                                jG_LO[ibilmom][ibil][ijack][r+_nr*mA][r+_nr*mB]/_nr;
//                            //EM
//                            jG_EM_mom_temp[ibil][ijack][mA][mB] +=
//                                jG_EM[ibilmom][ibil][ijack][r+nr*mA][r+nr*mB]/_nr;
//                        }
//        
//        (out.jG_LO)[ibilmom]=jG_LO_mom_temp;
//        (out.jG_EM)[ibilmom]=jG_EM_mom_temp;
//    }
//    
//    out.compute_Zbil();
//    
//    if(compute_4f)
//    {
//        
//        for(int imeslepmom=0;imeslepmom<_meslepmoms;imeslepmom++)
//            for(int iop1=0;iop1<5;iop1++)
//                for(int iop2=0;iop2<5;iop2++)
//                    for(int ijack=0;ijack<njacks;ijack++)
//                        for(int mA=0; mA<_nm; mA++)
//                            for(int mB=0; mB<_nm; mB++)
//                                for(int r=0; r<_nr; r++)
//                                {
//                                    //LO
//                                    (out.jpr_meslep_LO)[imeslepmom][iop1][iop2][ijack][mA][mB] += jpr_meslep_LO[imeslepmom][iop1][iop2][ijack][r+_nr*mA][r+_nr*mB]/_nr;
//                                    //EM
//                                    (out.jpr_meslep_EM)[imeslepmom][iop1][iop2][ijack][mA][mB] += jpr_meslep_EM[imeslepmom][iop1][iop2][ijack][r+_nr*mA][r+_nr*mB]/_nr;;
//                                    (out.jpr_meslep_nasty)[imeslepmom][iop1][iop2][ijack][mA][mB] += jpr_meslep_nasty[imeslepmom][iop1][iop2][ijack][r+_nr*mA][r+_nr*mB]/_nr;;
//                                }
//        
//        out.compute_Z4f();
//    }
//    
//    return out;
//}

//oper_t oper_t::chiral_extr()
//{
//    cout<<"Chiral extrapolation"<<endl<<endl;
//    
//    oper_t out=(*this);
//    
//    out._nr=_nr;
//    out._nm=1;
//    out._nmr=(out._nm)*(out._nr);
//    
////    resize_output(out);
//    out.allocate();
//    
//    vvvvd_t G_LO_err = get<1>(ave_err(jG_LO));    //[imom][ibil][mr1][mr2]
//    vvvvd_t G_EM_err = get<1>(ave_err(jG_EM));
//    
//    vvd_t Zq_err = get<1>(ave_err(jZq));        //[imom][mr]
//    vvd_t Zq_EM_err = get<1>(ave_err(jZq_EM));
//    
//    vvvvvd_t pr_meslep_LO_err=get<1>(ave_err(jpr_meslep_LO));  //[imom][iop1][iop2][mr1][mr2];
//    vvvvvd_t pr_meslep_EM_err=get<1>(ave_err(jpr_meslep_EM));
//    vvvvvd_t pr_meslep_nasty_err=get<1>(ave_err(jpr_meslep_nasty));
//    
//    //Sum of quark masses for the extrapolation
////    vd_t mass_sum(0.0,10);
////    int i_sum = 0;
////    for (int i=0; i<nm; i++)
////        for(int j=i;j<nm;j++)
////        {
////            mass_sum[i_sum] = mass_val[i]+mass_val[j];
////            i_sum++;
////        }
//
//    // average of eff_mass
//    vvd_t M_eff = get<0>(ave_err(eff_mass));
//    
//    //range for fit Zq
//    int x_min_q=0;
//    int x_max_q=_nm-1;
//    
//    // range for fit bilinears
//    int x_min=0;
//    int x_max=_nm*(_nm+1)/2-1;
//    
//    // number of fit parameters for bilinears
//    int npar[5]={3,2,3,2,2};
//    
//    // number of fit parameters for meslep
//    int npar_meslep[5]={2,2,3,3,2};
//    
//    //extrapolate Zq
//    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
//    {
//        for(int r=0; r<_nr; r++)
//        {
//            vvd_t coord_q(vd_t(0.0,_nm),2); // coords at fixed r
//            
//            vvvd_t jZq_r(vvd_t(vd_t(0.0,_nm),njacks),_linmoms);
//            vvvd_t jZq_EM_r(vvd_t(vd_t(0.0,_nm),njacks),_linmoms);
//            
//            vvd_t Zq_err_r(vd_t(0.0,_nm),_linmoms);
//            vvd_t Zq_EM_err_r(vd_t(0.0,_nm),_linmoms);
//            
//            for(int m=0; m<_nm; m++)
//            {
//                int mr = r + _nr*m;
//                
//                coord_q[0][m] = 1.0;
//                if(UseEffMass==0)
//                    coord_q[1][m]= mass_val[m];
//                else if(UseEffMass==0)
//                    coord_q[1][m] = pow(M_eff[mr][mr],2.0);
//                
//                for(int ijack=0;ijack<njacks;ijack++)
//                {
//                    jZq_r[ilinmom][ijack][m]=jZq[ilinmom][ijack][mr];
//                    jZq_EM_r[ilinmom][ijack][m]=jZq_EM[ilinmom][ijack][mr];
//                }
//                
//                Zq_err_r[ilinmom][m]=Zq_err[ilinmom][mr];
//                Zq_EM_err_r[ilinmom][m]=Zq_EM_err[ilinmom][mr];
//            }
//            
//            vvd_t jZq_pars_mom_r = polyfit(coord_q,2,Zq_err_r[ilinmom],jZq_r[ilinmom],x_min_q,x_max_q);
//            vvd_t jZq_EM_pars_mom_r = polyfit(coord_q,2,Zq_EM_err_r[ilinmom],jZq_EM_r[ilinmom],x_min_q,x_max_q);
//            
//            for(int ijack=0; ijack<njacks; ijack++)
//            {
//                (out.jZq)[ilinmom][ijack][r]=jZq_pars_mom_r[ijack][0];
//                (out.jZq_EM)[ilinmom][ijack][r]=jZq_EM_pars_mom_r[ijack][0];
//            }
//        }
//    }
//    
//    //extrapolate bilinears
//    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//    {
//        for(int r1=0; r1<_nr; r1++)
//            for(int r2=0; r2<_nr; r2++)
//            {
//                vvd_t coord_bil(vd_t(0.0,_nm*(_nm+1)/2),3); // coords at fixed r1 and r2
//                
//                vvvvd_t jG_LO_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),_bilmoms);
//                vvvvd_t jG_EM_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),_bilmoms);
//                
//                vvvd_t G_LO_err_r1_r2(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),_bilmoms);
//                vvvd_t G_EM_err_r1_r2(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),_bilmoms);
//
//                int ieq=0;
//                for(int m1=0; m1<_nm; m1++)
//                    for(int m2=m1; m2<_nm; m2++)
//                    {
//                        int mr1 = r1 + _nr*m1;
//                        int mr2 = r2 + _nr*m2;
//
//                        coord_bil[0][ieq] = 1.0;
//                        if(UseEffMass==0)
//                        {
//                            coord_bil[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
//                            coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];    // 1/(am1+am2)
//                        }
//                        else if(UseEffMass==1)
//                        {
//                            coord_bil[1][ieq] = pow((M_eff[mr1][mr2]+M_eff[mr2][mr1])/2.0,2.0);   //M^2 (averaged over equivalent combinations)
//                            coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];  //1/M^2
//                        }
//                    
//                        for(int ibil=0;ibil<nbil;ibil++)
//                        {
//                            for(int ijack=0;ijack<njacks;ijack++)
//                            {
//                                jG_LO_r1_r2[ibilmom][ibil][ijack][ieq] = (jG_LO[ibilmom][ibil][ijack][mr1][mr2]/*+jG_LO[ibilmom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                                jG_EM_r1_r2[ibilmom][ibil][ijack][ieq] = (jG_EM[ibilmom][ibil][ijack][mr1][mr2]/*+jG_EM[ibilmom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                            }
//                            
//                            G_LO_err_r1_r2[ibilmom][ibil][ieq] = (G_LO_err[ibilmom][ibil][mr1][mr2]/* + G_LO_err[ibilmom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                            G_EM_err_r1_r2[ibilmom][ibil][ieq] = (G_EM_err[ibilmom][ibil][mr1][mr2] /*+ G_EM_err[ibilmom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                        }
//                        
//                        ieq++;
//                    }
//                
//                for(int ibil=0;ibil<nbil;ibil++)
//                {
//                    vvd_t jG_LO_pars_mom_ibil_r1_r2 = polyfit(coord_bil,npar[ibil],G_LO_err_r1_r2[ibilmom][ibil],jG_LO_r1_r2[ibilmom][ibil],x_min,x_max);
//                    vvd_t jG_EM_pars_mom_ibil_r1_r2 = polyfit(coord_bil,npar[ibil],G_EM_err_r1_r2[ibilmom][ibil],jG_EM_r1_r2[ibilmom][ibil],x_min,x_max);
//                    
//                    for(int ijack=0;ijack<njacks;ijack++)
//                    {
////                        if(ibil==0 or ibil==2)
////                            for(int ieq=0;ieq<neq;ieq++)
////                            {
////                                // Goldstone pole subtraction from bilinears
////                                jG_LO_ave_r[imom][ibil][ijack][ieq] -= jG_LO_pars_mom[ibil][ijack][2];
////                                jG_EM_ave_r[imom][ibil][ijack][ieq] -= jG_EM_pars_mom[ibil][ijack][2];
////                            }
//                        
//                        // extrapolated value
//                        (out.jG_LO)[ibilmom][ibil][ijack][r1][r2] = jG_LO_pars_mom_ibil_r1_r2[ijack][0];
//                        (out.jG_EM)[ibilmom][ibil][ijack][r1][r2] = jG_EM_pars_mom_ibil_r1_r2[ijack][0];
//                    }
//                }
//            }
//    }
//    
//    out.compute_Zbil();
//    
//    if(compute_4f)
//    {
//        
//        //extrapolate meslep
//        for(int imeslepmom=0;imeslepmom<_meslepmoms;imeslepmom++)
//        {
//            for(int r1=0; r1<_nr; r1++)
//                for(int r2=0; r2<_nr; r2++)
//                {
//                    vvd_t coord_meslep(vd_t(0.0,_nm*(_nm+1)/2),3); // coords at fixed r1 and r2
//                    
//                    //                jpr_meslep_LO[imeslepmom][iop1][iop2][ijack][r+_nr*mA][r+_nr*mB]
//                    
//                    vvvvvd_t jpr_meslep_LO_r1_r2(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),nbil),_meslepmoms);
//                    vvvvvd_t jpr_meslep_EM_r1_r2(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),nbil),_meslepmoms);
//                    vvvvvd_t jpr_meslep_nasty_r1_r2(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),nbil),_meslepmoms);
//                    
//                    vvvvd_t pr_meslep_LO_err_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),nbil),_meslepmoms);
//                    vvvvd_t pr_meslep_EM_err_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),nbil),_meslepmoms);
//                    vvvvd_t pr_meslep_nasty_err_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),nbil),_meslepmoms);
//                    
//                    int ieq=0;
//                    for(int m1=0; m1<_nm; m1++)
//                        for(int m2=m1; m2<_nm; m2++)
//                        {
//                            int mr1 = r1 + _nr*m1;
//                            int mr2 = r2 + _nr*m2;
//                            
//                            coord_meslep[0][ieq] = 1.0;
//                            if(UseEffMass==0)
//                            {
//                                coord_meslep[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
//                                coord_meslep[2][ieq] = 1.0/coord_meslep[1][ieq];    // 1/(am1+am2)
//                            }
//                            else if(UseEffMass==1)
//                            {
//                                coord_meslep[1][ieq] = pow((M_eff[mr1][mr2]+M_eff[mr2][mr1])/2.0,2.0);   //M^2 (averaged over equivalent combinations)
//                                coord_meslep[2][ieq] = 1.0/coord_meslep[1][ieq];  //1/M^2
//                            }
//                            
//                            for(int iop1=0;iop1<nbil;iop1++)
//                                for(int iop2=0;iop2<nbil;iop2++)
//                                {
//                                    for(int ijack=0;ijack<njacks;ijack++)
//                                    {
//                                        jpr_meslep_LO_r1_r2[imeslepmom][iop1][iop2][ijack][ieq] = (jpr_meslep_LO[imeslepmom][iop1][iop2][ijack][mr1][mr2]/*+jG_LO[ibilmom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                                        jpr_meslep_EM_r1_r2[imeslepmom][iop1][iop2][ijack][ieq] = (jpr_meslep_EM[imeslepmom][iop1][iop2][ijack][mr1][mr2]/*+jG_LO[ibilmom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                                        jpr_meslep_nasty_r1_r2[imeslepmom][iop1][iop2][ijack][ieq] = (jpr_meslep_nasty[imeslepmom][iop1][iop2][ijack][mr1][mr2]/*+jG_LO[ibilmom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                                    }
//                                    
//                                    pr_meslep_LO_err_r1_r2[imeslepmom][iop1][iop2][ieq] = (pr_meslep_LO_err[imeslepmom][iop1][iop2][mr1][mr2]/* + G_LO_err[ibilmom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                                    pr_meslep_EM_err_r1_r2[imeslepmom][iop1][iop2][ieq] = (pr_meslep_EM_err[imeslepmom][iop1][iop2][mr1][mr2]/* + G_LO_err[ibilmom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                                    pr_meslep_nasty_err_r1_r2[imeslepmom][iop1][iop2][ieq] = (pr_meslep_nasty_err[imeslepmom][iop1][iop2][mr1][mr2]/* + G_LO_err[ibilmom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0*/);
//                                }
//                            
//                            ieq++;
//                        }
//                    
//                    for(int iop1=0;iop1<nbil;iop1++)
//                        for(int iop2=0;iop2<nbil;iop2++)
//                        {
//                            vvd_t jpr_meslep_LO_pars_mom_iop1_iop2_r1_r2 = polyfit(coord_meslep,npar_meslep[iop1],pr_meslep_LO_err_r1_r2[imeslepmom][iop1][iop2],jpr_meslep_LO_r1_r2[imeslepmom][iop1][iop2],x_min,x_max);
//                            vvd_t jpr_meslep_EM_pars_mom_iop1_iop2_r1_r2 = polyfit(coord_meslep,npar_meslep[iop1],pr_meslep_EM_err_r1_r2[imeslepmom][iop1][iop2],jpr_meslep_EM_r1_r2[imeslepmom][iop1][iop2],x_min,x_max);
//                            vvd_t jpr_meslep_nasty_pars_mom_iop1_iop2_r1_r2 = polyfit(coord_meslep,npar_meslep[iop1],pr_meslep_nasty_err_r1_r2[imeslepmom][iop1][iop2],jpr_meslep_nasty_r1_r2[imeslepmom][iop1][iop2],x_min,x_max);
//                            
//                            for(int ijack=0;ijack<njacks;ijack++)
//                            {
//                                //                        if(ibil==0 or ibil==2)
//                                //                            for(int ieq=0;ieq<neq;ieq++)
//                                //                            {
//                                //                                // Goldstone pole subtraction from bilinears
//                                //                                jG_LO_ave_r[imom][ibil][ijack][ieq] -= jG_LO_pars_mom[ibil][ijack][2];
//                                //                                jG_EM_ave_r[imom][ibil][ijack][ieq] -= jG_EM_pars_mom[ibil][ijack][2];
//                                //                            }
//                                
//                                // extrapolated value
//                                (out.jpr_meslep_LO)[imeslepmom][iop1][iop2][ijack][r1][r2] = jpr_meslep_LO_pars_mom_iop1_iop2_r1_r2[ijack][0];
//                                (out.jpr_meslep_EM)[imeslepmom][iop1][iop2][ijack][r1][r2] = jpr_meslep_EM_pars_mom_iop1_iop2_r1_r2[ijack][0];
//                                (out.jpr_meslep_nasty)[imeslepmom][iop1][iop2][ijack][r1][r2] = jpr_meslep_nasty_pars_mom_iop1_iop2_r1_r2[ijack][0];
//                            }
//                        }
//                }
//        }
//        
//        out.compute_Z4f();
//    }
//    
//    return out;
//}

//oper_t oper_t::subtract()
//{
//    cout<<"Subtracting the O(a2) effects"<<endl<<endl;
//    
//    oper_t out=(*this);
//    
////    resize_output(out);
//    out.allocate();
//    
//#pragma omp parallel for collapse(3)
//    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
//        for(int ijack=0;ijack<njacks;ijack++)
//            for(int mr1=0; mr1<_nmr; mr1++)
//            {
//                (out.jZq)[ilinmom][ijack][mr1] = jZq[ilinmom][ijack][mr1] - subtraction_q(ilinmom,LO);
//                (out.jZq_EM)[ilinmom][ijack][mr1] = jZq_EM[ilinmom][ijack][mr1] + /*(!)*/ subtraction_q(ilinmom,EM)/**jZq[ilinmom][ijack][mr1]*/;
//                // N.B.: the subtraction gets an extra minus sign due to the definition of the e.m. expansion!
//            }
//    
//#pragma omp parallel for collapse(5)
//    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//        for(int ibil=0;ibil<5;ibil++)
//            for(int ijack=0;ijack<njacks;ijack++)
//                for(int mr1=0; mr1<_nmr; mr1++)
//                    for(int mr2=0; mr2<_nmr; mr2++)
//                    {
//                        (out.jG_LO)[ibilmom][ibil][ijack][mr1][mr2] = jG_LO[ibilmom][ibil][ijack][mr1][mr2] - subtraction(ibilmom,ibil,LO);
//                        (out.jG_EM)[ibilmom][ibil][ijack][mr1][mr2] = jG_EM[ibilmom][ibil][ijack][mr1][mr2] - subtraction(ibilmom,ibil,EM)*jG_LO[ibilmom][ibil][ijack][mr1][mr2];
//                    }
//    
//    out.compute_Zbil();
//    
//#warning missing subtraction for 4f
//    
//    return out;
//}

//oper_t chiral_sea_extr(voper_t in)
//{
//    cout<<"Chiral sea extrapolation"<<endl<<endl;
//    
//    oper_t out=in[0];  //?
//    
//    int nmSea = in[0]._nm_Sea;
//    int _linmoms = in[0]._linmoms;
//    int _bilmoms = in[0]._bilmoms;
//    
//    //    resize_output(out);
//    out.allocate();
//    
//    out.path_to_ens = in[0].path_to_beta;
//
//    vd_t x(0.0,nmSea);
//    
//    vvd_t dy_Zq(vd_t(0.0,nmSea),_linmoms);
//    vvd_t dy_Zq_EM(vd_t(0.0,nmSea),_linmoms);
//    
//    vvvd_t dy_G(vvd_t(vd_t(0.0,nmSea),nbil),_bilmoms);
//    vvvd_t dy_G_EM(vvd_t(vd_t(0.0,nmSea),nbil),_bilmoms);
//
//    vvvd_t y_Zq(vvd_t(vd_t(0.0,nmSea),njacks),_linmoms);
//    vvvd_t y_Zq_EM(vvd_t(vd_t(0.0,nmSea),njacks),_linmoms);
//    
//    vvvvd_t y_G(vvvd_t(vvd_t(vd_t(0.0,nmSea),njacks),nbil),_bilmoms);
//    vvvvd_t y_G_EM(vvvd_t(vvd_t(vd_t(0.0,nmSea),njacks),nbil),_bilmoms);
//    
//    // range for fit
//    int x_min=0;
//    int x_max=nmSea-1;
//    
//    for(int msea=0; msea<nmSea; msea++)
//    {
//        x[msea] = ( get<0>(ave_err(in[msea].eff_mass_sea)) )[0][0];
//
//        for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
//        {
//            dy_Zq[ilinmom][msea] = (get<1>(ave_err(in[msea].jZq)))[ilinmom][0];
//            dy_Zq_EM[ilinmom][msea] = (get<1>(ave_err(in[msea].jZq_EM)))[ilinmom][0];
//            
//            for(int ijack=0;ijack<njacks;ijack++)
//            {
//                y_Zq[ilinmom][ijack][msea] = in[msea].jZq[ilinmom][ijack][0];
//                y_Zq_EM[ilinmom][ijack][msea] = in[msea].jZq_EM[ilinmom][ijack][0];
//            }
//        }
//        
//        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//            for(int ibil=0;ibil<nbil;ibil++)
//            {
//                dy_G[ibilmom][ibil][msea] = (get<1>(ave_err(in[msea].jG_LO)))[ibilmom][ibil][0][0];
//                dy_G_EM[ibilmom][ibil][msea] = (get<1>(ave_err(in[msea].jG_EM)))[ibilmom][ibil][0][0];
//            
//                for(int ijack=0;ijack<njacks;ijack++)
//                {
//                    y_G[ibilmom][ibil][ijack][msea] = in[msea].jG_LO[ibilmom][ibil][ijack][0][0];
//                    y_G_EM[ibilmom][ibil][ijack][msea] = in[msea].jG_EM[ibilmom][ibil][ijack][0][0];
//                }
//            }
//    }
//    
//    // extrapolate Zq
//    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
//    {
//        vvd_t coord_q(vd_t(0.0,nmSea),2); // coords at fixed r
//
//        for(int msea=0; msea<nmSea; msea++)
//        {
//            coord_q[0][msea] = 1.0;
//            if(UseEffMass==0)
//            {
//                cout<<" Impossible to extrapolate without using the effective mass. "<<endl;
//                exit(0);
//                //      coord_q[1][m]= mass_val[m];
//            }
//            else if(UseEffMass==1)
//                coord_q[1][msea] = pow(x[msea],2.0);
//
//            vvd_t jZq_pars_mom = polyfit(coord_q,2,dy_Zq[ilinmom],y_Zq[ilinmom],x_min,x_max);
//            vvd_t jZq_EM_pars_mom = polyfit(coord_q,2,dy_Zq_EM[ilinmom],y_Zq_EM[ilinmom],x_min,x_max);
//            
//            for(int ijack=0; ijack<njacks; ijack++)
//            {
//                (out.jZq)[ilinmom][ijack][0]=jZq_pars_mom[ijack][0];
//                (out.jZq_EM)[ilinmom][ijack][0]=jZq_EM_pars_mom[ijack][0];
//            }
//        }
//    }
//    
//    // extrapolate bilinears
//    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//    {
//        vvd_t coord_bil(vd_t(0.0,nmSea),2); // linear fit in sea extrapolation
//        
//        for(int msea=0; msea<nmSea; msea++)
//        {
//            coord_bil[0][msea] = 1.0;
//            if(UseEffMass==0)
//            {
//                cout<<" Impossible to extrapolate without using the effective mass. "<<endl;
//                exit(0);
////                coord_bil[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
////                coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];    // 1/(am1+am2)
//            }
//            else if(UseEffMass==1)
//                coord_bil[1][msea] = pow(x[msea],2.0);
//            
//            
//            for(int ibil=0;ibil<nbil;ibil++)
//            {
//                vvd_t jG_LO_pars_mom_ibil = polyfit(coord_bil,2,dy_G[ibilmom][ibil],y_G[ibilmom][ibil],x_min,x_max);
//                vvd_t jG_EM_pars_mom_ibil = polyfit(coord_bil,2,dy_G_EM[ibilmom][ibil],y_G_EM[ibilmom][ibil],x_min,x_max);
//                
//                for(int ijack=0;ijack<njacks;ijack++)
//                {
//                    // extrapolated value
//                    (out.jG_LO)[ibilmom][ibil][ijack][0][0] = jG_LO_pars_mom_ibil[ijack][0];
//                    (out.jG_EM)[ibilmom][ibil][ijack][0][0] = jG_EM_pars_mom_ibil[ijack][0];
//                }
//            }
//        }
//    }
//    
//    out.compute_Zbil();
//    
//#warning missing sea extrapolation for 4f
//    
//    return out;
//}

//oper_t theta_average( voper_t in)
//{
//    cout<<"Theta average"<<endl<<endl;
//    
//    oper_t out=in[0];  //?
//    
//    int _linmoms = in[0]._linmoms;
//    int _bilmoms = in[0]._bilmoms;
//    
//    out.allocate();
//    out.path_to_ens = in[0].path_to_beta;
//    
//    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
//        for(int ijack=0;ijack<njacks;ijack++)
//        {
//            out.jZq[ilinmom][ijack][0] = 0.5*(in[0].jZq[ilinmom][ijack][0] + in[1].jZq[ilinmom][ijack][0]);
//            out.jZq_EM[ilinmom][ijack][0] = 0.5*(in[0].jZq_EM[ilinmom][ijack][0] + in[1].jZq_EM[ilinmom][ijack][0]);
//        }
//    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//        for(int ibil=0;ibil<nbil;ibil++)
//            for(int ijack=0;ijack<njacks;ijack++)
//            {
//                out.jZ[ibilmom][ibil][ijack][0][0] = 0.5*(in[0].jZ[ibilmom][ibil][ijack][0][0] + in[1].jZ[ibilmom][ibil][ijack][0][0]);
//                out.jZ_EM[ibilmom][ibil][ijack][0][0] = 0.5*(in[0].jZ[ibilmom][ibil][ijack][0][0] + in[1].jZ[ibilmom][ibil][ijack][0][0]);
//            }
//    
//#warning missing theta average for 4f
//    
//    return out;
//}


//oper_t oper_t::evolve(const int b)
//{
//    cout<<"Evolving the Z's to the scale 1/a"<<endl<<endl;
//    
//    oper_t out=(*this);
//
//    double cq=0.0;
//    vd_t cO(0.0,5);
//    
//    double _ainv=ainv[b];
//    
//    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
//    {
//        cq=q_evolution_to_RIp_ainv(Nf,_ainv,p2[ilinmom]);
//        
//        for(int ijack=0;ijack<njacks;ijack++)
//            for(int mr1=0; mr1<_nmr; mr1++)
//            {
//                (out.jZq)[ilinmom][ijack][mr1] = jZq[ilinmom][ijack][mr1]/cq;
//                (out.jZq_EM)[ilinmom][ijack][mr1] = jZq_EM[ilinmom][ijack][mr1]/cq;
//            }
//    }
//
//    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//    {
//        // Note that ZV  ZA are RGI because they're protected by the WIs
//        cO[0]=S_evolution_to_RIp_ainv(Nf,_ainv,p2[ibilmom]); //S
//        cO[1]=1.0;                                       //A
//        cO[2]=P_evolution_to_RIp_ainv(Nf,_ainv,p2[ibilmom]); //P
//        cO[3]=1.0;                                       //V
//        cO[4]=T_evolution_to_RIp_ainv(Nf,_ainv,p2[ibilmom]); //T
//        
//        for(int ibil=0;ibil<5;ibil++)
//            for(int ijack=0;ijack<njacks;ijack++)
//                for(int mr1=0; mr1<_nmr; mr1++)
//                    for(int mr2=0; mr2<_nmr; mr2++)
//                    {
//                        (out.jZ)[ibilmom][ibil][ijack][mr1][mr2] = jZ[ibilmom][ibil][ijack][mr1][mr2]/cO[ibil];
//                        (out.jZ_EM)[ibilmom][ibil][ijack][mr1][mr2] = jZ_EM[ibilmom][ibil][ijack][mr1][mr2]/cO[ibil];
//                    }
//    }
//    
//#warning missing evolution for 4f
//    
//    return out;
//}

//int mom_list_xyz(vector<coords_t> &mom_list, const size_t imom)
//{
//    return abs(mom_list[imom][1])*abs(mom_list[imom][2])*abs(mom_list[imom][3]);
//}
//
//oper_t oper_t::average_equiv_moms()
//{
//    cout<<"Averaging over the equivalent momenta -- ";
//    
//    oper_t out=(*this);
//    
//    // Find equivalent linmoms
//    int tag=0, tag_aux=0;
//    double eps=1.0e-15;
//    
//    vector<int> tag_lin_vector;
//    tag_lin_vector.push_back(0);
//    
//    // Tag assignment to linmoms
//    for(int imom=0;imom<_linmoms;imom++)
//    {
//        int count_no=0;
//        
//        for(int j=0;j<imom;j++)
//        {
//            if( 2.0*abs(p2_tilde[j]-p2_tilde[imom])<eps*(p2_tilde[j]+p2_tilde[imom]) && mom_list_xyz(mom_list,j)==mom_list_xyz(mom_list,imom) &&
//               2.0*abs(abs(p[j][0])-abs(p[imom][0]))<eps*(abs(p[j][0])+abs(p[imom][0])) )
//            {
//                tag_aux = tag_lin_vector[j];
//            }else count_no++;
//            
//            if(count_no==imom)
//            {
//                tag++;
//                tag_lin_vector.push_back(tag);
//            }else if(j==imom-1)
//            {
//                tag_lin_vector.push_back(tag_aux);
//            }
//        }
//    }
//    
//    // number of equivalent linmoms
//    int neq_lin_moms = tag+1;
//    
////    int neqmoms = neq_lin_moms;
//    
//    out._linmoms=neq_lin_moms;
//    cout<<"found: "<<out._linmoms<<" equivalent linmoms ";
//    (out.linmoms).resize(out._linmoms);
//    
//    vector<double> p2_tilde_eqmoms(out._linmoms,0.0);
//
//
//    // count the different tags
//    vector<int> count_tag_lin_vector(out._linmoms);
//    int count=0;
//    for(int tag=0;tag<out._linmoms;tag++)
//    {
//        count=0;
//        for(int imom=0;imom<_linmoms;imom++)
//        {
//            if(tag_lin_vector[imom]==tag) count++;
//        }
//        count_tag_lin_vector[tag]=count;
//    }
//    
//    for(int tag=0;tag<out._linmoms;tag++)
//        for(int imom=0;imom<_linmoms;imom++)
//        {
//            if(tag_lin_vector[imom]==tag)
//            {
//                // fill the new linmoms and p2tilde
//                out.linmoms[tag] = {tag};
//                p2_tilde_eqmoms[tag] = p2_tilde[imom];
//            }
//        }
//    
//    print_vec(p2_tilde_eqmoms,path_print+"p2_tilde_eqmoms.txt");
//    
//    // Find equivalent bilmoms
//    tag=0, tag_aux=0;
//    
//    vector<int> tag_bil_vector;
//    tag_bil_vector.push_back(0);
//    
//    
//    //Tag assignment to bilmoms
//    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//    {
//        int count_no=0;
//        
//        const int imom1=bilmoms[ibilmom][1]; // p1
//        const int imom2=bilmoms[ibilmom][2]; // p2
//        
//        for(int j=0;j<ibilmom;j++)
//        {
//            const int imomA=bilmoms[j][1]; // p1
//            const int imomB=bilmoms[j][2]; // p2
//            
//            if( (tag_lin_vector[imom1]==tag_lin_vector[imomA] and tag_lin_vector[imom2]==tag_lin_vector[imomB])
//               or (tag_lin_vector[imom1]==tag_lin_vector[imomB] and tag_lin_vector[imom2]==tag_lin_vector[imomA]))
////            if(tag_lin_vector[imom1]+tag_lin_vector[imom2]==tag_lin_vector[imomA]+tag_lin_vector[imomB] and
////               tag_lin_vector[imom1]*tag_lin_vector[imom2]==tag_lin_vector[imomA]*tag_lin_vector[imomB])
//            {
//                tag_aux=tag_bil_vector[j];
//            }else count_no++;
//            
//            if(count_no==ibilmom)
//            {
//                tag++;
//                tag_bil_vector.push_back(tag);
//            }else if(j==ibilmom-1)
//            {
//                tag_bil_vector.push_back(tag_aux);
//            }
//        }
//    }
//    
//    // number of equivalent bilmoms
//    int neq_bil_moms = tag+1;
//    
//    out._bilmoms=neq_bil_moms;
//    cout<<"and "<<neq_bil_moms<<" equivalent bilmoms "<<endl<<endl;
//    (out.bilmoms).resize(out._bilmoms);
//    
//    // count the different tags
//    vector<int> count_tag_bil_vector(out._bilmoms);
//    count=0;
//    for(int tag=0;tag<out._bilmoms;tag++)
//    {
//        count=0;
//        for(int imom=0;imom<_bilmoms;imom++)
//        {
//            if(tag_bil_vector[imom]==tag) count++;
//        }
//        count_tag_bil_vector[tag]=count;
//    }
//    
//    for(int tag=0;tag<out._bilmoms;tag++)
//        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
//        {
//            if(tag_bil_vector[ibilmom]==tag)
//            {
//                // fill the new bilmoms
//                const int imom0=bilmoms[tag][0]; // k
//                const int imom1=bilmoms[tag][1]; // p1
//                const int imom2=bilmoms[tag][2]; // p2
//                
//                out.bilmoms[tag] = {imom0,imom1,imom2};
//            }
//        }
//    
////    resize_output(out);
//    out.allocate();
//    
//    // average over the equivalent momenta
//    for(int tag=0;tag<neq_lin_moms;tag++)
//        for(int imom=0;imom<_linmoms;imom++)
//        {
//            if(tag_lin_vector[imom]==tag)
//            {
//                for(int ijack=0;ijack<njacks;ijack++)
//                    for(int mr1=0; mr1<_nmr; mr1++)
//                    {
//                        (out.jZq)[tag][ijack][mr1]+=jZq[imom][ijack][mr1]/count_tag_lin_vector[tag];
//                        (out.jZq_EM)[tag][ijack][mr1]+=jZq_EM[imom][ijack][mr1]/count_tag_lin_vector[tag];
//                    }
//            }
//        }
//    
//    out.compute_Zq();
//    
//    for(int tag=0;tag<neq_bil_moms;tag++)
//        for(int imom=0;imom<_bilmoms;imom++)
//        {
//            if(tag_bil_vector[imom]==tag)
//            {
//                for(int ibil=0;ibil<5;ibil++)
//                    for(int ijack=0;ijack<njacks;ijack++)
//                        for(int mr1=0; mr1<_nmr; mr1++)
//                            for(int mr2=0; mr2<_nmr; mr2++)
//                            {
//                                (out.jG_LO)[tag][ibil][ijack][mr1][mr2]+=jG_LO[imom][ibil][ijack][mr1][mr2]/count_tag_bil_vector[tag];
//                                (out.jG_EM)[tag][ibil][ijack][mr1][mr2]+=jG_EM[imom][ibil][ijack][mr1][mr2]/count_tag_bil_vector[tag];
//                            }
//            }
//        }
//
//    out.compute_Zbil();
//    
//    if(compute_4f)
//    {
//        for(int tag=0;tag<neq_bil_moms;tag++)
//            for(int imom=0;imom<_bilmoms;imom++)
//            {
//                if(tag_bil_vector[imom]==tag)
//                {
//                    for(int iop1=0;iop1<5;iop1++)
//                        for(int iop2=0;iop2<5;iop2++)
//                            for(int ijack=0;ijack<njacks;ijack++)
//                                for(int mr1=0; mr1<_nmr; mr1++)
//                                    for(int mr2=0; mr2<_nmr; mr2++)
//                                    {
//                                        (out.jpr_meslep_LO)[tag][iop1][iop2][ijack][mr1][mr2]+=jpr_meslep_LO[imom][iop1][iop2][ijack][mr1][mr2]/count_tag_bil_vector[tag];
//                                        (out.jpr_meslep_EM)[tag][iop1][iop2][ijack][mr1][mr2]+=jpr_meslep_EM[imom][iop1][iop2][ijack][mr1][mr2]/count_tag_bil_vector[tag];
//                                        (out.jpr_meslep_nasty)[tag][iop1][iop2][ijack][mr1][mr2]+=jpr_meslep_nasty[imom][iop1][iop2][ijack][mr1][mr2]/count_tag_bil_vector[tag];
//                                    }
//                }
//            }
//        
//        out.compute_Z4f();
//    }
//    
//    return out;
//}

//voper_t a2p2_extr(voper_t in /*, const int LO_or_EM*/)  // M1 method
//{
//    voper_t out;
//    
//    //    int neq_moms = (out.jZq).size();
//    int _linmoms=in[0]._linmoms;
//    int _bilmoms=in[0]._bilmoms;
//    
//    vector<double> p2_tilde_eqmoms(_linmoms);
//    read_vec(p2_tilde_eqmoms,in[0].path_print+"p2_tilde_eqmoms.txt");
//    
//    double b0 = in[1]._beta; // b0 is the 'central' value of beta.
//    vd_t b(nbeta);
//    for(auto &i : b)
//    {
//        b[i] = in[i]._beta;
//    }
//    
//    for(int ibeta=0; ibeta<nbeta; ibeta++)
//    {
//        out[ibeta]=in[ibeta];
//        
//        out[ibeta]._linmoms=1;
//        out[ibeta]._bilmoms=1;
//
//        out[ibeta].allocate();
//        
//        for(int LO_or_EM=0; LO_or_EM<2; LO_or_EM++)
//        {
//            vvd_t jZq_out(vd_t(0.0,_linmoms),njacks);
//            vvvd_t jZ_out(vvd_t(vd_t(0.0,_bilmoms),njacks),nbil);
//            
//            vd_t Zq_err(0.0,_linmoms);
//            vvd_t Z_err(vd_t(0.0,_bilmoms),nbil);
//            
//            if(LO_or_EM==0)
//            {
//                cout<<"-- Leading Order --"<<endl;
//                
//#pragma omp parallel for collapse(2)
//                for(int imom=0; imom<_linmoms; imom++)
//                    for(int ijack=0; ijack<njacks; ijack++)
//                        jZq_out[ijack][imom] = in[ibeta].jZq[imom][ijack][0];
//                
//#pragma omp parallel for collapse(3)
//                for(int imom=0; imom<_bilmoms; imom++)
//                    for(int ijack=0; ijack<njacks; ijack++)
//                        for(int ibil=0; ibil<nbil; ibil++)
//                            jZ_out[ibil][ijack][imom] = in[ibeta].jZ[imom][ibil][ijack][0][0];
//                
//                vvd_t Zq_err_tmp = get<1>(ave_err(in[ibeta].jZq));
//                vvvvd_t Z_err_tmp = get<1>(ave_err(in[ibeta].jZ));
//                
//                for(int imom=0; imom<_linmoms; imom++)
//                    Zq_err[imom] = Zq_err_tmp[imom][0];
//                
//                for(int imom=0; imom<_bilmoms; imom++)
//                    for(int ibil=0; ibil<nbil; ibil++)
//                        Z_err[ibil][imom] = Z_err_tmp[imom][ibil][0][0];
//                
//            }
//            else if(LO_or_EM==1)
//            {
//                cout<<"-- EM Correction --"<<endl;
//                
//#pragma omp parallel for collapse(2)
//                for(int imom=0; imom<_linmoms; imom++)
//                    for(int ijack=0; ijack<njacks; ijack++)
//                        jZq_out[ijack][imom] = in[ibeta].jZq_EM[imom][ijack][0];
//                
//#pragma omp parallel for collapse(3)
//                for(int imom=0; imom<_bilmoms; imom++)
//                    for(int ijack=0; ijack<njacks; ijack++)
//                        for(int ibil=0; ibil<nbil; ibil++)
//                            jZ_out[ibil][ijack][imom] = in[ibeta].jZ_EM[imom][ibil][ijack][0][0];
//                
//                vvd_t Zq_err_tmp = get<1>(ave_err(in[ibeta].jZq_EM));
//                vvvvd_t Z_err_tmp = get<1>(ave_err(in[ibeta].jZ_EM));
//                
//                for(int imom=0; imom<_linmoms; imom++)
//                    Zq_err[imom] = Zq_err_tmp[imom][0];
//                
//                for(int imom=0; imom<_bilmoms; imom++)
//                    for(int ibil=0; ibil<nbil; ibil++)
//                        Z_err[ibil][imom] = Z_err_tmp[imom][ibil][0][0];
//            }
//            
//            // Simultaneous extrapolation: y = A + B(g2)*a2p2
//            //  B(g^2) = B(g0^2) + C*(g^2 - g0^2)  where g0 is the 'central' value of the coupling constant
//            //         = B(g0^2) + C*6*(b0 - b)/(b0*b)
//            int npar = 3;
//            
//            //linear fit Zq
//            int range_min=0;  //a2p2~1
//            int range_max=_linmoms;
//            double p_min_value=p2min;
//            
//            vvd_t coord_lin_linear(vd_t(0.0,_linmoms),npar);
//            
//            for(int i=0; i<range_max; i++)
//            {
//                coord_lin_linear[0][i] = 1.0;  //costante
//                coord_lin_linear[1][i] = p2_tilde_eqmoms[i];   //p^2
//                coord_lin_linear[2][i] = p2_tilde_eqmoms[i]*6.0*(b0-b[ibeta])/(b0*b[ibeta]);
//            }
//            
//            vd_t jZq_out_par_ijack(0.0,npar);
//            
//            double Zq_ave_cont=0.0, sqr_Zq_ave_cont=0.0, Zq_err_cont=0.0;
//            
//            for(int ijack=0; ijack<njacks; ijack++)
//            {
//                jZq_out_par_ijack=fit_continuum(coord_lin_linear,Zq_err,jZq_out[ijack],range_min,range_max,p_min_value);
//                
//                Zq_ave_cont += jZq_out_par_ijack[0]/njacks;
//                sqr_Zq_ave_cont += jZq_out_par_ijack[0]*jZq_out_par_ijack[0]/njacks;
//                
//                if(LO_or_EM==0)
//                {
//                    (out[ibeta].jZq)[0][ijack][0] = jZq_out_par_ijack[0];
//                }
//                else if(LO_or_EM==1)
//                {
//                    (out[ibeta].jZq_EM)[0][ijack][0] = jZq_out_par_ijack[0];
//                }
//            }
//            
//            Zq_err_cont=sqrt((double)(njacks-1))*sqrt(sqr_Zq_ave_cont-Zq_ave_cont*Zq_ave_cont);
//            
//            cout<<"ZQ = "<<Zq_ave_cont<<" +/- "<<Zq_err_cont<<endl;
//            
//            //linear fit Z
//            range_min=0;  //a2p2~1
//            range_max=_bilmoms;
//            
//            vvd_t coord_bil_linear(vd_t(0.0,_bilmoms),npar);
//            
//            for(int i=0; i<range_max; i++)
//            {
//                //        int imomk = (out.bilmoms)[i][0];
//                int imomk = i;      /// it will work temporarily only for RIMOM (!!!!!!!)
//                
//                coord_bil_linear[0][i] = 1.0;  //costante
//                coord_bil_linear[1][i] = p2_tilde_eqmoms[imomk];   //p^2
//                coord_bil_linear[2][i] = p2_tilde_eqmoms[imomk]*6.0*(b0-b[ibeta])/(b0*b[ibeta]);
//            }
//            
//            vvd_t jZ_out_par_ijack(vd_t(0.0,npar),nbil);
//            vd_t Z_ave_cont(0.0,nbil), sqr_Z_ave_cont(0.0,nbil), Z_err_cont(0.0,nbil);
//            
//            for(int ijack=0; ijack<njacks; ijack++)
//                for(int ibil=0; ibil<nbil; ibil++)
//                {
//                    jZ_out_par_ijack[ibil]=fit_continuum(coord_bil_linear,Z_err[ibil],jZ_out[ibil][ijack],range_min,range_max,p_min_value);
//                    
//                    Z_ave_cont[ibil] += jZ_out_par_ijack[ibil][0]/njacks;
//                    sqr_Z_ave_cont[ibil] += jZ_out_par_ijack[ibil][0]*jZ_out_par_ijack[ibil][0]/njacks;
//                    
//                    if(LO_or_EM==0)
//                    {
//                        (out[ibeta].jZ)[0][ibil][ijack][0][0] = jZ_out_par_ijack[ibil][0];
//                    }
//                    else if(LO_or_EM==1)
//                    {
//                        (out[ibeta].jZ_EM)[0][ibil][ijack][0][0] = jZ_out_par_ijack[ibil][0];
//                    }
//                }
//            
//            for(int ibil=0; ibil<nbil;ibil++)
//                Z_err_cont[ibil]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_ave_cont[ibil]-Z_ave_cont[ibil]*Z_ave_cont[ibil]));
//            
//            vector<string> bil={"S","A","P","V","T"};
//            
//            for(int ibil=0; ibil<nbil;ibil++)
//            {
//                cout<<"Z"<<bil[ibil]<<" = "<<Z_ave_cont[ibil]<<" +/- "<<Z_err_cont[ibil]<<endl;
//            }
//            
//            //    vector<double> pert={-0.0695545,-0.100031,-0.118281,-0.130564,-0.108664}; // Martinelli-Zhang
//            //
//            //    if(LO_or_EM==1)
//            //    {
//            //        cout<<"Z divided by the perturbative estimates (to be evolved in MSbar"
//            //    for(int ibil=0;i<nbil;ibil++)
//            //    {
//            //        cout<<"Z"<<bil[ibil]<<"(fact) = "<<A_bil[ibil]/pert[ibil]<<" +/- "<<A_err[ibil]/pert[ibil]<<endl;
//            //    }
//            //    }
//            
//            cout<<endl;
//            
//#warning missing a2p2 extrapolation for 4f
//            
//        } // close LO_or_EM loop
//    }// close beta loop
//    return out;
//}

void oper_t::plot(const string suffix)
{
    oper_t in=(*this);
    
    Zq_tup Zq_ave_err = ave_err(in.jZq);
    Zq_tup Zq_EM_ave_err = ave_err(in.jZq_EM);
    
    Zbil_tup Zbil_ave_err = ave_err(in.jZ);
    Zbil_tup Zbil_EM_ave_err = ave_err(in.jZ_EM);
    
    vvd_t Zq_ave = get<0>(Zq_ave_err);        //[imom][mr]
    vvd_t Zq_EM_ave = get<0>(Zq_EM_ave_err);
    
    vvd_t Zq_err = get<1>(Zq_ave_err);        //[imom][mr]
    vvd_t Zq_EM_err = get<1>(Zq_EM_ave_err);
    
    vvvvd_t Z_ave = get<0>(Zbil_ave_err);    //[imom][ibil][mr1][mr2]
    vvvvd_t Z_EM_ave = get<0>(Zbil_EM_ave_err);
    
    vvvvd_t Z_err = get<1>(Zbil_ave_err);    //[imom][ibil][mr1][mr2]
    vvvvd_t Z_EM_err = get<1>(Zbil_EM_ave_err);
    
//    Zmeslep_tup meslep_ave_err = ave_err(in.jpr_meslep_LO);
//    Zmeslep_tup meslep_EM_ave_err = ave_err(in.jpr_meslep_EM);
//    Zmeslep_tup meslep_nasty_ave_err = ave_err(in.jpr_meslep_nasty);
//    
//
//    vvvvvd_t meslep_ave=get<0>(meslep_ave_err);  //[imom][iop1][iop2][mr1][mr2];
//    vvvvvd_t meslep_EM_ave=get<0>(meslep_EM_ave_err);
//    vvvvvd_t meslep_nasty_ave=get<0>(meslep_nasty_ave_err);
//    
//    vvvvvd_t meslep_err=get<1>(meslep_ave_err);  //[imom][iop1][iop2][mr1][mr2];
//    vvvvvd_t meslep_EM_err=get<1>(meslep_EM_ave_err);
//    vvvvvd_t meslep_nasty_err=get<1>(meslep_nasty_ave_err);
//
    
    Z4f_tup Z_4f_ave_err = ave_err(in.jZ_4f);
    Z4f_tup Z_4f_EM_ave_err = ave_err(in.jZ_4f_EM);

    vvvvvd_t Z_4f_ave=get<0>(Z_4f_ave_err);  //[imom][iop1][iop2][mr1][mr2];
    vvvvvd_t Z_4f_EM_ave=get<0>(Z_4f_EM_ave_err);
    
    vvvvvd_t Z_4f_err=get<1>(Z_4f_ave_err);  //[imom][iop1][iop2][mr1][mr2];
    vvvvvd_t Z_4f_EM_err=get<1>(Z_4f_EM_ave_err);
    
    // this choice is relative to the twisted basis
    vector<string> bil={"S","V","P","A","T"};
    
    ofstream Zq_data, Zq_EM_data;
    vector<ofstream> Zbil_data(nbil), Zbil_EM_data(nbil);
//    vector<ofstream> Zmeslep_data(nbil*nbil), Zmeslep_EM_data(nbil*nbil), Zmeslep_nasty_data(nbil*nbil);
    vector<ofstream> Z_4f_data(nbil*nbil), Z_4f_EM_data(nbil*nbil);
    
    Zq_data.open(path_to_ens+"plots/Zq"+(suffix!=""?("_"+suffix):string(""))+".txt");
    Zq_EM_data.open(path_to_ens+"plots/Zq_EM"+(suffix!=""?("_"+suffix):string(""))+".txt");
    
    vector<double> p2t;
    
    if(in._linmoms==moms)
    {
        p2t.resize(in._linmoms);
        read_vec(p2t,path_print+"p2_tilde.txt");
    }
    else
    {
        p2t.resize(in._linmoms);
        read_vec(p2t,path_print+"p2_tilde_eqmoms.txt");
    }
    
    cout<<"Plotting Zq"<<endl;
    for(int imom=0; imom<in._linmoms; imom++)
    {
        Zq_data<<p2t[imom]<<"\t"<<Zq_ave[imom][0]<<"\t"<<Zq_err[imom][0]<<endl;
        Zq_EM_data<<p2t[imom]<<"\t"<<Zq_EM_ave[imom][0]<<"\t"<<Zq_EM_err[imom][0]<<endl;
    }
    
    cout<<"Plotting Zbil"<<endl;
    for(int ibil=0;ibil<nbil;ibil++)
    {
        Zbil_data[ibil].open(path_to_ens+"plots/Z"+bil[ibil]+(suffix!=""?("_"+suffix):string(""))+".txt");
        Zbil_EM_data[ibil].open(path_to_ens+"plots/Z"+bil[ibil]+"_EM"+(suffix!=""?("_"+suffix):string(""))+".txt");
        
        for(int imom=0; imom<in._bilmoms; imom++)
        {
//            int imomq = in.bilmoms[imom][0];
//            cout<<"imomq: "<<imomq<<endl;
//            int imomk = in.linmoms[imomq][0];
            int imomk = imom;   // NB: it works only for RIMOM!
            
            Zbil_data[ibil]<<p2t[imomk]<<"\t"<<Z_ave[imom][ibil][0][0]<<"\t"<<Z_err[imom][ibil][0][0]<<endl;
            Zbil_EM_data[ibil]<<p2t[imomk]<<"\t"<<Z_EM_ave[imom][ibil][0][0]<<"\t"<<Z_EM_err[imom][ibil][0][0]<<endl;
        }
    }
    
    if(compute_4f)
    {
//        for(int i=0;i<nbil*nbil;i++)
//        {
//            int iop2=i%nbil;
//            int iop1=(i-iop2)/nbil;
//            
//            Zmeslep_data[i].open(path_to_ens+"plots/meslep_"+to_string(iop1)+"_"+to_string(iop2)+(suffix!=""?("_"+suffix):string(""))+".txt");
//            Zmeslep_EM_data[i].open(path_to_ens+"plots/meslep_"+to_string(iop1)+"_"+to_string(iop2)+"_EM"+(suffix!=""?("_"+suffix):string(""))+".txt");
//            Zmeslep_nasty_data[i].open(path_to_ens+"plots/meslep_"+to_string(iop1)+"_"+to_string(iop2)+"_NASTY"+(suffix!=""?("_"+suffix):string(""))+".txt");
//            
//            for(int imom=0; imom<in._bilmoms; imom++)
//            {
//                //            int imomq = in.bilmoms[imom][0];
//                //            cout<<"imomq: "<<imomq<<endl;
//                //            int imomk = in.linmoms[imomq][0];
//                int imomk = imom;   // NB: it works only for RIMOM!
//                
//                Zmeslep_data[i]<<p2t[imomk]<<"\t"<<meslep_ave[imom][iop1][iop2][0][0]<<"\t"<<meslep_err[imom][iop1][iop2][0][0]<<endl;
//                Zmeslep_EM_data[i]<<p2t[imomk]<<"\t"<<meslep_EM_ave[imom][iop1][iop2][0][0]<<"\t"<<meslep_EM_err[imom][iop1][iop2][0][0]<<endl;
//                Zmeslep_nasty_data[i]<<p2t[imomk]<<"\t"<<meslep_nasty_ave[imom][iop1][iop2][0][0]<<"\t"<<meslep_nasty_err[imom][iop1][iop2][0][0]<<endl;
//            }
//        }
        
        cout<<"Plotting Z4f"<<endl;
        for(int i=0;i<nbil*nbil;i++)
        {
            int iop2=i%nbil;
            int iop1=(i-iop2)/nbil;
            
            Z_4f_data[i].open(path_to_ens+"plots/Z4f_"+to_string(iop1)+"_"+to_string(iop2)+(suffix!=""?("_"+suffix):string(""))+".txt");
            Z_4f_EM_data[i].open(path_to_ens+"plots/Z4f_"+to_string(iop1)+"_"+to_string(iop2)+"_EM"+(suffix!=""?("_"+suffix):string(""))+".txt");
            
            for(int imom=0; imom<in._bilmoms; imom++)
            {
                //            int imomq = in.bilmoms[imom][0];
                //            cout<<"imomq: "<<imomq<<endl;
                //            int imomk = in.linmoms[imomq][0];
                int imomk = imom;   // NB: it works only for RIMOM!
                
                Z_4f_data[i]<<p2t[imomk]<<"\t"<<Z_4f_ave[imom][iop1][iop2][0][0]<<"\t"<<Z_4f_err[imom][iop1][iop2][0][0]<<endl;
                Z_4f_EM_data[i]<<p2t[imomk]<<"\t"<<Z_4f_EM_ave[imom][iop1][iop2][0][0]<<"\t"<<Z_4f_EM_err[imom][iop1][iop2][0][0]<<endl;
            }
        }
    }
    
}
