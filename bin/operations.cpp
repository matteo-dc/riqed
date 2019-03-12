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
#include "allocate.hpp"

//#define EXTERN_OPER
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
    jmeslep::set_ins();
    pr_meslep::set_ins();
}

void oper_t::set_moms()
{
    //read mom list
    read_mom_list(path_to_moms);
    moms=mom_list.size();
    cout<<"Read: "<<moms<<" momenta from \""<<mom_path<<"\" (BC: "<<BC<<")."<<endl;
    
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
    
    int count_filtered=0;
    
    for(int imom=0;imom<moms;imom++)
        if(filt_moms[imom])
        {
            count_filtered++;
            
            linmoms[imom]={imom};
            bilmoms[imom]={imom,imom,imom};
            meslepmoms[imom]=bilmoms[imom];
        }
    cout<<"Filtered "<<count_filtered<<" momenta."<<endl;
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
    cout<<endl;
    cout<<"----- basic -----"<<endl<<endl;
    
    _beta=beta[b];
    _beta_label=beta_label[b];
    _nm_Sea=nm_Sea[b];
    _SeaMasses_label=to_string(SeaMasses_label[b][msea]);
    _theta_label=theta_label[th];
    
    if(inte_analysis)
    {
        // e.g. /.../matteo/Nf4/
        path_ensemble = path_folder + path_analysis[0]+"/";
        
        // e.g. /.../matteo/Nf4/B_b1.95/
        if(strcmp(an_suffix.c_str(),"")==0)
            path_to_beta = path_ensemble + _beta_label + "_b" +
            to_string_with_precision(_beta,2) + "/";
        else
            path_to_beta = path_ensemble;
        
        // e.g. B1m
        ensemble_name = _beta_label + _SeaMasses_label + _theta_label;
        
        // e.g. /.../matteo/Nf4/B_b1.95/B1m/
        path_to_ens =  path_to_beta + ensemble_name + "/";
    }
    else if(free_analysis)
    {
        // e.g. /.../matteo/free_matching/
        path_ensemble = path_folder + path_analysis[0]+"/";
        
        // e.g. /.../matteo/free_matching/B/
        path_to_beta = path_ensemble + _beta_label + "/";
        
        // e.g. B1m
        ensemble_name = _beta_label + _SeaMasses_label + _theta_label;
        
        // e.g. /.../matteo/free_matching/B1m/
        path_to_ens = path_ensemble + ensemble_name + "/";
    }
    else if(eta_analysis)
    {
        // e.g. /.../matteo/Rat/
        path_to_beta = path_folder + path_analysis[0]+"/";
        
        // e.g. /.../matteo/****/
        if(!recompute_basic)
        {
            if(strcmp(an_suffix.c_str(),"")==0) /* Nf4/B_b1.95 */
                path_ensemble = path_folder + path_analysis[1]+"/" +
                                _beta_label + "_b" + to_string_with_precision(_beta,2) + "/";
            else    /* Nf4/ */
                path_ensemble = path_folder + path_analysis[1]+"/";
        }
        else                 /* free_matching */
            path_ensemble = path_folder + path_analysis[2]+"/";
        
        // e.g. B1m
        ensemble_name = _beta_label + _SeaMasses_label + _theta_label;
        
        // e.g /.../matteo/*****/B1m/
        path_to_ens = path_ensemble + ensemble_name + "/";
    }
    
    // impose not to read mes_contr in the 2nd loop (free)
    if(recompute_basic)
    {
        compute_mpcac=0;
        UseEffMass=0;
    }
    
    read_input(path_to_ens,ensemble_name);
    path_to_moms = path_to_ens + mom_path;
    
    path_print = path_to_ens+"print/";
   
    V=size[0]*size[1]*size[2]*size[3];
 
    if(!recompute_basic)
    {
        _nm=nm;
        _nr=nr;
    }
    else
    {
        _nm=1;
        _nr=1;
    }
    _nmr=_nm*_nr;
    
    g2=6.0/_beta;
    g2_tilde=g2/plaquette;
    
    set_moms();
    
    set_ins();
    
    allocate_val();
    if(!load_ave) allocate();
    
    if(compute_mpcac)
    {
        compute_mPCAC("");
        if(!free_analysis) compute_mPCAC("sea");
    }
    if(UseEffMass and !free_analysis and !recompute_basic)
    {
        eff_mass=read_eff_mass(path_to_ens+"eff_mass_array");
        eff_mass_corr=read_eff_mass_corr(path_to_ens+"eff_mass_corr_array");
        
        eff_mass_time=read_eff_mass_time(path_to_ens+"eff_mass_array_time");
        eff_mass_corr_time=read_eff_mass_corr_time(path_to_ens+"eff_mass_corr_array_time");
        
        if(_nm_Sea>1 and !free_analysis and !recompute_basic)
            eff_mass_sea=read_eff_mass_sea(path_to_ens+"eff_mass_sea_array");
    }
    
    if(!load_ave)
    {
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
        if(ntypes!=3)
            compute_deltam_from_prop();
        
        compute_Zq();
        compute_Zbil();
        if(compute_4f) compute_Z4f();
    }
}

//////////

oper_t oper_t::average_r()
{
    cout<<endl;
    cout<<"----- r average -----"<<endl<<endl;
    
    oper_t out=(*this);
    
    out._nr=1;
    out._nm=_nm;
    out._nmr=(out._nm)*(out._nr);
    
    out.allocate_val();
    
    out.eff_mass=eff_mass;
    out.eff_mass_sea=eff_mass_sea;
    
    if(!load_ave)
    {
        out.allocate();
        
#pragma omp parallel for collapse(5)
        for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
            for(int iproj=0;iproj<sigma::nproj;iproj++)
                for(int ins=0;ins<sigma::nins;ins++)
                    for(int m=0; m<_nm; m++)
                        for(int ijack=0;ijack<njacks;ijack++)
                            for(int r=0; r<_nr; r++)
                            {
                                const int coeff=(iproj==sigma::SIGMA3 and r==1)?-1:+1;
                                
                                (out.sigma)[ilinmom][iproj][ins][ijack][m] +=
                                    coeff*sigma[ilinmom][iproj][ins][ijack][r+_nr*m]/_nr;
                            }
        
        if(ntypes!=3)
        {
            out.deltam_computed=true;
            out.compute_deltam_from_prop();
        }
        
        out.compute_Zq();
        
#pragma omp parallel for collapse(6)
        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
            for(int mA=0; mA<_nm; mA++)
                for(int mB=0; mB<_nm; mB++)
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int ibil=0; ibil<5; ibil++)
                            for(int ins=0; ins<gbil::nins; ins++)
                                for(int r=0; r<_nr; r++)
                                    (out.jG)[ibilmom][ins][ibil][ijack][mA][mB] +=
                                        jG[ibilmom][ins][ibil][ijack][r+_nr*mA][r+_nr*mB]/_nr;
        
        out.compute_Zbil();
        
        if(compute_4f)
        {
#pragma omp parallel for collapse(7)
            for(int imeslepmom=0;imeslepmom<_meslepmoms;imeslepmom++)
                for(int iop1=0;iop1<5;iop1++)
                    for(int iop2=0;iop2<5;iop2++)
                        for(int ijack=0;ijack<njacks;ijack++)
                            for(int ins=0;ins<pr_meslep::nins;ins++)
                                for(int mA=0;mA<_nm;mA++)
                                    for(int mB=0;mB<_nm;mB++)
                                        for(int r=0;r<_nr;r++)
                                            (out.jpr_meslep)[imeslepmom][ins][iop1][iop2][ijack][mA][mB] +=
                                                jpr_meslep[imeslepmom][ins][iop1][iop2][ijack][r+_nr*mA][r+_nr*mB]/_nr;
            
            out.compute_Z4f();
            
        }
    }
    
    return out;
}

oper_t oper_t::chiral_extr()
{
    cout<<endl;
    cout<<"----- valence chiral extrapolation -----"<<endl<<endl;
    
    oper_t out=(*this);
    
    out._nr=_nr;
    out._nm=1;
    out._nmr=(out._nm)*(out._nr);
    
    out.allocate_val();
    out.allocate();
    
    out.eff_mass = eff_mass;
    out.eff_mass_sea = eff_mass_sea;

    // average of eff_mass
    vvd_t M_eff = get<0>(ave_err(eff_mass));
    vvd_t dM_eff = get<0>(ave_err(eff_mass_corr));
    for(int m=0; m<_nm; m++)
        printf("m: %d \t M_eff: %lg \t dM_eff: %lg\n",m,M_eff[m][m],dM_eff[m][m]);
    
    // printing the squared mass
    vvvd_t eff_mass_sqr(vvd_t(vd_t(0.0,_nm),_nm),njacks);
    
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m1=0; m1<_nm; m1++)
            for(int m2=0; m2<_nm; m2++)
                eff_mass_sqr[ijack][m1][m2]=eff_mass[ijack][m1][m2]*eff_mass[ijack][m1][m2];
    
    vvd_t M2_eff = get<0>(ave_err(eff_mass_sqr));
    vvd_t M2_eff_err = get<1>(ave_err(eff_mass_sqr));
    
    for(int m=0; m<_nm; m++)
        printf("m: %d \t M2_eff: %lg +- %lg\n",m,M2_eff[m][m],M2_eff_err[m][m]);
    
    //range for fit Zq
    int x_min_q=0;
    int x_max_q=_nm-1;
    
    // range for fit bilinears
    int x_min=0;
    int x_max=_nm*(_nm+1)/2-1;
    
    // number of fit parameters for sigma
    int npar_sigma=2;
    // number of fit parameters for bilinears
    int npar_bil[5]={3,2,3,2,2};
    int npar_bil_max=3;
    // number of fit parameters for meslep
    int npar_meslep[5]={2,2,3,3,2};
    int npar_meslep_max=3;
    
    //extrapolate sigma
    
    vvd_t sigma_pars_QCD(vd_t(0.0,npar_sigma),njacks);
    
//#pragma omp parallel for
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
        for(int iproj=0; iproj<sigma::nproj; iproj++)
            for(int r=0; r<_nr; r++)
                for(int ins=0; ins<sigma::nins; ins++) // not collapsed
                {
                    vvd_t coord_sigma(vd_t(0.0,_nm),npar_sigma);
                    
                    vvd_t sigma_r(vd_t(0.0,_nm),njacks);
                    vd_t sig_ave_r(0.0,_nmr), sqr_sig_ave_r(0.0,_nmr), sig_err_r(0.0,_nmr);
                    
                    for(int m=0; m<_nm; m++)
                    {
                        int mr = r + _nr*m;
                        
                        if(!UseEffMass)
                        {
                            coord_sigma[0][m] = 1.0;
                            coord_sigma[1][m] = mass_val[m];
                        }
                        else if(UseEffMass)
                        {
                            coord_sigma[0][m] = 1.0;
                            coord_sigma[1][m] = pow(M_eff[m][m],2.0);
                        }
                        
                        // subtraction of mass correction
                        if(ins==sigma::QED and UseEffMass)
                            for(int ijack=0;ijack<njacks;ijack++)
                            {
                                double b0 = sigma_pars_QCD[ijack][1];
                                double varb = 2.0*b0*eff_mass[ijack][m][m]*eff_mass_corr[ijack][m][m];
                                
                                sigma[ilinmom][iproj][ins][ijack][mr] -= varb;
                            }
                        
                        for(int ijack=0;ijack<njacks;ijack++)
                        {
                            sigma_r[ijack][m]=sigma[ilinmom][iproj][ins][ijack][mr];
                            
                            sig_ave_r[m] += sigma[ilinmom][iproj][ins][ijack][mr]/njacks;
                            sqr_sig_ave_r[m] +=
                                sigma[ilinmom][iproj][ins][ijack][mr]*
                                sigma[ilinmom][iproj][ins][ijack][mr]/njacks;
                        }
                        sig_err_r[m] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_sig_ave_r[mr]-sig_ave_r[mr]*sig_ave_r[mr]));
                    }
                    
                    vvd_t sigma_pars = polyfit(coord_sigma,npar_sigma,sig_err_r,sigma_r,x_min_q,x_max_q);
                    
                    //save fit parameters to be used to subtract dM
                    if(ins==sigma::LO)
                        for(int ijack=0;ijack<njacks;ijack++)
                            sigma_pars_QCD[ijack]=sigma_pars[ijack];
                    
                    for(int ijack=0; ijack<njacks; ijack++)
                        (out.sigma)[ilinmom][iproj][ins][ijack][r]=sigma_pars[ijack][0];
                }
    
    if(ntypes!=3)
    {
        out.deltam_computed=true;
        out.compute_deltam_from_prop();
    }
    
    out.compute_Zq();
    
    
    //extrapolate bilinears
    
    vvd_t gbil_pars_QCD(vd_t(0.0,npar_bil_max),njacks);
    
//#pragma omp parallel for //collapse(4)
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        for(int r1=0; r1<_nr; r1++)
            for(int r2=0; r2<_nr; r2++)
                for(int ibil=0;ibil<nbil;ibil++)
                    for(int ins=0; ins<gbil::nins; ins++) // not collapsed
                    {
                        vvd_t coord_bil(vd_t(0.0,_nm*(_nm+1)/2),npar_bil_max);
                        
                        vvd_t jG_r1_r2(vd_t(0.0,_nm*(_nm+1)/2),njacks);
                        
                        vd_t G_ave_r1_r2(0.0,_nm*(_nm+1)/2);
                        vd_t sqr_G_ave_r1_r2(0.0,_nm*(_nm+1)/2);
                        vd_t G_err_r1_r2(0.0,_nm*(_nm+1)/2);
                        
                        int ieq=0;
                        for(int m1=0; m1<_nm; m1++)
                            for(int m2=m1; m2<_nm; m2++)
                            {
                                
                                int mr1 = r1 + _nr*m1;
                                int mr2 = r2 + _nr*m2;
                                
                                if(!UseEffMass)
                                {
                                    coord_bil[0][ieq] = 1.0;
                                    // (am1+am2)
                                    coord_bil[1][ieq] = mass_val[m1]+mass_val[m2];
                                    // 1/(am1+am2)
                                    coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];
                                }
                                else if(UseEffMass)
                                {
                                    coord_bil[0][ieq] = 1.0;
                                    // M^2 (averaged over equivalent combinations)
                                    coord_bil[1][ieq] = pow((M_eff[m1][m2]+M_eff[m2][m1])/2.0,2.0);
                                    // 1/M^2
                                    coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];
                                }
                                
                                // subtraction of mass correction
                                if(ins==gbil::QED and UseEffMass)
                                    for(int ijack=0;ijack<njacks;ijack++)
                                    {
                                        double b0 = gbil_pars_QCD[ijack][1];
                                        double c0 = gbil_pars_QCD[ijack][2];
                                        
                                        double jM  = eff_mass[ijack][m1][m2];
                                        double jdM = eff_mass_corr[ijack][m1][m2];
                                        
                                        double varb = 2.0*b0*jM*jdM;
                                        double varc = -2.0*c0*jdM/(jM*jM*jM);
                                        
                                        jG[ibilmom][ins][ibil][ijack][mr1][mr2] -= varb + varc;
                                    }
                                
                                for(int ijack=0;ijack<njacks;ijack++)
                                {
                                    jG_r1_r2[ijack][ieq] = jG[ibilmom][ins][ibil][ijack][mr1][mr2];
                                    
                                    G_ave_r1_r2[ieq] += jG_r1_r2[ijack][ieq]/njacks;
                                    sqr_G_ave_r1_r2[ieq] += jG_r1_r2[ijack][ieq]*jG_r1_r2[ijack][ieq]/njacks;
                                }
                                G_err_r1_r2[ieq] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_G_ave_r1_r2[ieq]-G_ave_r1_r2[ieq]*G_ave_r1_r2[ieq]));
                                
                                ieq++;
                            }
                        
                        vvd_t jG_pars = polyfit(coord_bil,npar_bil[ibil],G_err_r1_r2,jG_r1_r2,x_min,x_max);
                        
                        //save fit parameters to be used to subtract dM
                        if(ins==gbil::LO)
                            for(int ijack=0;ijack<njacks;ijack++)
                            {
                                gbil_pars_QCD[ijack][0]=jG_pars[ijack][0];
                                gbil_pars_QCD[ijack][1]=jG_pars[ijack][1];
                                
                                if(npar_bil[ibil]<npar_bil_max)
                                    gbil_pars_QCD[ijack][2]=0.0;
                                else
                                    gbil_pars_QCD[ijack][2]=jG_pars[ijack][2];
                                
                            }
                        
                        for(int ijack=0;ijack<njacks;ijack++)
                            (out.jG)[ibilmom][ins][ibil][ijack][r1][r2] = jG_pars[ijack][0];
                    }
    
    out.compute_Zbil();
    
    
    if(compute_4f)
    {
        //extrapolate pr_meslep
        
        vvd_t pr_meslep_pars_QCD(vd_t(0.0,npar_meslep_max),njacks);
        
//#pragma omp parallel for //collapse(6)
        for(int imom=0;imom<_meslepmoms;imom++)
            for(int r1=0; r1<_nr; r1++)
                for(int r2=0; r2<_nr; r2++)
                    for(int iop1=0;iop1<nbil;iop1++)
                        for(int iop2=0;iop2<nbil;iop2++)
                            for(int ins=0; ins<pr_meslep::nins; ins++)
                            {
                                vvd_t coord_meslep(vd_t(0.0,_nm*(_nm+1)/2),npar_meslep_max);
                                
                                vvd_t jpr_meslep_r1_r2(vd_t(0.0,_nm*(_nm+1)/2),njacks);
                                
                                vd_t pr_meslep_ave_r1_r2(0.0,_nm*(_nm+1)/2);
                                vd_t sqr_pr_meslep_ave_r1_r2(0.0,_nm*(_nm+1)/2);
                                vd_t pr_meslep_err_r1_r2(0.0,_nm*(_nm+1)/2);
                                
                                int ieq=0;
                                
                                for(int m1=0; m1<_nm; m1++)
                                    for(int m2=m1; m2<_nm; m2++)
                                    {
                                        int mr1 = r1 + _nr*m1;
                                        int mr2 = r2 + _nr*m2;
                                        
                                        if(!UseEffMass)
                                        {
                                            coord_meslep[0][ieq] = 1.0;
                                            // (am1+am2)
                                            coord_meslep[1][ieq] = mass_val[m1]+mass_val[m2];
                                            // 1/(am1+am2)
                                            coord_meslep[2][ieq] = 1.0/coord_meslep[1][ieq];
                                        }
                                        else if(UseEffMass)
                                        {
                                            coord_meslep[0][ieq] = 1.0;
                                            // M^2 (averaged over equivalent combinations)
                                            coord_meslep[1][ieq] = pow((M_eff[m1][m2]+M_eff[m2][m1])/2.0,2.0);
                                            // 1/M^2
                                            coord_meslep[2][ieq] = 1.0/coord_meslep[1][ieq];
                                        }
                                        
                                        // subtraction of mass correction
                                        if(ins==pr_meslep::QED and UseEffMass)
                                            for(int ijack=0;ijack<njacks;ijack++)
                                            {
                                                double b0 = pr_meslep_pars_QCD[ijack][1];
                                                double c0 = pr_meslep_pars_QCD[ijack][2];
                                                
                                                double jM  = eff_mass[ijack][m1][m2];
                                                double jdM = eff_mass_corr[ijack][m1][m2];
                                                
                                                double varb = 2.0*b0*jM*jdM;
                                                double varc = -2.0*c0*jdM/(jM*jM*jM);
                                                
//                                                jpr_meslep[imom][ins][iop1][iop2][ijack][mr1][mr2] -= varb + varc;
                                                jpr_meslep_r1_r2[ijack][ieq] -= varb + varc;
                                            }
                                        
                                    
                                        for(int ijack=0;ijack<njacks;ijack++)
                                        {
                                            // the average on the masses restores parity
                                            jpr_meslep_r1_r2[ijack][ieq] +=
//                                                    jpr_meslep[imom][ins][iop1][iop2][ijack][mr1][mr2];
                                                (jpr_meslep[imom][ins][iop1][iop2][ijack][mr1][mr2]+
                                                 jpr_meslep[imom][ins][iop1][iop2][ijack][mr2][mr1])/2.0;
                                            
                                            pr_meslep_ave_r1_r2[ieq] +=
                                                jpr_meslep_r1_r2[ijack][ieq]/njacks;
                                            sqr_pr_meslep_ave_r1_r2[ieq] +=
                                                jpr_meslep_r1_r2[ijack][ieq]*jpr_meslep_r1_r2[ijack][ieq]/njacks;
                                        }
                                        pr_meslep_err_r1_r2[ieq] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_pr_meslep_ave_r1_r2[ieq]-pr_meslep_ave_r1_r2[ieq]*pr_meslep_ave_r1_r2[ieq]));
                                        
                                        ieq++;
                                    }
                                
                                int npar_combined = max(npar_meslep[iop1],npar_meslep[iop2]);
                                
                                vvd_t jpr_meslep_pars = polyfit(coord_meslep,npar_combined,pr_meslep_err_r1_r2,jpr_meslep_r1_r2,x_min,x_max);
                                
                                //save fit parameters to be used to subtract dM
                                if(ins==pr_meslep::LO)
                                    for(int ijack=0;ijack<njacks;ijack++)
                                    {
                                        pr_meslep_pars_QCD[ijack][0]=jpr_meslep_pars[ijack][0];
                                        pr_meslep_pars_QCD[ijack][1]=jpr_meslep_pars[ijack][1];
                                        
                                        if(npar_combined<npar_meslep_max)
                                            pr_meslep_pars_QCD[ijack][2]=0.0;
                                        else
                                            pr_meslep_pars_QCD[ijack][2]=jpr_meslep_pars[ijack][2];
                                    }
            
                                for(int ijack=0;ijack<njacks;ijack++)
                                   (out.jpr_meslep)[imom][ins][iop1][iop2][ijack][r1][r2] = jpr_meslep_pars[ijack][0];
                            
                                
                            }
        
        out.compute_Z4f();
    
    }

    return out;
}

//oper_t oper_t::subtract()
//{
//    cout<<endl;
//    cout<<"----- subtraction of O(a2) effects -----"<<endl<<endl;
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

oper_t chiral_sea_extr(voper_t in)
{
    cout<<endl;
    cout<<"----- chiral sea extrapolation -----"<<endl<<endl;
    
    oper_t out=in[0];  //?
    
    int nmSea = in[0]._nm_Sea;
    int _linmoms = in[0]._linmoms;
    int _bilmoms = in[0]._bilmoms;
    int _meslepmoms = in[0]._meslepmoms;
    
    out.allocate_val();
    out.allocate();
    
    out.path_to_ens = in[0].path_to_beta;

    vd_t x(0.0,nmSea);
    
    vvvvd_t dy_sigma(vvvd_t(vvd_t(vd_t(0.0,nmSea),sigma::nins),sigma::nproj),_linmoms);
    vvvvd_t dy_G(vvvd_t(vvd_t(vd_t(0.0,nmSea),nbil),gbil::nins),_bilmoms);
    vvvvvd_t dy_meslep(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmSea),nbil),nbil),pr_meslep::nins),_meslepmoms);

    vvvvvd_t y_sigma(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmSea),njacks),sigma::nins),sigma::nproj),_linmoms);
    vvvvvd_t y_G(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmSea),njacks),nbil),gbil::nins),_bilmoms);
    vvvvvvd_t y_meslep(vvvvvd_t(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmSea),njacks),nbil),nbil),pr_meslep::nins),_meslepmoms);

    
    // range for fit
    int x_min=0;
    int x_max=nmSea-1;
    
    for(int msea=0; msea<nmSea; msea++)
        x[msea] = get<0>(ave_err(in[msea].eff_mass_sea));
    
    // extrapolate sigma
#pragma omp parallel for collapse(3)
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
        for(int iproj=0; iproj<sigma::nproj; iproj++)
            for(int ins=0; ins<sigma::nins; ins++)
            {
                vvd_t coord_q(vd_t(0.0,nmSea),2); // coords at fixed r
                
                vvd_t y_sigma(vd_t(0.0,nmSea),njacks);
                vd_t dy_sigma(0.0,nmSea);
                
                for(int msea=0; msea<nmSea; msea++)
                {
                    coord_q[0][msea] = 1.0;
                    if(!UseEffMass)
                    {
                        cout<<" Impossible to extrapolate without using the effective mass. "<<endl;
                        exit(0);
                        //      coord_q[1][m]= mass_val[m];
                    }
                    else if(UseEffMass)
                        coord_q[1][msea] = pow(x[msea],2.0);
                    
                    for(int ijack=0;ijack<njacks;ijack++)
                        y_sigma[ijack][msea] = in[msea].sigma[ilinmom][iproj][ins][ijack][0];
                    
                    dy_sigma[msea] = (get<1>(ave_err(in[msea].sigma)))[ilinmom][iproj][ins][0];
                }
                
                vvd_t sigma_pars = polyfit(coord_q,2,dy_sigma,y_sigma,x_min,x_max);
                
                for(int ijack=0; ijack<njacks; ijack++)
                    (out.sigma)[ilinmom][iproj][ins][ijack][0]=sigma_pars[ijack][0];
            }
    
    if(ntypes!=3)
    {
        out.deltam_computed=true;
        out.compute_deltam_from_prop();
    }
    
    out.compute_Zq();
    
    // extrapolate bilinears
#pragma omp parallel for collapse(3)
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        for(int ins=0; ins<gbil::nins; ins++)
            for(int ibil=0;ibil<nbil;ibil++)
            {
                vvd_t coord_bil(vd_t(0.0,nmSea),2); // linear fit in sea extrapolation
                
                vvd_t y_G(vd_t(0.0,nmSea),njacks);
                vd_t dy_G(0.0,nmSea);
                
                for(int msea=0; msea<nmSea; msea++)
                {
                    coord_bil[0][msea] = 1.0;
                    if(!UseEffMass)
                    {
                        cout<<" Impossible to extrapolate without using the effective mass. "<<endl;
                        exit(0);
                        //                coord_bil[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
                        //                coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];    // 1/(am1+am2)
                    }
                    else if(UseEffMass)
                        coord_bil[1][msea] = pow(x[msea],2.0);

                    for(int ijack=0;ijack<njacks;ijack++)
                        y_G[ijack][msea] = in[msea].jG[ibilmom][ins][ibil][ijack][0][0];
                    
                    dy_G[msea] = (get<1>(ave_err(in[msea].jG)))[ibilmom][ins][ibil][0][0];
                }
                
                vvd_t jG_pars = polyfit(coord_bil,2,dy_G,y_G,x_min,x_max);
                
                for(int ijack=0;ijack<njacks;ijack++)
                    (out.jG)[ibilmom][ins][ibil][ijack][0][0] = jG_pars[ijack][0];
            }
    
    out.compute_Zbil();
    
    if(compute_4f)
    {
        // extrapolate meslep
#pragma omp parallel for collapse(4)
        for(int imom=0;imom<_meslepmoms;imom++)
            for(int ins=0; ins<gbil::nins; ins++)
                for(int iop1=0;iop1<nbil;iop1++)
                    for(int iop2=0;iop2<nbil;iop2++)
                    {
                        vvd_t coord_meslep(vd_t(0.0,nmSea),2); // linear fit in sea extrapolation
                        
                        vvd_t y_meslep(vd_t(0.0,nmSea),njacks);
                        vd_t dy_meslep(0.0,nmSea);
                        
                        for(int msea=0; msea<nmSea; msea++)
                        {
                            coord_meslep[0][msea] = 1.0;
                            if(!UseEffMass)
                            {
                                cout<<" Impossible to extrapolate without using the effective mass. "<<endl;
                                exit(0);
                                //                coord_meslep[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
                                //                coord_meslep[2][ieq] = 1.0/coord_bil[1][ieq];    // 1/(am1+am2)
                            }
                            else if(UseEffMass)
                                coord_meslep[1][msea] = pow(x[msea],2.0);
                            
                            for(int ijack=0;ijack<njacks;ijack++)
                                y_meslep[ijack][msea] = in[msea].jpr_meslep[imom][ins][iop1][iop2][ijack][0][0];
                            
                            dy_meslep[msea] = (get<1>(ave_err(in[msea].jpr_meslep)))[imom][ins][iop1][iop2][0][0];
                        }
                        
                        vvd_t jmeslep_pars = polyfit(coord_meslep,2,dy_meslep,y_meslep,x_min,x_max);
                        
                        for(int ijack=0;ijack<njacks;ijack++)
                            (out.jpr_meslep)[imom][ins][iop1][iop2][ijack][0][0] = jmeslep_pars[ijack][0];
                    }
        
        out.compute_Z4f();
    }
    
    return out;
}

oper_t theta_average( voper_t in)
{
    cout<<endl;
    cout<<"----- theta average -----"<<endl<<endl;
    
    oper_t out=in[0];  //?
    
    int _linmoms = in[0]._linmoms;
    int _bilmoms = in[0]._bilmoms;
    int _meslepmoms = in[0]._meslepmoms;
    
    out.allocate_val();
    out.allocate();
    
    out.path_to_ens = in[0].path_to_beta;
    
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            out.jZq[ilinmom][ijack][0] = 0.5*(in[0].jZq[ilinmom][ijack][0] + in[1].jZq[ilinmom][ijack][0]);
            out.jZq_EM[ilinmom][ijack][0] = 0.5*(in[0].jZq_EM[ilinmom][ijack][0] + in[1].jZq_EM[ilinmom][ijack][0]);
        }
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        for(int ibil=0;ibil<nbil;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                out.jZ[ibilmom][ibil][ijack][0][0] =
                    0.5*(in[0].jZ[ibilmom][ibil][ijack][0][0] + in[1].jZ[ibilmom][ibil][ijack][0][0]);
                out.jZ_EM[ibilmom][ibil][ijack][0][0] =
                    0.5*(in[0].jZ_EM[ibilmom][ibil][ijack][0][0] + in[1].jZ_EM[ibilmom][ibil][ijack][0][0]);
            }
    for(int imom=0;imom<_meslepmoms;imom++)
        for(int iop1=0;iop1<nbil;iop1++)
            for(int iop2=0;iop2<nbil;iop2++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                out.jZ_4f[imom][iop1][iop2][ijack][0][0] =
                    0.5*(in[0].jZ_4f[imom][iop1][iop2][ijack][0][0] + in[1].jZ_4f[imom][iop1][iop2][ijack][0][0]);
                out.jZ_4f_EM[imom][iop1][iop2][ijack][0][0] =
                    0.5*(in[0].jZ_4f_EM[imom][iop1][iop2][ijack][0][0] + in[1].jZ_4f_EM[imom][iop1][iop2][ijack][0][0]);
            }

    return out;
}


//oper_t oper_t::evolve(const int b)
//{
//    cout<<endl;
//    cout<<"----- evolution to the scale 1/a -----"<<endl<<endl;
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

int mom_list_xyz(vector<coords_t> &mom_list, const size_t imom)
{
    return abs(mom_list[imom][1])*abs(mom_list[imom][2])*abs(mom_list[imom][3]);
}

oper_t oper_t::average_equiv_moms()
{
    cout<<endl;
    cout<<"Averaging over the equivalent momenta -- ";
    
    oper_t out=(*this);
    
    // Find equivalent linmoms
    int tag=0, tag_aux=0;
    double eps=1.0e-15;
    
    vector<int> tag_lin_vector;
    tag_lin_vector.push_back(0);
    
    // Tag assignment to linmoms
    for(int imom=0;imom<_linmoms;imom++)
    {
        int count_no=0;
        
        for(int j=0;j<imom;j++)
        {
            bool cond{2.0*abs(p2_tilde[j]-p2_tilde[imom])<eps*(p2_tilde[j]+p2_tilde[imom]) &&
                      mom_list_xyz(mom_list,j)==mom_list_xyz(mom_list,imom) &&
                      2.0*abs(abs(p[j][0])-abs(p[imom][0]))<eps*(abs(p[j][0])+abs(p[imom][0]))};
            if(cond)
                tag_aux = tag_lin_vector[j];
            else
                count_no++;
            
            if(count_no==imom)
            {
                tag++;
                tag_lin_vector.push_back(tag);
            }
            else if(j==imom-1)
                tag_lin_vector.push_back(tag_aux);
        }
    }
    
    // number of equivalent linmoms
    int neq_lin_moms = tag+1;
    
    out._linmoms=neq_lin_moms;
    cout<<"found: "<<out._linmoms<<" equivalent linmoms ";
    (out.linmoms).resize(out._linmoms);
    
    vector<double> p2_eqmoms(out._linmoms,0.0);
    vector<double> p2_tilde_eqmoms(out._linmoms,0.0);

    // count the different tags
    vector<int> count_tag_lin_vector(out._linmoms);
    int count=0;
    for(int tag=0;tag<out._linmoms;tag++)
    {
        count=0;
        for(int imom=0;imom<_linmoms;imom++)
            if(tag_lin_vector[imom]==tag) count++;
        
        count_tag_lin_vector[tag]=count;
    }
    
    for(int tag=0;tag<out._linmoms;tag++)
        for(int imom=0;imom<_linmoms;imom++)
            if(tag_lin_vector[imom]==tag)
            {
                // fill the new linmoms and p2tilde
                out.linmoms[tag] = {tag};
                p2_eqmoms[tag] = p2[imom];
                p2_tilde_eqmoms[tag] = p2_tilde[imom];
            }
    
    out.p2=p2_eqmoms;
    out.p2_tilde=p2_tilde_eqmoms;
    
    print_vec(p2_eqmoms,path_print+"p2_eqmoms.txt");
    print_vec(p2_tilde_eqmoms,path_print+"p2_tilde_eqmoms.txt");
    
    // Find equivalent bilmoms
    tag=0, tag_aux=0;
    
    vector<int> tag_bil_vector;
    tag_bil_vector.push_back(0);
    
    //Tag assignment to bilmoms
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        int count_no=0;
        
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        for(int j=0;j<ibilmom;j++)
        {
            const int imomA=bilmoms[j][1]; // p1
            const int imomB=bilmoms[j][2]; // p2
            
            const bool cond{(tag_lin_vector[imom1]==tag_lin_vector[imomA] and
                             tag_lin_vector[imom2]==tag_lin_vector[imomB]) or
                            (tag_lin_vector[imom1]==tag_lin_vector[imomB] and
                             tag_lin_vector[imom2]==tag_lin_vector[imomA])};
            
//            const bool cond{tag_lin_vector[imom1]+tag_lin_vector[imom2]==tag_lin_vector[imomA]+tag_lin_vector[imomB] and
//                            tag_lin_vector[imom1]*tag_lin_vector[imom2]==tag_lin_vector[imomA]*tag_lin_vector[imomB]};
            
            if(cond)
                tag_aux=tag_bil_vector[j];
            else
                count_no++;
            
            if(count_no==ibilmom)
            {
                tag++;
                tag_bil_vector.push_back(tag);
            }
            else if(j==ibilmom-1)
            {
                tag_bil_vector.push_back(tag_aux);
            }
        }
    }
    
    // number of equivalent bilmoms
    int neq_bil_moms = tag+1;
    
    out._bilmoms=neq_bil_moms;
    cout<<"and "<<neq_bil_moms<<" equivalent bilmoms "<<endl<<endl;
    (out.bilmoms).resize(out._bilmoms);
    
    // number of equivalent meslepmoms
    int neq_meslep_moms = neq_bil_moms;
    out._meslepmoms=neq_meslep_moms;
    
    // count the different tags
    vector<int> count_tag_bil_vector(out._bilmoms);
    count=0;
    for(int tag=0;tag<out._bilmoms;tag++)
    {
        count=0;
        for(int imom=0;imom<_bilmoms;imom++)
            if(tag_bil_vector[imom]==tag) count++;

        count_tag_bil_vector[tag]=count;
    }
    
    for(int tag=0;tag<out._bilmoms;tag++)
        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
            if(tag_bil_vector[ibilmom]==tag)
            {
                // fill the new bilmoms
                const int imom0=bilmoms[tag][0]; // k
                const int imom1=bilmoms[tag][1]; // p1
                const int imom2=bilmoms[tag][2]; // p2
                
                out.bilmoms[tag] = {imom0,imom1,imom2};
            }
    
    out.allocate_val();
    
    out.eff_mass = eff_mass;
    out.eff_mass_sea = eff_mass_sea;
    
    if(!load_ave)
    {
        out.allocate();
        
        for(int tag=0;tag<neq_lin_moms;tag++)
            for(int imom=0;imom<_linmoms;imom++)
                if(tag_lin_vector[imom]==tag)
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int mr=0;mr<_nmr;mr++)
                            for(int iproj=0;iproj<sigma::nproj;iproj++)
                                for(int ins=0;ins<sigma::nins;ins++)
                                    (out.sigma)[tag][iproj][ins][ijack][mr]+=
                                    sigma[imom][iproj][ins][ijack][mr]/count_tag_lin_vector[tag];
        
        if(ntypes!=3)
        {
            out.deltam_computed=true;
            out.compute_deltam_from_prop();
        }
        
        out.compute_Zq();
        
        for(int tag=0;tag<neq_bil_moms;tag++)
            for(int imom=0;imom<_bilmoms;imom++)
                if(tag_bil_vector[imom]==tag)
                    for(int ibil=0;ibil<5;ibil++)
                        for(int ijack=0;ijack<njacks;ijack++)
                            for(int ins=0;ins<gbil::nins;ins++)
                                for(int mr1=0; mr1<_nmr; mr1++)
                                    for(int mr2=0; mr2<_nmr; mr2++)
                                        (out.jG)[tag][ins][ibil][ijack][mr1][mr2]+=
                                        jG[imom][ins][ibil][ijack][mr1][mr2]/count_tag_bil_vector[tag];
        
        
        out.compute_Zbil();
        
        if(compute_4f)
            for(int tag=0;tag<neq_bil_moms;tag++)
                for(int imom=0;imom<_bilmoms;imom++)
                    if(tag_bil_vector[imom]==tag)
                        for(int iop1=0;iop1<nbil;iop1++)
                            for(int iop2=0;iop2<nbil;iop2++)
                                for(int ijack=0;ijack<njacks;ijack++)
                                    for(int ins=0;ins<pr_meslep::nins;ins++)
                                        for(int mr1=0; mr1<_nmr; mr1++)
                                            for(int mr2=0; mr2<_nmr; mr2++)
                                                (out.jpr_meslep)[tag][ins][iop1][iop2][ijack][mr1][mr2]+=
                                                jpr_meslep[imom][ins][iop1][iop2][ijack][mr1][mr2]/count_tag_bil_vector[tag];
        
        out.compute_Z4f();
        
        //// DEBUG /////
//        Z4f_tup Z_4f_ave_err = ave_err_Z4f(out.jZ_4f);
//        Z4f_tup Z_4f_EM_ave_err = ave_err_Z4f(out.jZ_4f_EM);
//        
//        vvvvvd_t Z_4f_ave=get<0>(Z_4f_ave_err);  //[imom][iop1][iop2][mr1][mr2];
//        vvvvvd_t Z_4f_EM_ave=get<0>(Z_4f_EM_ave_err);
//        
//        vvvvvd_t Z_4f_err=get<1>(Z_4f_ave_err);  //[imom][iop1][iop2][mr1][mr2];
//        vvvvvd_t Z_4f_EM_err=get<1>(Z_4f_EM_ave_err);
//        
//        vector<double> p2t;
//        
//        if(out._linmoms==moms)
//        {
//            p2t.resize(out._linmoms);
//            read_vec(p2t,path_print+"p2_tilde.txt");
//        }
//        else
//        {
//            p2t.resize(out._linmoms);
//            read_vec(p2t,path_print+"p2_tilde_eqmoms.txt");
//        }
//        
//        for(int i=0;i<nbil*nbil;i++)
//        {
//            int iop2=i%nbil;
//            int iop1=(i-iop2)/nbil;
//            
//            for(int m1=0; m1<_nm; m1++)
//                for(int m2=0; m2<_nm; m2++)
//                    for(int r1=0; r1<_nr; r1++)
//                        for(int r2=0; r2<_nr; r2++)
//                        {
//                            int mr1 = r1 + _nr*m1;
//                            int mr2 = r2 + _nr*m2;
//                            
//                            cout<<"------------------------"<<endl;
//                            cout<<"iop=["<<iop1<<","<<iop2<<"]  mass=["<<m1<<","<<m2<<"]  r=["<<r1<<","<<r2<<"] "<<endl;
//                            cout<<"------------------------"<<endl;
//                            
//                            for(int imom=0; imom<out._bilmoms; imom++)
//                            {
//                                //            int imomq = in.bilmoms[imom][0];
//                                //            cout<<"imomq: "<<imomq<<endl;
//                                //            int imomk = in.linmoms[imomq][0];
//                                int imomk = imom;   // NB: it works only for RIMOM!
//                                
//                                cout.precision(16);
//                                cout<<p2t[imomk]<<"\t"<<Z_4f_ave[imom][iop1][iop2][mr1][mr2]<<"\t"<<Z_4f_err[imom][iop1][iop2][mr1][mr2]<<endl;
//                            }
//                            
//                        }
//        } //end debug
    }
    return out;
}

oper_t compute_eta(voper_t in) // in[loop]
{
    cout<<endl;
    cout<<"----- eta -----"<<endl<<endl;
    
    oper_t out=in[1]; // out
    
    out._nmr = in[1]._nmr;
    out._linmoms = in[1]._linmoms;
    out._bilmoms = in[1]._bilmoms;
    out._meslepmoms = in[1]._meslepmoms;
    
    out.allocate_val();
    out.allocate();
    
    out.path_to_ens = in[1].path_to_beta + in[1].ensemble_name + "/";
    
    // Zq
    for(int imom=0;imom<out._linmoms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<out._nmr;mr++)
                (out.jZq)[imom][ijack][mr] =
                    (in[0].jZq_EM)[imom][ijack][mr] -
                    (in[1].jZq_EM)[imom][ijack][mr];
    // Zbil
    for(int imom=0;imom<out._bilmoms;imom++)
        for(int ibil=0;ibil<nbil;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr1=0;mr1<out._nmr;mr1++)
                    for(int mr2=0;mr2<out._nmr;mr2++)
                        (out.jZ_EM)[imom][ibil][ijack][mr1][mr2] =
                            (in[0].jZ_EM)[imom][ibil][ijack][mr1][mr2] -
                            (in[1].jZ_EM)[imom][ibil][ijack][mr1][mr2];
    
    // Z4f
    for(int imom=0;imom<out._meslepmoms;imom++)
        for(int iop1=0;iop1<nbil;iop1++)
            for(int iop2=0;iop2<nbil;iop2++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr1=0;mr1<out._nmr;mr1++)
                        for(int mr2=0;mr2<out._nmr;mr2++)
                            (out.jZ_4f_EM)[imom][iop1][iop2][ijack][mr1][mr2] =
                                (in[0].jZ_4f_EM)[imom][iop1][iop2][ijack][mr1][mr2] -
                                (in[1].jZ_4f_EM)[imom][iop1][iop2][ijack][mr1][mr2];
    
    
    return out;
}

void oper_t::print(const string suffix)
{
    print_vec_bin(sigma,path_print+"sigmas_"+suffix);
    print_vec_bin(jG,path_print+"jG_"+suffix);
    print_vec_bin(jpr_meslep,path_print+"jpr_meslep_"+suffix);
}

void oper_t::printZ(const string suffix)
{
    print_vec_bin(jZq,path_print+"jZq_"+suffix);
    print_vec_bin(jZ,path_print+"jZ_"+suffix);
    print_vec_bin(jZ_4f,path_print+"jZ_4f_"+suffix);
    print_vec_bin(jZq_EM,path_print+"jZq_EM_"+suffix);
    print_vec_bin(jZ_EM,path_print+"jZ_EM_"+suffix);
    print_vec_bin(jZ_4f_EM,path_print+"jZ_4f_EM_"+suffix);
}

void oper_t::load(const string suffix)
{
    (*this).allocate();
    
    ifstream sigma_data(path_print+"sigmas_"+suffix);
    ifstream jG_data(path_print+"jG_"+suffix);
    ifstream jpr_meslep_data(path_print+"jpr_meslep_"+suffix);
    
    if(sigma_data.good() and jG_data.good() and jpr_meslep_data.good())
    {
        cout<<"Loading averaged quantities from files in \""<<path_print<<"\"."<<endl;
        read_vec_bin(sigma,path_print+"sigmas_"+suffix);
        read_vec_bin(jG,path_print+"jG_"+suffix);
        read_vec_bin(jpr_meslep,path_print+"jpr_meslep_"+suffix);
    }
    else
    {
        cout<<"Cannot open files of "<<suffix<<" quantities in \""<<path_print<<"\"."<<endl;
        exit(0);
    }
    
    if(suffix.compare("chir")==1)
    {
        ifstream jZq_data(path_print+"jZq_"+suffix), jZq_EM_data(path_print+"jZq_EM_"+suffix);
        ifstream jZ_data(path_print+"jZ_"+suffix),  jZ_EM_data(path_print+"jZ_EM_"+suffix);
        ifstream jZ_4f_data(path_print+"jZ_4f_"+suffix),  jZ_4f_EM_data(path_print+"jZ_4f_EM_"+suffix);
        
        if(jZq_data.good() and jZ_data.good() and jZ_4f_data.good() and
           jZq_EM_data.good() and jZ_EM_data.good() and jZ_4f_EM_data.good())
        {
            cout<<"Loading Z from files in \""<<path_print<<"\"."<<endl;
            read_vec_bin(jZq,path_print+"jZq_"+suffix);
            read_vec_bin(jZ,path_print+"jZ_"+suffix);
            read_vec_bin(jZ_4f,path_print+"jZ_4f_"+suffix);
            read_vec_bin(jZq_EM,path_print+"jZq_EM_"+suffix);
            read_vec_bin(jZ_EM,path_print+"jZ_EM_"+suffix);
            read_vec_bin(jZ_4f_EM,path_print+"jZ_4f_EM_"+suffix);
        }
        else
        {
            cout<<"Cannot open files of "<<suffix<<" Z in \""<<path_print<<"\"."<<endl;
            exit(0);
        }
    }
    
    (*this).compute_Zq();
    (*this).compute_Zbil();
    (*this).compute_Z4f();
}

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
    
    Zq_tup Zq_ave_err = ave_err_Zq(in.jZq);
    Zq_tup Zq_EM_ave_err = ave_err_Zq(in.jZq_EM);
    
    vvd_t Zq_ave = get<0>(Zq_ave_err);        //[imom][mr]
    vvd_t Zq_EM_ave = get<0>(Zq_EM_ave_err);
    
    vvd_t Zq_err = get<1>(Zq_ave_err);        //[imom][mr]
    vvd_t Zq_EM_err = get<1>(Zq_EM_ave_err);
    
    Zbil_tup Zbil_ave_err = ave_err_Z(in.jZ);
    Zbil_tup Zbil_EM_ave_err = ave_err_Z(in.jZ_EM);
    
    vvvvd_t Z_ave = get<0>(Zbil_ave_err);    //[imom][ibil][mr1][mr2]
    vvvvd_t Z_EM_ave = get<0>(Zbil_EM_ave_err);
    
    vvvvd_t Z_err = get<1>(Zbil_ave_err);    //[imom][ibil][mr1][mr2]
    vvvvd_t Z_EM_err = get<1>(Zbil_EM_ave_err);
    
    Z4f_tup Z_4f_ave_err = ave_err_Z4f(in.jZ_4f);
    Z4f_tup Z_4f_EM_ave_err = ave_err_Z4f(in.jZ_4f_EM);

    vvvvvd_t Z_4f_ave=get<0>(Z_4f_ave_err);  //[imom][iop1][iop2][mr1][mr2];
    vvvvvd_t Z_4f_EM_ave=get<0>(Z_4f_EM_ave_err);
    
    vvvvvd_t Z_4f_err=get<1>(Z_4f_ave_err);  //[imom][iop1][iop2][mr1][mr2];
    vvvvvd_t Z_4f_EM_err=get<1>(Z_4f_EM_ave_err);
    
    // this choice is relative to the twisted basis
    vector<string> bil={"S","V","P","A","T"};
    
    ofstream Zq_data, Zq_EM_data;
    vector<ofstream> Zbil_data(nbil), Zbil_EM_data(nbil);
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
    
    cout<<"Plotting Zq";
    for(int imom=0; imom<in._linmoms; imom++)
    {
        Zq_data<<p2t[imom]<<"\t"<<Zq_ave[imom][0]<<"\t"<<Zq_err[imom][0]<<endl;
        Zq_EM_data<<p2t[imom]<<"\t"<<Zq_EM_ave[imom][0]<<"\t"<<Zq_EM_err[imom][0]<<endl;
    }
    
    cout<<", Zbil";
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
        cout<<" and Z4f :"<<endl;
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
    else
        cout<<" :"<<endl;
    
    cout<<"\""<<path_to_ens<<"plots\""<<endl;
    
    
}
