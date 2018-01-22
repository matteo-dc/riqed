#ifdef HAVE_CONFIG_H
#include <config.hpp>
#endif

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "aliases.hpp"
#include "read.hpp"
#include "read_input.hpp"
#include "Dirac.hpp"
#include "global.hpp"
#include "operations.hpp"
#include "print.hpp"

#include "fit.hpp"

ORDER ord;

// mom lists
vector<coords_t> mom_list;
vector<p_t> p, p_tilde;
vector<double> p2, p2_tilde;
vector<double> p4, p4_tilde;

// list of N(p)
vector<int> Np;

// effective mass
vector< vector<double> > eff_mass;

// deltam_cr
vvvd_t deltam_cr;

//using namespace std;

int main()
{
    omp_set_nested(1);
#pragma omp parallel
#pragma omp master
    {
        system("clear");
        cout<<"Using "<<omp_get_num_threads()<<" threads"<<endl;
    }
    
    char path[128]="input.txt";
    read_input(path);
    
    //read mom list
    read_mom_list(mom_path);
    moms=mom_list.size();
    cout<<"Read: "<<mom_list.size()<<" momenta from \""<<mom_path<<"\" (BC: "<<BC_str<<")."<<endl<<endl;
    
    oper_t basic;
    
    PRINT(p2_tilde);
    
    deltam_cr=read_deltam_cr("deltam_cr_array");
    
    basic.step="basic";
    basic.create_basic();
    
    if(UseEffMass) eff_mass=read_eff_mass("eff_mass_array");
    
    oper_t rave = basic.average_r();
    
    oper_t rave_chir = rave.chiral_extr();
    
    oper_t rave_chir_sub = rave_chir.subtract();
    
    oper_t rave_chir_sub_evo = rave_chir_sub.evolve();
    
    oper_t rave_chir_sub_evo_ave = rave_chir_sub_evo.average_equiv_moms();
    
//    for(int tag=0;tag<(int)(rave_chir_sub_evo_ave.jZq_evo).size();tag++)
//        for(int ijack=0;ijack<njacks;ijack++)
//            cout<<"mom: "<<tag<<" ijack: "<<ijack<<" "<<(rave_chir_sub_evo_ave.jZq_evo)[tag][ijack]<<endl;
//    cout<<endl;
    
//    for(int ijack=0;ijack<njacks;ijack++)
//        cout<<"mom: "<<0<<" ijack: "<<ijack<<" "<<(rave_chir_sub_evo.jZq_evo)[0][ijack]<<endl;
    
    
    continuum_limit(rave_chir_sub_evo_ave,LO);
    continuum_limit(rave_chir_sub_evo_ave,EM);
  
    vector<vd_t> a=rave_chir_sub_evo_ave.jZq_evo;
    
    PRINT(a);
    
    

    //    int neq_moms = (rave_chir_sub_evo_ave.jZq_evo).size();
    
    //    cout<<"ZQ EVOLVED (ALLMOMS)"<<endl;
//    for (int imom=0; imom<moms; imom++)
//        for (int ijack=0; ijack<njacks; ijack++)
//        {
//            cout<<"imom: "<<imom<<" ijack: "<<ijack<<"  "<<rave_chir_sub_evo.jZq_evo[imom][ijack]<<endl;
//        }
//    cout<<endl;
//    cout<<"ZQ EVOLVED (EQMOMS)"<<endl;
//    for (int imom=0; imom<neq_moms; imom++)
//        for (int ijack=0; ijack<njacks; ijack++)
//        {
//            cout<<"imom: "<<imom<<" ijack: "<<ijack<<"  "<<rave_chir_sub_evo_ave.jZq_evo[imom][ijack]<<endl;
//        }
    
    
//    cout<<"Zq  "<<(sub.jZq_sub)[0][0]<<"  "<<(evo.jZq_evo)[0][0]<<endl;
//    cout<<"Zs  "<<sub.jZ_sub[0][0][0]<<"  "<<(evo.jZ_evo)[0][0][0]<<endl;
//    cout<<"Za  "<<sub.jZ_sub[0][1][0]<<"  "<<(evo.jZ_evo)[0][1][0]<<endl;
//    cout<<"Zp  "<<sub.jZ_sub[0][2][0]<<"  "<<(evo.jZ_evo)[0][2][0]<<endl;
//    cout<<"Zv  "<<sub.jZ_sub[0][3][0]<<"  "<<(evo.jZ_evo)[0][3][0]<<endl;
//    cout<<"Zt  "<<sub.jZ_sub[0][4][0]<<"  "<<(evo.jZ_evo)[0][4][0]<<endl;
    
//    for(int m=0;m<neq2;m++)
//        cout<<m<<"  basic: "<<basic.m_eff_equivalent_Zq[m]<<endl;
//    for(int m=0;m<neq2;m++)
//        cout<<m<<"  rave: "<<rave.m_eff_equivalent_Zq[m]<<endl;
//    
//    cout<<"basic ave: "<<((basic.jG_0)[0][0][0][0][0]+(basic.jG_0)[0][0][1][1][0])/2.0<<endl;
//    cout<<"rave: "<<(rave.jG_0_ave_r)[0][0][0][0]<<endl;
    
    
//    cout<<get<STEP>(basic.Zq[0])<<endl;
    
//    vvd_t coord(vd_t(0.0,5),2);
//    vvd_t y(vd_t(0.0,5),njacks);
//    vd_t err(0.0,5);
//    for(int i=0; i<5; i++)
//    {
//        coord[0][i] = 1.0;  //costante
//        coord[1][i] = i+1;   //M^2
//        
//        for(int ijack=0;ijack<njacks;ijack++)
//        {
//            y[ijack][i] = 3.0*i+4.0;
//        }
//        err[i] = 0.01;
//    }
//    
//    vvd_t pars=fit_par_jackknife(coord,2,err,y,0,4);

    
    cout<<"DONE!"<<endl;
    
    return 0;
}
