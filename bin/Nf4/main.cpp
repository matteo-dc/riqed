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
vvd_t eff_mass;

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
    if(UseEffMass) eff_mass=read_eff_mass("eff_mass_array");
    
    basic.step="basic";
    basic.create_basic();
    
    oper_t rave = basic.average_r();
    
    oper_t rave_chir = rave.chiral_extr();
    
    oper_t rave_chir_sub = rave_chir.subtract();

    oper_t rave_chir_sub_evo = rave_chir_sub.evolve();

    oper_t rave_chir_sub_evo_ave = rave_chir_sub_evo.average_equiv_moms();

    cout<<"Continuum limit extrapolation:"<<endl<<endl;
    continuum_limit(rave_chir_sub_evo_ave,LO);
    continuum_limit(rave_chir_sub_evo_ave,EM);
//
//    
//    vector<vd_t> a=rave_chir_sub_evo_ave.jZq_evo;
//    
//    PRINT(a);
   

    
    cout<<"DONE!"<<endl;
    
    return 0;
}
