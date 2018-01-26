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

//debug
#include "subtraction.hpp"

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
vvvd_t eff_mass;

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
    
//    //debug
//    for(int imom=0; imom<moms;imom++)
//    {
//        cout<<imom<<"____"<<endl;
//        cout<<"LO"<<endl;
//        for(int ibil=0;ibil<nbil;ibil++)
//            cout<<subtraction(imom,ibil,LO)<<endl;
//        cout<<"EM"<<endl;
//        for(int ibil=0;ibil<nbil;ibil++)
//            cout<<subtraction(imom,ibil,EM)<<endl;;
//    }

    
//    basic.step="basic";
    
    /*  basic  */
    
    basic.create_basic();
    basic.plot("");
    oper_t ave = basic.average_equiv_moms();
    ave.plot("ave");
    
    /*  average r  */

    oper_t rave = basic.average_r();
    rave.plot("rave");
    oper_t rave_ave = rave.average_equiv_moms();  ///
    rave_ave.plot("rave_ave");
    
    /*  chiral extr  */

    oper_t rave_chir = rave.chiral_extr();
    rave_chir.plot("rave_chir");
    oper_t rave_chir_ave = rave_chir.average_equiv_moms();  ///
    rave_chir_ave.plot("rave_chir_ave");
    

    /*  O(a2g2) subtraction  */

    oper_t rave_chir_sub = rave_chir.subtract();
    rave_chir_sub.plot("rave_chir_sub");
    oper_t rave_chir_sub_ave = rave_chir_sub.average_equiv_moms();  ///
    rave_chir_sub_ave.plot("rave_chir_sub_ave");

    /*  evolution to 1/a  */
    
    oper_t rave_chir_sub_evo = rave_chir_sub.evolve();
    rave_chir_sub_evo.plot("rave_chir_sub_evo");
    oper_t rave_chir_sub_evo_ave = rave_chir_sub_evo.average_equiv_moms();
    rave_chir_sub_evo_ave.plot("rave_chir_sub_evo_ave");

    /*  a2p2->0 extrapolation  */
    
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
