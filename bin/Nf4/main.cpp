#ifdef HAVE_CONFIG_H
#include <config.hpp>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
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

ORDER ord;

// mom lists
vector<coords_t> mom_list;
vector<p_t> p, p_tilde;
vector<double> p2, p2_tilde;
vector<double> p4, p4_tilde;

// list of N(p)
vector<int> Np;

//read effective mass
vector< vector<double> > eff_mass;

//using namespace std;

int main()
{
#pragma omp parallel
#pragma omp master
    system("clear");
    cout<<"Using "<<omp_get_num_threads()<<" threads"<<endl;
    
    char path[128]="input.txt";
    read_input(path);
    
    //read mom list
    read_mom_list(mom_path);
    moms=mom_list.size();
    cout<<"Read: "<<mom_list.size()<<" momenta."<<endl<<endl;
    
    eff_mass=read_eff_mass("eff_mass_array",nmr);
    
    cout<<LO<<"  "<<EM<<"  "<<endl;

    
    oper_t basic;
    
    basic.step="basic";
    basic.create_basic();
    
    oper_t rave = basic.average_r();
    
    oper_t chir = rave.chiral_extr();
    
   // oper_t sub = chir.subtract();
    
   // oper_t evo = sub.evolve();
    
//    for(int m=0;m<neq2;m++)
//        cout<<m<<"  basic: "<<basic.m_eff_equivalent_Zq[m]<<endl;
//    for(int m=0;m<neq2;m++)
//        cout<<m<<"  rave: "<<rave.m_eff_equivalent_Zq[m]<<endl;
//    
//    cout<<"basic ave: "<<((basic.jG_0)[0][0][0][0][0]+(basic.jG_0)[0][0][1][1][0])/2.0<<endl;
//    cout<<"rave: "<<(rave.jG_0_ave_r)[0][0][0][0]<<endl;
    
    
//    cout<<get<STEP>(basic.Zq[0])<<endl;
    
    cout<<"DONE!"<<endl;
    
    return 0;
}
