#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"
#include "vertices.hpp"
#include "meslep.hpp"
#include "Zq.hpp"
#include "allocate.hpp"

#include "sigmas.hpp"

//read file
void allocate_vec_internal(double &t, const vector<int> sizes, int isize)
{
    t=0.0;
}

void allocate_vec_internal(oper_t &o, const vector<int> sizes, int isize)
{}

void oper_t::clear_all()
{
    sigma.clear();
    sigma.shrink_to_fit();
    
    jG.clear();
    jG.shrink_to_fit();

    jpr_meslep.clear();
    jpr_meslep.shrink_to_fit();

    jZq.clear();
    jZq.shrink_to_fit();
    jZq_EM.clear();
    jZq_EM.shrink_to_fit();

    jZ.clear();
    jZ.shrink_to_fit();
    jZ_EM.clear();
    jZ_EM.shrink_to_fit();

    jZ_4f.clear();
    jZ_4f.shrink_to_fit();
    jZ_4f_EM.clear();
    jZ_4f_EM.shrink_to_fit();
}

void oper_t::allocate_val()
{
    // allocate effective masses
    allocate_vec(eff_mass,{njacks,_nm,_nm});
    allocate_vec(eff_mass_sea,{njacks});
    
    // allocate deltam
    allocate_vec(deltam_cr,{njacks,_nmr});
    allocate_vec(deltamu,{njacks,_nmr});
}

void oper_t::allocate()
{
    clear_all();
    
    // allocate sigmas
    allocate_vec(sigma,{_linmoms,sigma::nproj,sigma::nins,njacks,_nmr});
    // allocate pr_bil
    allocate_vec(jG,{_bilmoms,gbil::nins,nbil,njacks,_nmr,_nmr});
    // allocate pr_meslep
    allocate_vec(jpr_meslep,{_meslepmoms,pr_meslep::nins,nbil,nbil,njacks,_nmr,_nmr});
    
    // allocate Zq
    allocate_vec(jZq,{_linmoms,njacks,_nmr});
    allocate_vec(jZq_EM,{_linmoms,njacks,_nmr});
    // allocate Zbil
    allocate_vec(jZ,{_bilmoms,nbil,njacks,_nmr,_nmr});
    allocate_vec(jZ_EM,{_bilmoms,nbil,njacks,_nmr,_nmr});
    // allocate Z4f
    allocate_vec(jZ_4f,{_meslepmoms,nbil,nbil,njacks,_nmr,_nmr});
    allocate_vec(jZ_4f_EM,{_meslepmoms,nbil,nbil,njacks,_nmr,_nmr});
}
