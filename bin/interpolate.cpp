#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"
#include "allocate.hpp"
#include "interpolate.hpp"
#include "ave_err.hpp"
#include "fit.hpp"


//! returns the range in which x is contained
pair<double,double> oper_t::get_a2p2_range(const int size,const double a2p2_ref,const size_t N) const
{
    // select the neighbours
    vector<pair<double,double>> dist_list;
    vector<double> a2p2_prox(N);
    
    for(int imom=0;imom<size;imom++)
    {
        double dist=fabs(p2[imom]-a2p2_ref);
        dist_list.push_back(make_pair(dist,p2[imom]));
    }
    sort(dist_list.begin(), dist_list.end());
    
    for(size_t i=0;i<N;i++)
        a2p2_prox[i] = dist_list[i].second;
    sort(a2p2_prox.begin(),a2p2_prox.end());
    
    double a2p2min = a2p2_prox[0];
    double a2p2max = a2p2_prox[N-1];
    
    return {a2p2min,a2p2max};
}

//! returns the range in which x is contained
pair<double,double> oper_t::get_a2p2tilde_range(const int size,const double a2p2_ref,const size_t N) const
{
    // select the neighbours
    vector<pair<double,double>> dist_list;
    vector<double> a2p2_prox(N);
    
    for(int imom=0;imom<size;imom++)
    {
        double dist=fabs(p2_tilde[imom]-a2p2_ref);
        dist_list.push_back(make_pair(dist,p2_tilde[imom]));
    }
    sort(dist_list.begin(), dist_list.end());
    
    for(size_t i=0;i<N;i++)
        a2p2_prox[i] = dist_list[i].second;
    sort(a2p2_prox.begin(),a2p2_prox.end());
    
    double a2p2min = a2p2_prox[0];
    double a2p2max = a2p2_prox[N-1];
    
    return {a2p2min,a2p2max};
}


oper_t oper_t::interpolate_to_ainv(int b)
{
    double ainv2 = ainv[b]*ainv[b];
    double a2p2ref = 1.0; // p2ref in lattice units //this is p2 (NOT p2tilde)
    
    cout<<endl;
    cout<<"----- interpolation to p2 = 1/a2 = "<<ainv2<<" Gev^2 -----"<<endl<<endl;
    
    oper_t out=(*this);
    
    out.linmoms=vector<array<int,1>>{{0}};
    out.bilmoms=vector<array<int,3>>{{0,0,0}};
    out.meslepmoms=out.bilmoms;
    
    out._linmoms=1;
    out._bilmoms=1;
    out._meslepmoms=1;
    
    out.allocate();
    
    pair<double,double> a2p2minmax=(*this).get_a2p2_range(_linmoms,a2p2ref);
    double a2p2_min=a2p2minmax.first;
    double a2p2_max=a2p2minmax.second;
    cout<<"a2p2 range (lattice units):   "<<a2p2_min<<" - "<<a2p2_max<<endl;
    
    int npar=3;
    vvd_t coord(vd_t(0.0,_linmoms),npar);
    for(int j=0; j<_linmoms; j++)
    {
        // parabolic fit in lattice units
        coord[0][j] = 1.0;
        coord[1][j] = p2[j];  //momentum in lattice units
        coord[2][j] = coord[1][j]*coord[1][j];
    }
    
    // Interpolating Zq
    vvd_t y_Zq(vd_t(0.0,_linmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zq(0.0,_linmoms);                   // [moms]
    vvd_t dy_Zq_tmp = get<1>(ave_err_Zq((*this).jZq)); // [moms][nmr]
    
    vvd_t y_Zq_EM(vd_t(0.0,_linmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zq_EM(0.0,_linmoms);                   // [moms]
    vvd_t dy_Zq_EM_tmp = get<1>(ave_err_Zq((*this).jZq_EM)); // [moms][nmr]
    
    for(int imom=0;imom<_linmoms;imom++)
    {
        for(int ijack=0;ijack<njacks;ijack++)
        {
            y_Zq[ijack][imom] = jZq[imom][ijack][0];
            y_Zq_EM[ijack][imom] = jZq_EM[imom][ijack][0];
        }
        
        dy_Zq[imom] = dy_Zq_tmp[imom][0];
        dy_Zq_EM[imom] = dy_Zq_EM_tmp[imom][0];
    }
    
    vvd_t jZq_pars = polyfit(coord,npar,dy_Zq,y_Zq,a2p2_min,a2p2_max); // [ijack][ipar]
    vvd_t jZq_EM_pars = polyfit(coord,npar,dy_Zq_EM,y_Zq_EM,a2p2_min,a2p2_max); // [ijack][ipar]
    
    for(int ijack=0;ijack<njacks;ijack++)
    {
        (out.jZq)[0][ijack][0] = jZq_pars[ijack][0] +
                                 jZq_pars[ijack][1]*a2p2ref +
                                 jZq_pars[ijack][2]*a2p2ref*a2p2ref;
        
        (out.jZq_EM)[0][ijack][0] = jZq_EM_pars[ijack][0] +
                                    jZq_EM_pars[ijack][1]*a2p2ref +
                                    jZq_EM_pars[ijack][2]*a2p2ref*a2p2ref;
    }

    
    // Interpolating Zbil
    vvd_t y_Zbil(vd_t(0.0,_bilmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zbil(0.0,_bilmoms);                   // [moms]
    vvvvd_t dy_Zbil_tmp = get<1>(ave_err_Z((*this).jZ)); // [moms][nbil][nmr][nmr]
    
    vvd_t y_Zbil_EM(vd_t(0.0,_bilmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zbil_EM(0.0,_bilmoms);                   // [moms]
    vvvvd_t dy_Zbil_EM_tmp = get<1>(ave_err_Z((*this).jZ_EM)); // [moms][nbil][nmr][nmr]

    for(int ibil=0;ibil<nbil;ibil++)
    {
        
        for(int imom=0;imom<_bilmoms;imom++)
        {
            for(int ijack=0;ijack<njacks;ijack++)
            {
                y_Zbil[ijack][imom] = jZ[imom][ibil][ijack][0][0];
                y_Zbil_EM[ijack][imom] = jZ_EM[imom][ibil][ijack][0][0];
            }
            
            dy_Zbil[imom] = dy_Zbil_tmp[imom][ibil][0][0];
            dy_Zbil_EM[imom] = dy_Zbil_EM_tmp[imom][ibil][0][0];
        }
        
        vvd_t jZ_pars = polyfit(coord,npar,dy_Zbil,y_Zbil,a2p2_min,a2p2_max); // [ijack][ipar]
        vvd_t jZ_EM_pars = polyfit(coord,npar,dy_Zbil_EM,y_Zbil_EM,a2p2_min,a2p2_max); // [ijack][ipar]
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            (out.jZ)[0][ibil][ijack][0][0] = jZ_pars[ijack][0] +
                                             jZ_pars[ijack][1]*a2p2ref +
                                             jZ_pars[ijack][2]*a2p2ref*a2p2ref;
            
            (out.jZ_EM)[0][ibil][ijack][0][0] = jZ_EM_pars[ijack][0] +
                                                jZ_EM_pars[ijack][1]*a2p2ref +
                                                jZ_EM_pars[ijack][2]*a2p2ref*a2p2ref;
        }
        
    }
    
    // Interpolating Z4f
    vvd_t y_Z4f(vd_t(0.0,_meslepmoms),njacks);       // [njacks][moms]
    vd_t  dy_Z4f(0.0,_meslepmoms);                   // [moms]
    vvvvvd_t dy_Z4f_tmp = get<1>(ave_err_Z4f((*this).jZ_4f)); // [moms][nbil][nbil][nmr][nmr]
    
    vvd_t y_Z4f_EM(vd_t(0.0,_meslepmoms),njacks);       // [njacks][moms]
    vd_t  dy_Z4f_EM(0.0,_meslepmoms);                   // [moms]
    vvvvvd_t dy_Z4f_EM_tmp = get<1>(ave_err_Z4f((*this).jZ_4f_EM)); // [moms][nbil][nbil][nmr][nmr]
    
    for(int iop1=0;iop1<nbil;iop1++)
        for(int iop2=0;iop2<nbil;iop2++)
        {
            for(int imom=0;imom<_meslepmoms;imom++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    y_Z4f[ijack][imom] = jZ_4f[imom][iop1][iop2][ijack][0][0];
                    y_Z4f_EM[ijack][imom] = jZ_4f_EM[imom][iop1][iop2][ijack][0][0];
                }
                
                dy_Z4f[imom] = dy_Z4f_tmp[imom][iop1][iop2][0][0];
                dy_Z4f_EM[imom] = dy_Z4f_EM_tmp[imom][iop1][iop2][0][0];
            }
            
            vvd_t jZ4f_pars = polyfit(coord,npar,dy_Z4f,y_Z4f,a2p2_min,a2p2_max); // [ijack][ipar]
            vvd_t jZ4f_EM_pars = polyfit(coord,npar,dy_Z4f_EM,y_Z4f_EM,a2p2_min,a2p2_max); // [ijack][ipar]
            
            for(int ijack=0;ijack<njacks;ijack++)
            {
                (out.jZ_4f)[0][iop1][iop2][ijack][0][0] = jZ4f_pars[ijack][0] +
                                                          jZ4f_pars[ijack][1]*a2p2ref +
                                                          jZ4f_pars[ijack][2]*a2p2ref*a2p2ref;
                
                (out.jZ_4f_EM)[0][iop1][iop2][ijack][0][0] = jZ4f_EM_pars[ijack][0] +
                                                             jZ4f_EM_pars[ijack][1]*a2p2ref +
                                                             jZ4f_EM_pars[ijack][2]*a2p2ref*a2p2ref;
                
            }
        }

    
    return out;
}

oper_t oper_t::interpolate_to_p2ref(int b)
{
    cout<<endl;
    cout<<"----- interpolation to p2 = "<<p2ref<<" Gev^2 -----"<<endl<<endl;
    
    double ainv2 = ainv[b]*ainv[b];
    double a2p2ref = p2ref/ainv2; // p2ref in lattice units //this is p2tilde
    
    oper_t out=(*this);
    
    out.linmoms=vector<array<int,1>>{{0}};
    out.bilmoms=vector<array<int,3>>{{0,0,0}};
    out.meslepmoms=out.bilmoms;
    
    out._linmoms=1;
    out._bilmoms=1;
    out._meslepmoms=1;
    
    out.allocate();
    
    pair<double,double> a2p2minmax=(*this).get_a2p2tilde_range(_linmoms,a2p2ref);
    double p2_min=a2p2minmax.first*ainv2;
    double p2_max=a2p2minmax.second*ainv2;
    cout<<"p2tilde range (physical units):   "<<p2_min<<" - "<<p2_max<<endl;
    
    int npar=3;
    vvd_t coord(vd_t(0.0,_linmoms),npar);
    for(int j=0; j<_linmoms; j++)
    {
        // parabolic fit in physical units
        coord[0][j] = 1.0;
        coord[1][j] = p2_tilde[j]*ainv2;
        coord[2][j] = coord[1][j]*coord[1][j];
    }
    
    // Interpolating Zq
    vvd_t y_Zq(vd_t(0.0,_linmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zq(0.0,_linmoms);                   // [moms]
    vvd_t dy_Zq_tmp = get<1>(ave_err_Zq((*this).jZq)); // [moms][nmr]
    
    vvd_t y_Zq_EM(vd_t(0.0,_linmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zq_EM(0.0,_linmoms);                   // [moms]
    vvd_t dy_Zq_EM_tmp = get<1>(ave_err_Zq((*this).jZq_EM)); // [moms][nmr]
    
    for(int imom=0;imom<_linmoms;imom++)
    {
        for(int ijack=0;ijack<njacks;ijack++)
        {
            y_Zq[ijack][imom] = jZq[imom][ijack][0];
            y_Zq_EM[ijack][imom] = jZq_EM[imom][ijack][0];
        }
        
        dy_Zq[imom] = dy_Zq_tmp[imom][0];
        dy_Zq_EM[imom] = dy_Zq_EM_tmp[imom][0];
    }
    
    vvd_t jZq_pars = polyfit(coord,npar,dy_Zq,y_Zq,p2_min,p2_max); // [ijack][ipar]
    vvd_t jZq_EM_pars = polyfit(coord,npar,dy_Zq_EM,y_Zq_EM,p2_min,p2_max); // [ijack][ipar]
    
    for(int ijack=0;ijack<njacks;ijack++)
    {
        (out.jZq)[0][ijack][0] = jZq_pars[ijack][0] +
        jZq_pars[ijack][1]*p2ref +
        jZq_pars[ijack][2]*p2ref*p2ref;
        
        (out.jZq_EM)[0][ijack][0] = jZq_EM_pars[ijack][0] +
        jZq_EM_pars[ijack][1]*p2ref +
        jZq_EM_pars[ijack][2]*p2ref*p2ref;
    }
    
    
    // Interpolating Zbil
    vvd_t y_Zbil(vd_t(0.0,_bilmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zbil(0.0,_bilmoms);                   // [moms]
    vvvvd_t dy_Zbil_tmp = get<1>(ave_err_Z((*this).jZ)); // [moms][nbil][nmr][nmr]
    
    vvd_t y_Zbil_EM(vd_t(0.0,_bilmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zbil_EM(0.0,_bilmoms);                   // [moms]
    vvvvd_t dy_Zbil_EM_tmp = get<1>(ave_err_Z((*this).jZ_EM)); // [moms][nbil][nmr][nmr]
    
    for(int ibil=0;ibil<nbil;ibil++)
    {
        
        for(int imom=0;imom<_bilmoms;imom++)
        {
            for(int ijack=0;ijack<njacks;ijack++)
            {
                y_Zbil[ijack][imom] = jZ[imom][ibil][ijack][0][0];
                y_Zbil_EM[ijack][imom] = jZ_EM[imom][ibil][ijack][0][0];
            }
            
            dy_Zbil[imom] = dy_Zbil_tmp[imom][ibil][0][0];
            dy_Zbil_EM[imom] = dy_Zbil_EM_tmp[imom][ibil][0][0];
        }
        
        vvd_t jZ_pars = polyfit(coord,npar,dy_Zbil,y_Zbil,p2_min,p2_max); // [ijack][ipar]
        vvd_t jZ_EM_pars = polyfit(coord,npar,dy_Zbil_EM,y_Zbil_EM,p2_min,p2_max); // [ijack][ipar]
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            (out.jZ)[0][ibil][ijack][0][0] = jZ_pars[ijack][0] +
            jZ_pars[ijack][1]*p2ref +
            jZ_pars[ijack][2]*p2ref*p2ref;
            
            (out.jZ_EM)[0][ibil][ijack][0][0] = jZ_EM_pars[ijack][0] +
            jZ_EM_pars[ijack][1]*p2ref +
            jZ_EM_pars[ijack][2]*p2ref*p2ref;
        }
    }
    
    // Interpolating ZV/ZA and ZP/ZS
    vvd_t y_ZVovZA(vd_t(0.0,_bilmoms),njacks);       // [njacks][moms]
    vd_t  dy_ZVovZA(0.0,_bilmoms);                   // [moms]
    vvvvd_t dy_ZVovZA_tmp = get<1>(ave_err_Z((*this).jZVoverZA)); // [moms][nbil][nmr][nmr]
    
    vvd_t y_ZPovZS(vd_t(0.0,_bilmoms),njacks);       // [njacks][moms]
    vd_t  dy_ZPovZS(0.0,_bilmoms);                   // [moms]
    vvvvd_t dy_ZPovZS_tmp = get<1>(ave_err_Z((*this).jZPoverZS)); // [moms][nbil][nmr][nmr]
    
    for(int imom=0;imom<_bilmoms;imom++)
    {
        for(int ijack=0;ijack<njacks;ijack++)
        {
            y_ZVovZA[ijack][imom] = jZVoverZA[imom][0][ijack][0][0];
            y_ZPovZS[ijack][imom] = jZPoverZS[imom][0][ijack][0][0];
        }
        
        dy_ZVovZA[imom] = dy_ZVovZA_tmp[imom][0][0][0];
        dy_ZPovZS[imom] = dy_ZPovZS_tmp[imom][0][0][0];
    }
    
    vvd_t jZVovZA_pars = polyfit(coord,npar,dy_ZVovZA,y_ZVovZA,p2_min,p2_max); // [ijack][ipar]
    vvd_t jZPovZS_pars = polyfit(coord,npar,dy_ZPovZS,y_ZPovZS,p2_min,p2_max); // [ijack][ipar]
    
    for(int ijack=0;ijack<njacks;ijack++)
    {
        (out.jZVoverZA)[0][0][ijack][0][0] = jZVovZA_pars[ijack][0] +
        jZVovZA_pars[ijack][1]*p2ref +
        jZVovZA_pars[ijack][2]*p2ref*p2ref;
        
        (out.jZPoverZS)[0][0][ijack][0][0] = jZPovZS_pars[ijack][0] +
        jZPovZS_pars[ijack][1]*p2ref +
        jZPovZS_pars[ijack][2]*p2ref*p2ref;
    }
    
    // Interpolating Z4f
    vvd_t y_Z4f(vd_t(0.0,_meslepmoms),njacks);       // [njacks][moms]
    vd_t  dy_Z4f(0.0,_meslepmoms);                   // [moms]
    vvvvvd_t dy_Z4f_tmp = get<1>(ave_err_Z4f((*this).jZ_4f)); // [moms][nbil][nbil][nmr][nmr]
    
    vvd_t y_Z4f_EM(vd_t(0.0,_meslepmoms),njacks);       // [njacks][moms]
    vd_t  dy_Z4f_EM(0.0,_meslepmoms);                   // [moms]
    vvvvvd_t dy_Z4f_EM_tmp = get<1>(ave_err_Z4f((*this).jZ_4f_EM)); // [moms][nbil][nbil][nmr][nmr]
    
    for(int iop1=0;iop1<nbil;iop1++)
        for(int iop2=0;iop2<nbil;iop2++)
        {
            for(int imom=0;imom<_meslepmoms;imom++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    y_Z4f[ijack][imom] = jZ_4f[imom][iop1][iop2][ijack][0][0];
                    y_Z4f_EM[ijack][imom] = jZ_4f_EM[imom][iop1][iop2][ijack][0][0];
                }
                
                dy_Z4f[imom] = dy_Z4f_tmp[imom][iop1][iop2][0][0];
                dy_Z4f_EM[imom] = dy_Z4f_EM_tmp[imom][iop1][iop2][0][0];
            }
            
            vvd_t jZ4f_pars = polyfit(coord,npar,dy_Z4f,y_Z4f,p2_min,p2_max); // [ijack][ipar]
            vvd_t jZ4f_EM_pars = polyfit(coord,npar,dy_Z4f_EM,y_Z4f_EM,p2_min,p2_max); // [ijack][ipar]
            
            for(int ijack=0;ijack<njacks;ijack++)
            {
                (out.jZ_4f)[0][iop1][iop2][ijack][0][0] = jZ4f_pars[ijack][0] +
                jZ4f_pars[ijack][1]*p2ref +
                jZ4f_pars[ijack][2]*p2ref*p2ref;
                
                (out.jZ_4f_EM)[0][iop1][iop2][ijack][0][0] = jZ4f_EM_pars[ijack][0] +
                jZ4f_EM_pars[ijack][1]*p2ref +
                jZ4f_EM_pars[ijack][2]*p2ref*p2ref;
                
            }
        }
    
    
    return out;
}

