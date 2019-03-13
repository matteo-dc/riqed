#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"
#include "allocate.hpp"
#include "interpolate.hpp"
#include "ave_err.hpp"
#include "fit.hpp"

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

//vector<jZq_t> oper_t::interpolate_to_p2ref_Zq(const double a2p2_ref,const int LO_or_EM)
//{
//    vector<jZq_t> out;
//    int moms=(*this)._linmoms;
//    
//    allocate_vec(out,{1,njacks,(*this)._nmr});
//    
//    pair<double,double> a2p2minmax=(*this).get_a2p2tilde_range(moms,a2p2);
//    p2min=a2p2minmax.first*ainv2;
//    p2max=a2p2minmax.second*ainv2;
//    cout<<"p2 range:   "<<p2min<<" - "<<p2max<<endl;
//    
//    int npar=3;
//    vvd_t coord(vd_t(0.0,_linmoms),npar);
//    for(int j=0; j<_linmoms; j++)
//    {
//        // parabolic fit in lattice units
//        coord[0][j] = 1.0;
//        coord[1][j] = p2_tilde[j];
//        coord[2][j] = coord[1][j]*coord[1][j];
//    }
//
//    
//    
//    return out;
//}

oper_t oper_t::interpolate_to_p2ref(int b)
{
    cout<<endl;
    cout<<"----- interpolation to p2 = "<<p2ref<<" Gev^2 -----"<<endl<<endl;
    
    double ainv2 = ainv[b]*ainv[b];
    double a2p2ref = p2ref/ainv2; // p2ref in lattice units
//    double p2min,p2max;
    
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
    cout<<"p2 range (physical units):   "<<p2_min<<" - "<<p2_max<<endl;
    
    int npar=3;
    vvd_t coord(vd_t(0.0,_linmoms),npar);
    for(int j=0; j<_linmoms; j++)
    {
        // parabolic fit in lattice units
        coord[0][j] = 1.0;
        coord[1][j] = p2_tilde[j]*ainv2;
        coord[2][j] = coord[1][j]*coord[1][j];
    }
    
    // Interpolating Zq
    vvd_t y_Zq(vd_t(0.0,_linmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zq(0.0,_linmoms);                   // [moms]
    vvd_t dy_Zq_tmp = get<1>(ave_err_Zq((*this).jZq_EM)); // [moms][nmr]
    
    for(int imom=0;imom<_linmoms;imom++)
    {
        for(int ijack=0;ijack<njacks;ijack++)
            y_Zq[ijack][imom] = jZq_EM[imom][ijack][0];
        dy_Zq[imom] = dy_Zq_tmp[imom][0];
    }
    
    vvd_t jZq_pars = polyfit(coord,npar,dy_Zq,y_Zq,p2_min,p2_max); // [ijack][ipar]
    
    for(int ijack=0;ijack<njacks;ijack++)
        (out.jZq_EM)[0][ijack][0] = jZq_pars[ijack][0] +
                                    jZq_pars[ijack][1]*p2ref +
                                    jZq_pars[ijack][2]*p2ref*p2ref;

    
    // Interpolating Zbil
    vvd_t y_Zbil(vd_t(0.0,_bilmoms),njacks);       // [njacks][moms]
    vd_t  dy_Zbil(0.0,_bilmoms);                   // [moms]
    vvvvd_t dy_Zbil_tmp = get<1>(ave_err_Z((*this).jZ_EM)); // [moms][nbil][nmr][nmr]
    
    for(int ibil=0;ibil<nbil;ibil++)
    {
        
        for(int imom=0;imom<_bilmoms;imom++)
        {
            for(int ijack=0;ijack<njacks;ijack++)
                y_Zbil[ijack][imom] = jZ_EM[imom][ibil][ijack][0][0];
            dy_Zbil[imom] = dy_Zbil_tmp[imom][ibil][0][0];
        }
        
        vvd_t jZ_pars = polyfit(coord,npar,dy_Zbil,y_Zbil,p2_min,p2_max); // [ijack][ipar]
        
        for(int ijack=0;ijack<njacks;ijack++)
            (out.jZ_EM)[0][ibil][ijack][0][0] = jZ_pars[ijack][0] +
                                                jZ_pars[ijack][1]*p2ref +
                                                jZ_pars[ijack][2]*p2ref*p2ref;
            
    }
    
    // Interpolating Z4f
    vvd_t y_Z4f(vd_t(0.0,_meslepmoms),njacks);       // [njacks][moms]
    vd_t  dy_Z4f(0.0,_meslepmoms);                   // [moms]
    vvvvvd_t dy_Z4f_tmp = get<1>(ave_err_Z4f((*this).jZ_4f_EM)); // [moms][nbil][nbil][nmr][nmr]
    
    for(int iop1=0;iop1<nbil;iop1++)
        for(int iop2=0;iop2<nbil;iop2++)
        {
            for(int imom=0;imom<_meslepmoms;imom++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                    y_Z4f[ijack][imom] = jZ_4f_EM[imom][iop1][iop2][ijack][0][0];
                dy_Z4f[imom] = dy_Z4f_tmp[imom][iop1][iop2][0][0];
            }
            
            vvd_t jZ4f_pars = polyfit(coord,npar,dy_Z4f,y_Z4f,p2_min,p2_max); // [ijack][ipar]
            
            for(int ijack=0;ijack<njacks;ijack++)
                (out.jZ_4f_EM)[0][iop1][iop2][ijack][0][0] = jZ4f_pars[ijack][0] +
                                                             jZ4f_pars[ijack][1]*p2ref +
                                                             jZ4f_pars[ijack][2]*p2ref*p2ref;
        }

    
    //////////////
    
    vvvvd_t Z_err = get<1>(ave_err_Z(jZ));
    vvvvvd_t Z4f_err=get<1>(ave_err(jZ_4f));
    
    return out;
}

