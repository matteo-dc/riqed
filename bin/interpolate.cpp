#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"
#include "allocate.hpp"
#include "interpolate.hpp"
#include "ave_err.hpp"

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

vector<jZq_t> oper_t::interpolate_to_p2ref_Zq(const double a2p2_ref,const int LO_or_EM)
{
    vector<jZq_t> out;
    int moms=(*this)._linmoms;
    
    allocate_vec(out,{1,njacks,(*this)._nmr});
    
    pair<double,double> a2p2minmax=(*this).get_a2p2tilde_range(moms,a2p2);
    p2min=a2p2minmax.first*ainv2;
    p2max=a2p2minmax.second*ainv2;
    cout<<"p2 range:   "<<p2min<<" - "<<p2max<<endl;
    
    int npar=3;
    vvd_t coord(vd_t(0.0,_linmoms),npar);
    for(int j=0; j<_linmoms; j++)
    {
        // parabolic fit in lattice units
        coord[0][j] = 1.0;
        coord[1][j] = p2_tilde[j];
        coord[2][j] = coord[1][j]*coord[1][j];
    }

    
    return out;
}

oper_t oper_t::interpolate_to_p2ref(int b)
{
    cout<<endl;
    cout<<"----- interpolation to p2 = "<<p2ref<<" Gev^2 -----"<<endl<<endl;
    
    double ainv2 = ainv[b]*ainv[b];
    double a2p2 = p2ref/ainv2; // p2ref in lattice units
    double p2min,p2max;
    
    oper_t out=(*this);
    
    out.linmoms=vector<array<int,1>>{{0}};
    out.bilmoms=vector<array<int,3>>{{0,0,0}};
    out.meslepmoms=out.bilmoms;
    
    out._linmoms=1;
    out._bilmoms=1;
    out._meslepmoms=1;
    
    out.allocate();
    
    // Interpolating Zq
    out.jZq=(*this).interpolate_to_p2ref_Zq(a2p2,LO);
    out.jZq_EM=(*this).interpolate_to_p2ref_Zq(a2p2,EM);
    
    
    //////////////
    
    vvvvd_t Z_err = get<1>(ave_err_Z(jZ));
    vvvvvd_t Z4f_err=get<1>(ave_err(jZ_4f));
    
    return out;
}

