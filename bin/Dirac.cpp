#include "read.hpp"
#include "aliases.hpp"
#include <math.h>
#include <vector>

//create Dirac Gamma matrices
vprop_t make_gamma()
{
    int NGamma=16;
    
    vprop_t gam(NGamma);
    vector< vector<int> > col(NGamma, vector<int>(4));
    vector< vector<int> > real_part(NGamma, vector<int>(4));
    vector< vector<int> > im_part(NGamma, vector<int>(4));
    
    //Identity=gamma0
    col[0]={0,1,2,3};
    real_part[0]={1,1,1,1};
    im_part[0]={0,0,0,0};
    //gamma1
    col[1]={3,2,1,0};
    real_part[1]={0,0,0,0};
    im_part[1]={-1,-1,1,1};
    //gamma2
    col[2]={3,2,1,0};
    real_part[2]={-1,1,1,-1};
    im_part[2]={0,0,0,0};
    //gamma3
    col[3]={2,3,0,1};
    real_part[3]={0,0,0,0};
    im_part[3]={-1,1,1,-1};
    //gamma4
    col[4]={2,3,0,1};
    real_part[4]={-1,-1,-1,-1};
    im_part[4]={0,0,0,0};
    //gamma5 = gamma4*gamma1*gamma2*gamma3
    col[5]={0,1,2,3};
    real_part[5]={1,1,-1,-1};
    im_part[5]={0,0,0,0};
    
    for(int i_gam=0;i_gam<6;i_gam++)
        for(int i_row=0;i_row<4;i_row++)
            for(int ic=0;ic<3;ic++)
            {
                gam[i_gam](isc(i_row,ic),isc(col[i_gam][i_row],ic))=dcompl(real_part[i_gam][i_row],im_part[i_gam][i_row] );
            }
    
    //gamma_mu*gamma5
    for(int j=0;j<4;j++)
    {
        gam[6+j]=gam[1+j]*gam[5];
    }
    //sigma
    size_t ind1[6]={4,4,4,2,3,1};
    size_t ind2[6]={1,2,3,3,1,2};
    for(int i=0;i<6;i++)
        gam[10+i]=0.5*(gam[ind1[i]]*gam[ind2[i]]-gam[ind2[i]]*gam[ind1[i]]);
    
    return gam;
}

vprop_t make_gamma_4f()
{
    int NGamma=16;
    int NGamma4f=16+6;
    
    vprop_t gam=make_gamma();
    vprop_t gam_4f(NGamma4f);
    
    for(int i=0;i<NGamma;i++) gam_4f[i]=gam[i];
    for(int i=NGamma;i<NGamma4f;i++) gam_4f[i]=gam[i-6]*gam[5];
    
    return gam_4f;
}


vprop_t GAMMA=make_gamma();

vprop_t GAMMA_4f=make_gamma_4f();

//create projectors such that tr(GAMMA*P)=Identity
vprop_t create_projectors()
{
    vprop_t P(prop_t::Zero(),16);
    vector<double> NL={1.0,4.0,4.0,4.0,4.0,1.0,4.0,4.0,4.0,4.0,6.0,6.0,6.0,6.0,6.0,6.0}; // normalize sum over Lorentz indices
    
    for(int igam=0;igam<16;igam++)
        P[igam]=GAMMA[igam].adjoint()/NL[igam];
    
    return P;
}

vprop_t create_projectors_4f()
{
    vprop_t P(prop_t::Zero(),22);
    vector<double> NL={1.0,4.0,4.0,4.0,4.0,1.0,4.0,4.0,4.0,4.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0}; // normalize sum over Lorentz indices
    
    for(int igam=0;igam<22;igam++)
        P[igam]=GAMMA_4f[igam].adjoint()/NL[igam];
    
    return P;
}

vprop_t Proj=create_projectors();
vprop_t Proj_4f=create_projectors_4f();
