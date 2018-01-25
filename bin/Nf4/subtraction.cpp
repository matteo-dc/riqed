#include "aliases.hpp"
#include "global.hpp"


vvd_t pr_bil_Oa2(const int LO_or_EM)
{
    vd_t c_v(3), c_a(3), c_s(3), c_p(3), c_t(3);
    
    // Coefficients from the Mathematica file 'O(g2a2).nb'
    
    // We divide the following coefficients by 4 (or 6 for T)
    // to account for the sum on Lorentz indices!
    
    // The coefficients c_v and c_a are used respectively on Gv and Ga,
    // BUT in the physical basis ZA=Zq/Gv and ZV=Zq/Ga!!!
    
    if(action=="Iwa")
    {
        if(LO_or_EM==LO)
        {
            c_a={0.2881372/4.,-0.2095/4.,-0.516583/4.};     // Iwasaki Action with Landau gauge
            c_v={0.9637998/4.,-0.2095/4.,-0.516583/4.};
            c_s={2.02123300,-1./4.,0.376167};
            c_p={0.66990790,-1./4.,0.376167};
            c_t={0.3861012/6.,-0.196/6.,-0.814167/6.};
        }
        else if(LO_or_EM==EM)
        {
            c_v={0.3997992/4.,1./16./4.,-1./4./4.};         // Wilson Action with Feynman gauge
            c_a={0.2394365/4.,-3./16./4.,-1./4./4.};
            c_s={0.32682365,1./2.,5./12.};
            c_p={0.00609817,0.,5./12.};
            c_t={0.3706701/6.,-1./6./6.,-17./36./6.};
        }
    }
    if (action=="Sym")
    {
        if(LO_or_EM==LO)
        {
            c_v={1.5240798/4.,-1./3./4.,-125./288./4.};     // Symanzik Action with Landau gauge
            c_a={0.6999177/4.,-1./3./4.,-125./288./4.};
            c_s={2.3547298,-1./4.,0.5};
            c_p={0.70640549,-1./4.,0.5};
            c_t={0.9724758/6.,-13./36./6.,-161./216./6.};
        }
        else if(LO_or_EM==EM)
        {
            c_v={0.3997992/4.,1./16./4.,-1./4./4.};         // Wilson Action with Feynman gauge
            c_a={0.2394365/4.,-3./16./4.,-1./4./4.};
            c_s={0.32682365,1./2.,5./12.};
            c_p={0.00609817,0.,5./12.};
            c_t={0.3706701/6.,-1./6./6.,-17./36./6.};
        }
    }
 
    vvd_t c={c_s,c_a,c_p,c_v,c_t};
    
    return c;
}

vd_t Zq_Oa2(const int LO_or_EM, const int imom)
{
    vd_t c_q(3);
    
    // Coefficients from the Mathematica file 'O(g2a2).nb'
    
    // Nf=4 analysis
    if(action=="Iwa")
    {
        if(LO_or_EM==LO)
        {
            c_q={0.6202244+1.8490436/(double)Np[imom],-0.0748167-0.963033/(double)Np[imom],0.0044};      //Iwasaki action
        }
        else if(LO_or_EM==EM)
        {
            c_q={-0.0112397+2.26296238/(double)Np[imom],31./240.-101./120./(double)Np[imom],17./120.};	   //Wilson action (QED)
        }
    }
    // Nf=2 analysis
    if (action=="Sym")
    {
        if(LO_or_EM==LO)
        {
            c_q={1.14716212+2.07733285/(double)Np[imom],-73./360.-157./180./(double)Np[imom],7./240.};   //Symanzik action
        }
        else if(LO_or_EM==EM)
        {
            c_q={-0.0112397+2.26296238/(double)Np[imom],31./240.-101./120./(double)Np[imom],17./120.};	   //Wilson action (QED)
        }
    }
    
    return c_q;
}

//subtraction of O(a^2) effects to bilinears
double subtraction(const int imom, const int ibil, const int LO_or_EM)
{
    double sub=0.0;
    
    vvd_t c=pr_bil_Oa2(LO_or_EM);  // c[ibil][k]
    
    if(LO_or_EM==LO)
        sub = g2_tilde*(p2_tilde[imom]*(c[ibil][0]+c[ibil][1]*log(p2_tilde[imom]))+c[ibil][2]*p4_tilde[imom]/p2_tilde[imom])/(12.*M_PI*M_PI);
    else if(LO_or_EM==EM)
        sub = (p2_tilde[imom]*(c[ibil][0]+c[ibil][1]*log(p2_tilde[imom]))+c[ibil][2]*p4_tilde[imom]/p2_tilde[imom])/(16.*M_PI*M_PI);
    
    return sub;
}

//subtraction of O(a^2) effects to Zq
double subtraction_q(const int imom, const int LO_or_EM)
{
    double sub=0.0;
    
    vd_t c=Zq_Oa2(LO_or_EM,imom);  // c[k]
    
    if(LO_or_EM==LO)
        sub = g2_tilde*(p2_tilde[imom]*(c[0]+c[1]*log(p2_tilde[imom]))+c[2]*p4_tilde[imom]/p2_tilde[imom])/(12.*M_PI*M_PI);
    else if(LO_or_EM==EM)
        sub = (p2_tilde[imom]*(c[0]+c[1]*log(p2_tilde[imom]))+c[2]*p4_tilde[imom]/p2_tilde[imom])/(16.*M_PI*M_PI);
    
    return sub;
}

