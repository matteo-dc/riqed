#include "aliases.hpp"
#include "global.hpp"


double** pr_bil_Oa2(int ibil, int LO_or_EM)
{
    double c_v[3], c_a[3], c_s[3], c_p[3], c_t[3];
    
    // Coefficients from the Mathematica file 'O(g2a2).nb'
    
    // WARNING: we divide the following coefficients by 4 (or 6 for T)
    //          to account for the sum on Lorentz indices!
    
    if(action=="Iwa")
    {
        if(LO_or_EM==LO)
        {
            c_v[3]={0.2881372/4.,-0.2095/4.,-0.516583/4.};     // Iwasaki Action with Landau gauge
            c_a={0.9637998/4.,-0.2095/4.,-0.516583/4.};
            c_s={2.02123300,-1./4.,0.376167};
            c_p={0.66990790,-1./4.,0.376167};
            c_t={0.3861012/6.,-0.196/6.,-0.814167/6.};
        }
        else if(LO_or_EM==EM)
        {
            c_a={0.3997992/4.,1./16./4.,-1./4./4.};         // Wilson Action with Feynman gauge
            c_v={0.2394365/4.,-3./16./4.,-1./4./4.};
            c_s={0.32682365,1./2.,5./12.};
            c_p={0.00609817,0.,5./12.};
            c_t={0.3706701/6.,-1./6./6.,-17./36./6.};
        }
    }
    if (action=="Sym")
    {
        if(LO_or_EM==LO)
        {
            c_a={1.5240798/4.,-1./3./4.,-125./288./4.};     // Symanzik Action with Landau gauge
            c_v={0.6999177/4.,-1./3./4.,-125./288./4.};
            c_s={2.3547298,-1./4.,0.5};
            c_p={0.70640549,-1./4.,0.5};
            c_t={0.9724758/6.,-13./36./6.,-161./216./6.};
        }
        else if(LO_or_EM==EM)
        {
            c_a={0.3997992/4.,1./16./4.,-1./4./4.};         // Wilson Action with Feynman gauge
            c_v={0.2394365/4.,-3./16./4.,-1./4./4.};
            c_s={0.32682365,1./2.,5./12.};
            c_p={0.00609817,0.,5./12.};
            c_t={0.3706701/6.,-1./6./6.,-17./36./6.};
        }

    }
 
    double c[5][3]={c_s,c_a,c_p,c_v,c_t};
    
    return c;
}


double

if(strcmp(arg[8],"sym")==0)
{
    c_q={1.14716212+2.07733285/(double)Np[imom],-73./360.-157./180./(double)Np[imom],7./240.};   //Symanzik action
    
    c_q_em={-0.0112397+2.26296238/(double)Np[imom],31./240.-101./120./(double)Np[imom],17./120.};	   //Wilson action (QED)
}
else if(strcmp(arg[8],"iwa")==0)
{
    c_q={0.6202244+1.8490436/(double)Np[imom],-0.0748167-0.963033/(double)Np[imom],0.0044};      //Iwasaki action
    
    c_q_em={-0.0112397+2.26296238/(double)Np[imom],31./240.-101./120./(double)Np[imom],17./120.};	   //Wilson action (QED)
}
