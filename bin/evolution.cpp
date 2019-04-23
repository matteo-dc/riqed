
//**  Evolution to the scale 1/a from Nuria  **//

#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"

#define Z3 1.2020569031595942

double alphas(int Nf,double mu2) // mu2 is in physical units (dimensional quantity)
{
    
    double CF = (Nc*Nc-1.)/(2.*Nc);
    
    double lam0,L2,LL2,b1,b2,b3;
    double als0, als1, als2, als3;
    double beta_0,beta_1,beta_2,beta_3;
    
    beta_0 = (11.*Nc-2.*Nf)/3.;
    
    beta_1 = 34./3.*pow(Nc,2) - 10./3.*Nc*Nf-2*CF*Nf;
    beta_2 = (2857./54.)*pow(Nc,3) + pow(CF,2)*Nf -
    205./18.*CF*Nc*Nf -1415./54.*pow(Nc,2)*Nf +
    11./9.*CF*pow(Nf,2) + 79./54.*Nc*pow(Nf,2);
    
    beta_3 = (150653./486. - 44./9.*Z3)*pow(Nc,4) +
    (-39143./162. + 68./3.*Z3)*pow(Nc,3)*Nf +
    (7073./486. - 328./9.*Z3)*CF*pow(Nc,2)*Nf +
    (-2102./27. + 176./9.*Z3)*pow(CF,2)*Nc*Nf +
    23.*pow(CF,3)*Nf + (3965./162. + 56./9.*Z3)*pow(Nc,2)*pow(Nf,2) +
    (338./27. - 176./9.*Z3)*pow(CF,2)*pow(Nf,2) +
    (4288./243. + 112./9.*Z3)*CF*Nc*pow(Nf,2) + 53./243.*Nc*pow(Nf,3) +
    154./243.*CF*pow(Nf,3) +
    (-10./27. + 88./9.*Z3)*pow(Nc,2)*(pow(Nc,2)+36.) +
    (32./27. - 104./9.*Z3)*Nc*(pow(Nc,2)+6)*Nf +
    (-22./27. + 16./9.*Z3)*(pow(Nc,4) - 6.*pow(Nc,2) + 18.)/pow(Nc,2)*pow(Nf,2);
    
    b1=beta_1/beta_0/4./M_PI;
    b2=beta_2/beta_0/16./pow(M_PI,2);
    b3=beta_3/beta_0/64./pow(M_PI,3);
    
    lam0=LambdaQCD;
    
    L2   = log( mu2/(pow(lam0,2) ) );
    LL2  = log( L2 );
    
    als0 = 4.*M_PI/beta_0/L2;
    als1 = als0 - pow(als0,2)*b1*LL2;
    als2 = als1 + pow(als0,3)*(pow(b1,2)*(pow(LL2,2) - LL2 -1.) + b2);
    als3 = als2 + pow(als0,4)*(pow(b1,3)*(-pow(LL2,3)+5./2.*pow(LL2,2)+2*LL2-1./2.)-
                               3.*b1*b2*LL2 + b3/2.);
    
    return als3;
}


///////////////////////////////////
// evolution from mu=p to mu0=1/a
// Z(mu0)=Z(mu) c(mu0)/c(mu)
// def: c=c(mu)/c(mu0)
// -> Z(mu=1/a) = Z(mu) /c
//////////////////////////////////

double q_evolution_to_RIp_ainv(int Nf,double ainv,double a2p2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=(a2p2)*(1/a^2) (dimensional quantity)
    //mu0_2=(1/a^2)
    
    double mu_2 = a2p2*pow(ainv,2.0);    // p2
    double mu0_2= pow(ainv,2.0);         // 1/a2
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,mu_2)/(4*M_PI);
    al0=alphas(Nf,mu0_2)/(4*M_PI);
    
    ////////////////////////////////
    // N3LO FORMULA
    // Assuming landau gauge
    ///////////////////////////////////
    if(Nf==2){
        cmu = 1. + 2.03448 * alm + 35.9579 * pow(alm,2) + 1199.16 * pow(alm,3);
        cmu0 = 1. + 2.03448 * al0 + 35.9579 * pow(al0,2) + 1199.16 * pow(al0,3);
    }if(Nf==0){
        cmu = 1. + 2.0303 * alm + 42.1268 * pow(alm,2) + 1728.43 * pow(alm,3);
        cmu0 = 1. + 2.0303 * al0 + 42.1268 * pow(al0,2) + 1728.43 * pow(al0,3);
    }if(Nf==4){
        cmu = 1. + 2.4000 * alm + 29.6724 * pow(alm,2) + 719.141 * pow(alm,3);
        cmu0 = 1. + 2.4000 * al0 + 29.6724 * pow(al0,2) + 719.141 * pow(al0,3);
    }
    
    return cmu/cmu0;
}
double S_evolution_to_RIp_ainv(int Nf,double ainv,double a2p2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=(a2p2)*(1/a^2) (dimensional quantity)
    //mu0_2=(1/a^2)
    
    double mu_2 = a2p2*pow(ainv,2.0);    // p2
    double mu0_2= pow(ainv,2.0);         // 1/a2
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,mu_2)/(4*M_PI);
    al0=alphas(Nf,mu0_2)/(4*M_PI);
    
    ////////////////////////////////
    // N3LO FORMULA
    // Assuming landau gauge
    ///////////////////////////////////
    
    if(Nf==2){
        cmu = pow(alm,-12./29) * (1. - 8.55727 * alm - 125.423 * pow(alm,2) -
                                  3797.71 * pow(alm,3));
        
        cmu0 = pow(al0,-12./29) * (1. - 8.55727 * al0 - 125.423 * pow(al0,2) -
                                   3797.71 * pow(al0,3));
    }if(Nf==0){
        cmu = pow(alm,-4./11) * (1. - 8.08264 * alm - 151.012 * pow(alm,2) -
                                 5247.93 * pow(alm,3));
        
        cmu0 = pow(al0,-4./11) * (1. - 8.08264 * al0 - 151.012 * pow(al0,2) -
                                  5247.93 * pow(al0,3));
    }if(Nf==4){
        cmu = pow(alm,-12./25) * (1. - 9.38987 * alm - 96.2883 * pow(alm,2) -
                                  2403.82 * pow(alm,3));
        
        cmu0 = pow(al0,-12./25) * (1. - 9.38987 * al0 - 96.2883 * pow(al0,2) -
                                   2403.82 * pow(al0,3));
    }
    
    return cmu/cmu0;
}

double P_evolution_to_RIp_ainv(int Nf,double ainv,double a2p2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=(a2p2)*(1/a^2) (dimensional quantity)
    //mu0_2=(1/a^2)
    
    double mu_2 = a2p2*pow(ainv,2.0);    // p2
    double mu0_2= pow(ainv,2.0);         // 1/a2
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,mu_2)/(4*M_PI);
    al0=alphas(Nf,mu0_2)/(4*M_PI);
    
    ////////////////////////////////
    // N3LO FORMULA
    // Assuming landau gauge
    ///////////////////////////////////
    if(Nf==2){
        cmu = pow(alm,-12./29) * (1. - 8.55727 * alm - 125.423 * pow(alm,2) -
                                  3797.71 * pow(alm,3));
        
        cmu0 = pow(al0,-12./29) * (1. - 8.55727 * al0 - 125.423 * pow(al0,2) -
                                   3797.71 * pow(al0,3));
    }if(Nf==0){
        cmu = pow(alm,-4./11) * (1. - 8.08264 * alm - 151.012 * pow(alm,2) -
                                 5247.93 * pow(alm,3));
        
        cmu0 = pow(al0,-4./11) * (1. - 8.08264 * al0 - 151.012 * pow(al0,2) -
                                  5247.93 * pow(al0,3));
    }if(Nf==4){
        cmu = pow(alm,-12./25) * (1. - 9.38987 * alm - 96.2883 * pow(alm,2) -
                                  2403.82 * pow(alm,3));
        
        cmu0 = pow(al0,-12./25) * (1. - 9.38987 * al0 - 96.2883 * pow(al0,2) -
                                   2403.82 * pow(al0,3));
    }
    
    return cmu/cmu0;
}

double T_evolution_to_RIp_ainv(int Nf,double ainv,double a2p2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=(a2p2)*(1/a^2) (dimensional quantity)
    //mu0_2=(1/a^2)
    
    double mu_2 = a2p2*pow(ainv,2.0);    // p2
    double mu0_2= pow(ainv,2.0);         // 1/a2
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,mu_2)/(4*M_PI);
    al0=alphas(Nf,mu0_2)/(4*M_PI);
    
    ////////////////////////////////
    // N2LO FORMULA
    // Assuming landau gauge
    ///////////////////////////////////
    
    if(Nf==2){
        cmu = pow(alm,4./29) * (1. + 2.66852 * alm + 47.9701 * pow(alm,2));
        
        cmu0 = pow(al0,4./29) * (1. + 2.66852 * al0 + 47.9701 * pow(al0,2));
    }if(Nf==0){
        cmu = pow(alm,4./33) * (1. + 2.53260 * alm + 57.8740 * pow(alm,2));
        
        cmu0 = pow(al0,4./33) * (1. + 2.53260 * al0 + 57.8740 * pow(al0,2));
    }if(Nf==4){
        cmu = pow(alm,4./25) * (1. + 2.91662 * alm + 37.9471 * pow(alm,2));
        
        cmu0 = pow(al0,4./25) * (1. + 2.91662 * al0 + 37.9471 * pow(al0,2));
    }
    
    return cmu/cmu0;
}

double P_evolution_to_RIp_two_GeV(int Nf,double ainv,double a2p2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=(a2p2)*(1/a^2) (dimensional quantity)
    //mu0_2=(1/a^2)
    
    double mu_2 = a2p2*pow(ainv,2.0);    // p2
    double mu0_2= pow(2.0,2.0);          // (2GeV)^2
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,mu_2)/(4*M_PI);
    al0=alphas(Nf,mu0_2)/(4*M_PI);
    
    ////////////////////////////////
    // N3LO FORMULA
    // Assuming landau gauge
    ///////////////////////////////////
    if(Nf==2){
        cmu = pow(alm,-12./29) * (1. - 8.55727 * alm - 125.423 * pow(alm,2) -
                                  3797.71 * pow(alm,3));
        
        cmu0 = pow(al0,-12./29) * (1. - 8.55727 * al0 - 125.423 * pow(al0,2) -
                                   3797.71 * pow(al0,3));
    }if(Nf==0){
        cmu = pow(alm,-4./11) * (1. - 8.08264 * alm - 151.012 * pow(alm,2) -
                                 5247.93 * pow(alm,3));
        
        cmu0 = pow(al0,-4./11) * (1. - 8.08264 * al0 - 151.012 * pow(al0,2) -
                                  5247.93 * pow(al0,3));
    }if(Nf==4){
        cmu = pow(alm,-12./25) * (1. - 9.38987 * alm - 96.2883 * pow(alm,2) -
                                  2403.82 * pow(alm,3));
        
        cmu0 = pow(al0,-12./25) * (1. - 9.38987 * al0 - 96.2883 * pow(al0,2) -
                                   2403.82 * pow(al0,3));
    }
    
    return cmu/cmu0;
}


oper_t oper_t::evolveToAinv(const double ainv)
{
    cout<<endl;
    cout<<"----- evolution to the scale 1/a -----"<<endl<<endl;
    
    oper_t out=(*this);
    
    double cq=0.0;
    vd_t cO(0.0,5);
    
    double gamma_q=0.0;
    double gamma_bil[5]={0.0};
    double gamma_meslep[5][5]={0.0};
    
    if(free_analysis)
    {
        //  pure QED anomalous dimensions
        
        if(strcmp(an_suffix.c_str(),"")==0) //Feynman
        {
            gamma_q = 2.0;
            gamma_bil[0] = -6.0; /* S */
            gamma_bil[1] =  0.0; /* V */
            gamma_bil[2] = -6.0; /* P */
            gamma_bil[3] =  0.0; /* A */
            gamma_bil[4] =  2.0; /* T */
//            gamma_meslep[0][0] = -4.0;
//            gamma_meslep[1][1] = -2.0;
//            gamma_meslep[2][2] = +4.0/3.0;
//            gamma_meslep[3][3] = +4.0/3.0;
//            gamma_meslep[4][4] = -40.0/9.0;
//            gamma_meslep[3][4] = -1.0/6.0;
//            gamma_meslep[4][3] = -8.0;
            gamma_meslep[0][0] = -5.0;
            gamma_meslep[1][1] = -3.0;
            gamma_meslep[2][2] = +1.0/3.0;
            gamma_meslep[3][3] = +1.0/3.0;
            gamma_meslep[4][4] = -49.0/9.0;
            gamma_meslep[3][4] = -1.0/6.0;
            gamma_meslep[4][3] = -8.0;
        }
        else                                //Landau
        {
            gamma_q = 0.0;  //no Zq an. dim. in Landau gauge!
            gamma_bil[0] = -6.0; /* S */
            gamma_bil[1] =  0.0; /* V */
            gamma_bil[2] = -6.0; /* P */
            gamma_bil[3] =  0.0; /* A */
            gamma_bil[4] =  2.0; /* T */
//            gamma_meslep[0][0] = -4.0;
//            gamma_meslep[1][1] = -2.0;
//            gamma_meslep[2][2] = +4.0/3.0;
//            gamma_meslep[3][3] = +4.0/3.0;
//            gamma_meslep[4][4] = -40.0/9.0;
//            gamma_meslep[3][4] = -1.0/6.0;
//            gamma_meslep[4][3] = -8.0;
            gamma_meslep[0][0] = -5.0;
            gamma_meslep[1][1] = -3.0;
            gamma_meslep[2][2] = +1.0/3.0;
            gamma_meslep[3][3] = +1.0/3.0;
            gamma_meslep[4][4] = -49.0/9.0;
            gamma_meslep[3][4] = -1.0/6.0;
            gamma_meslep[4][3] = -8.0;
        }
        
        
        // Zq
        for(int imom=0;imom<out._linmoms;imom++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr=0;mr<out._nmr;mr++)
                    (out.jZq_EM)[imom][ijack][mr] =
                        jZq_EM[imom][ijack][mr] + 1.0/pow(4*M_PI,2.0)*0.5*gamma_q*log(p2[imom]);
        
        // Zbil
        for(int imom=0;imom<out._bilmoms;imom++)
            for(int ibil=0;ibil<nbil;ibil++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr1=0;mr1<out._nmr;mr1++)
                        for(int mr2=0;mr2<out._nmr;mr2++)
                            (out.jZ_EM)[imom][ibil][ijack][mr1][mr2] =
                                jZ_EM[imom][ibil][ijack][mr1][mr2] + 1.0/pow(4*M_PI,2.0)*0.5*gamma_bil[ibil]*log(p2[imom]);
        
        
        // Z4f
        for(int imom=0;imom<out._meslepmoms;imom++)
            for(int iop1=0;iop1<nbil;iop1++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr1=0;mr1<out._nmr;mr1++)
                        for(int mr2=0;mr2<out._nmr;mr2++)
                            (out.jZ_4f_EM)[imom][iop1][iop1][ijack][mr1][mr2] =
                                jZ_4f_EM[imom][iop1][iop1][ijack][mr1][mr2] + 1.0/pow(4*M_PI,2.0)*0.5*gamma_meslep[iop1][iop1]*log(p2[imom]);
        
    }
    else if(inte_analysis)
    {
        cout<<"QCD evolution not yet implemented."<<endl;
        
        // Zq
        for(int imom=0;imom<out._linmoms;imom++)
        {
            cq=q_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);
            
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr=0; mr<out._nmr; mr++)
                    (out.jZq)[imom][ijack][mr] = jZq[imom][ijack][mr]/cq;
        }
        
        // Zbil
        for(int imom=0;imom<out._bilmoms;imom++)
        {
            // Note that ZV  ZA are RGI because they're protected by the WIs
            cO[0]=S_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);    //S
            cO[1]=1.0;                                           //V
            cO[2]=P_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);    //P
            cO[3]=1.0;                                           //A
            cO[4]=T_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);    //T
            
            for(int ibil=0;ibil<nbil;ibil++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr1=0;mr1<out._nmr;mr1++)
                        for(int mr2=0;mr2<out._nmr;mr2++)
                            (out.jZ)[imom][ibil][ijack][mr1][mr2] = jZ[imom][ibil][ijack][mr1][mr2]/cO[ibil];
        }

        // Z4f
        for(int imom=0;imom<out._meslepmoms;imom++)
        {
            cO[0]=1.0;                                          //00
            cO[1]=1.0;                                          //11
            cO[2]=P_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);    //22
            cO[3]=P_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);    //33
            cO[4]=T_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);    //44
            
            for(int imom=0;imom<out._meslepmoms;imom++)
                for(int iop1=0;iop1<nbil;iop1++)
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int mr1=0;mr1<out._nmr;mr1++)
                            for(int mr2=0;mr2<out._nmr;mr2++)
                                (out.jZ_4f)[imom][iop1][iop1][ijack][mr1][mr2] =
                                    jZ_4f[imom][iop1][iop1][ijack][mr1][mr2]/cO[iop1];
        }
        
    }
    
    return out;
}

O4f_t evaluate_Carrasco(double a2p2)
{
    double LL=log(a2p2);
    
    O4f_t ZQED;
    /* The factor 1/(16*Pi^2) is included */
    ZQED << -0.0242556 + 0.0158314*LL, 0.00339251, 0.0101775, -0.0203551, -0.00508877,
            +0.00339251, -0.0486754 + 0.00949886*LL, -0.0203551, 0.0101775, -0.00254439,
            +0.00254439, -0.00508877, -0.0308496 - 0.00105543*LL, -0.00678503, 0.0,
            -0.00508877, 0.00254439, -0.00678503, -0.0308496 - 0.00105543*LL, 0.00203498 + 0.000527714*LL,
            -0.0610652, -0.0305326, 0.0, 0.0976791 + 0.0253303*LL, -0.0383375 + 0.0172387*LL;
    
    return ZQED;
}

oper_t oper_t::evolve_mixed(double ainv)
{
    cout<<endl;
    cout<<"----- mixed evolution of the eta -----"<<endl<<endl;
    
    oper_t out=(*this);
    
    double CF=(Nc*Nc-1.0)/(2.0*Nc);
    
    double gamma_q_s0 = 0.0; /* in the Landau gauge the one-loop strong an. dim. is zero */
    double gamma_q_e0 = 2.0;
    double gamma_q_se1 = -8.0;
    
    double gamma_bil_s0[5] =   {-6.0*CF,0.0,-6.0*CF,0.0,+2.0*CF};
    double gamma_bil_e0[5] =   {-6.0,   0.0,-6.0,   0.0,+2.0};
    double gamma_bil_se1[5] =  {-8.0,   0.0,-8.0,   0.0,-152.0/3.0};
    
    for(int imom=0;imom<out._linmoms;imom++)
    {
        // eta_q

        double al0 = alphas(Nf,pow(ainv,2.0)*p2[imom])/(4.0*M_PI);
        
        double UQCD_q    = 1.0/q_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);
        double UQCDinv_q = 1.0/UQCD_q;
        double UQED1_q = 0.5*gamma_q_e0*log(p2[imom])/pow(4.0*M_PI,2.0);
        double UQED2_q = 0.5*gamma_q_se1*log(p2[imom])/pow(4.0*M_PI,2.0) +
                         0.25*pow(log(p2[imom]),2.0)*gamma_q_e0*gamma_q_s0/pow(4.0*M_PI,2.0);
        
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<out._nmr;mr++)
                (out.jZq_EM)[imom][ijack][mr] =
                    jZq_EM[imom][ijack][mr] + al0*UQCDinv_q*UQED2_q + UQCDinv_q*UQED1_q - UQED1_q;
    
        // eta_bil
        double UQCD_bil[5] = {
            1.0/S_evolution_to_RIp_ainv(Nf,ainv,p2[imom]),
            1.0,
            1.0/P_evolution_to_RIp_ainv(Nf,ainv,p2[imom]),
//            1.0/P_evolution_to_RIp_two_GeV(Nf,ainv,p2[imom]),
            1.0,
            1.0/T_evolution_to_RIp_ainv(Nf,ainv,p2[imom])};

        for(int ibil=0;ibil<nbil;ibil++)
        {
            double UQCDinv_bil = 1.0/UQCD_bil[ibil];
            double UQED1_bil = 0.5*gamma_bil_e0[ibil]*log(p2[imom])/pow(4.0*M_PI,2.0);
            double UQED2_bil = 0.5*gamma_bil_se1[ibil]*log(p2[imom])/pow(4.0*M_PI,2.0) +
                               0.25*pow(log(p2[imom]),2.0)*gamma_bil_e0[ibil]*gamma_bil_s0[ibil]/pow(4.0*M_PI,2.0);
            
/* evolution of ZP to 2GeV*/
//            double UQED1_bil = 0.5*gamma_bil_e0[ibil]*log(p2[imom]*pow(ainv,2.0)/4.0)/pow(4.0*M_PI,2.0);
//            double UQED2_bil = 0.5*gamma_bil_se1[ibil]*log(p2[imom]*pow(ainv,2.0)/4.0)/pow(4.0*M_PI,2.0) +
//                               0.25*pow(log(p2[imom]*pow(ainv,2.0)/4.0),2.0)*gamma_bil_e0[ibil]*gamma_bil_s0[ibil]/pow(4.0*M_PI,2.0);

            
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr1=0;mr1<out._nmr;mr1++)
                    for(int mr2=0;mr2<out._nmr;mr2++)
                        (out.jZ_EM)[imom][ibil][ijack][mr1][mr2] =
                            jZ_EM[imom][ibil][ijack][mr1][mr2]  + al0*UQCDinv_bil*UQED2_bil +  UQCDinv_bil*UQED1_bil - UQED1_bil;
        }
        
    }
    
    O4f_t UQCD(O4f_t::Zero()), UQCDinv(O4f_t::Zero());
    O4f_t ZQCD(O4f_t::Zero()), ZQCDinv(O4f_t::Zero());
    O4f_t UQED1(O4f_t::Zero()),UQED2(O4f_t::Zero());
    O4f_t eta(O4f_t::Zero());
    
    O4f_t ZQEDan;
    
    double gamma_se1[5][5] = {
        {+4.0,+0.0,+0.0,+0.0,+0.0},
        {+0.0,-4.0,+0.0,+0.0,+0.0},
        {+0.0,+0.0,+484.0/9.0,+0.0,+0.0},
        {+0.0,+0.0,+0.0,+412.0/9.0,-38.0/9.0},
        {+0.0,+0.0,+0.0,-928.0/9.0,-428.0/27.0}}; // to be updated
    /* including the lepton */
//    double gamma_e0[5][5] = {
//        {-4.0,+0.0,+0.0,+0.0,+0.0},
//        {+0.0,-2.0,+0.0,+0.0,+0.0},
//        {+0.0,+0.0,+4.0/3.0,+0.0,+0.0},
//        {+0.0,+0.0,+0.0,+4.0/3.0,-1.0/6.0},
//        {+0.0,+0.0,+0.0,-8.0,-40.0/9.0}};
    /* without the lepton contribution */
    double gamma_e0[5][5] = {
        {-5.0,+0.0,+0.0,+0.0,+0.0},
        {+0.0,-3.0,+0.0,+0.0,+0.0},
        {+0.0,+0.0,+1.0/3.0,+0.0,+0.0},
        {+0.0,+0.0,+0.0,+1.0/3.0,-1.0/6.0},
        {+0.0,+0.0,+0.0,-8.0,-49.0/9.0}};
    
    double gamma_s0[5] = {0.0,0.0,-6.0*CF,-6.0*CF,+2.0*CF};
    
    for(int imom=0;imom<out._meslepmoms;imom++)
    {
        UQCD(0,0)=1.0;
        UQCD(1,1)=1.0;
        UQCD(2,2)=1.0/P_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);
        UQCD(3,3)=1.0/P_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);
        UQCD(4,4)=1.0/T_evolution_to_RIp_ainv(Nf,ainv,p2[imom]);
        
        UQCDinv = UQCD.inverse();
        
        ZQEDan = evaluate_Carrasco(p2[imom]);
        
        double al0 = alphas(Nf,pow(ainv,2.0)*p2[imom])/(4.0*M_PI);
        
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0;mr1<out._nmr;mr1++)
                for(int mr2=0;mr2<out._nmr;mr2++)
                {
                    for(int iop1=0;iop1<nbil;iop1++)
                        for(int iop2=0;iop2<nbil;iop2++)
                        {
                            ZQCD(iop1,iop2) = jZ_4f[imom][iop1][iop2][ijack][mr1][mr2];
                            eta(iop1,iop2)  = jZ_4f_EM[imom][iop1][iop2][ijack][mr1][mr2];
                            
                            UQED1(iop1,iop2) = 0.5*gamma_e0[iop1][iop2]*log(p2[imom])/pow(4.0*M_PI,2.0);
                            UQED2(iop1,iop2) = 0.5*gamma_se1[iop1][iop2]*log(p2[imom])/pow(4.0*M_PI,2.0) +
                                               0.125*pow(log(p2[imom]),2.0)*gamma_e0[iop1][iop2]*(gamma_s0[iop1]+gamma_s0[iop2])/pow(4.0*M_PI,2.0);
                        }
                    
                    ZQCDinv = ZQCD.inverse();
                    
                    if(!QCD_on_the_right)
                    {
                        // eta_OLD_4f
                        for(int iop1=0;iop1<nbil;iop1++)
                            for(int iop2=0;iop2<nbil;iop2++)
                                (out.jZ_4f_EM)[imom][iop1][iop2][ijack][mr1][mr2] =
                                    eta(iop1,iop2) + al0*(ZQCDinv*UQCDinv*UQED2*ZQCD)(iop1,iop2) + (ZQCDinv*UQCDinv*UQED1*ZQCD)(iop1,iop2)
                                    - UQED1(iop1,iop2);
                        
                    }
                    else if(QCD_on_the_right)
                    {
                        // eta_NEW_4f
                        for(int iop1=0;iop1<nbil;iop1++)
                            for(int iop2=0;iop2<nbil;iop2++)
                                (out.jZ_4f_EM)[imom][iop1][iop2][ijack][mr1][mr2] =
                                    (UQCD*eta*UQCDinv)(iop1,iop2) + al0*(UQED2*UQCDinv)(iop1,iop2) + (UQCD*ZQEDan*UQCDinv)(iop1,iop2) +
                                    (UQED1*UQCDinv)(iop1,iop2) - ZQEDan(iop1,iop2) - UQED1(iop1,iop2);
                        
                    }
                }
    }
    
    return out;
    
}
