#include "aliases.hpp"
#include "global.hpp"
#include "operations.hpp"

//debug
#include <iostream>

double subZq(const int imom, const vector<double> a2p2_vec, const vector<double> a4p4_vec, const int LO_or_EM)
{
    double a2p2 = a2p2_vec[imom];
    double a4p4 = a4p4_vec[imom];
    
    double Zq_sub=0.0;
    
    
    if(free_analysis)  // pure QED
    {
        if(LO_or_EM==1) //  pure QED  (Plaquette)
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
//                cout<<" - Subtracting QED to Zq_EM: Plaquette/Feynman"<<endl;

                Zq_sub = - 0.00007117813150874176*a2p2 + (0.015227459368423301*a4p4)/a2p2 +
                        (0.0008179574721126228*a2p2 - (0.005329916431185478*a4p4)/a2p2)*log(a2p2);
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QED to Zq_EM: Plaquette/Landau"<<endl;

                Zq_sub =  0.009529331412677406*a2p2 + (0.0127208492759996*a4p4)/a2p2 +
                        (-0.001556757769504669*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
            }
        }
    }
    
    else if(inte_analysis)  // QCD+QED
    {
        if(LO_or_EM==0) // pure QCD   (Iwasaki)
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
                cout<<"QCD subtraction in Feynman gauge not implemented!"<<endl;
                exit(0);
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QCD to Zq: Iwasaki/Landau"<<endl;

                Zq_sub =  0.003927646025581469*a2p2 + (0.011737083908572275*a4p4)/a2p2 +
                            (-0.0004737841872855492*a2p2 - (0.006098477715414467*a4p4)/a2p2)*log(a2p2);
            }
        }
        if(LO_or_EM==1) // QCD+QED (Plaquette) <- we subtract the pure QED effects, since QCD is already factorized.
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
//                cout<<" - Subtracting QED to Z_EM: Plaquette/Feynman"<<endl;

                Zq_sub = - 0.00007117813150874176*a2p2 + (0.015227459368423301*a4p4)/a2p2 +
                            (0.0008179574721126228*a2p2 - (0.005329916431185478*a4p4)/a2p2)*log(a2p2);
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QED to Z_EM: Plaquette/Landau"<<endl;
                
                Zq_sub =  0.009529331412677406*a2p2 + (0.0127208492759996*a4p4)/a2p2 +
                            (-0.001556757769504669*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
            }
        }

    }
    return Zq_sub;
}

double subZ(const int imom, const vector<double> a2p2_vec, const vector<double> a4p4_vec, const int ibil, const int LO_or_EM)
{
    // {"S","V","P","A","T"}
    
    double a2p2 = a2p2_vec[imom];
    double a4p4 = a4p4_vec[imom];
    
    double Z_sub[5]={0.0};
    
    if(free_analysis)  // pure QED
    {
        if(LO_or_EM==1) //  pure QED  (Plaquette)
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
//                cout<<" - Subtracting QED to Z_EM: Plaquette/Feynman"<<endl;
                
                Z_sub[0] = - 0.002140808956604977*a2p2 + (0.012588884766879431*a4p4)/a2p2 +
                            (- 0.002348329516710433*a2p2 - (0.005329916431185476*a4p4)/a2p2)*log(a2p2);
                Z_sub[1] = - 0.0015874433121423493*a2p2 + (0.01681060286283483*a4p4)/a2p2 +
                            (0.0020053150929212685*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[2] = - 0.00010978783505045093*a2p2 + (0.012588884766879431*a4p4)/a2p2 +
                            (0.0008179574721126228*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[3] = - 0.0026029475403456337*a2p2 + (0.01681060286283483*a4p4)/a2p2 +
                            (0.0004221715985097406*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[4] = - 0.0024185113232466924*a2p2 + (0.018217840117295304*a4p4)/a2p2 +
                            (0.001873388579244967*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QED to Z_EM: Plaquette/Landau"<<endl;
                
                Z_sub[0] = - 0.006938001612551163*a2p2 + (0.009290702927249964*a4p4)/a2p2 +
                            (0.000026385724906858877*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[1] = 0.0034607181161415185*a2p2 + (0.015293457454418331*a4p4)/a2p2 +
                            (0.0008179574721126228*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[2] = 0.004183353463026625*a2p2 + (0.009290702927249964*a4p4)/a2p2 +
                            (0.000026385724906858877*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[3] = - 0.002099953089073398*a2p2 + (0.015293457454418331*a4p4)/a2p2 +
                            (0.0008179574721126228*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[4] = 0.001366263601052909*a2p2 + (0.01729437351928313*a4p4)/a2p2 +
                            (0.0010818168320392034*a2p2 - (0.005329916431185478*a4p4)/a2p2)*log(a2p2);
            }
        }
    }
    
    else if(inte_analysis)  // QCD+QED
    {
        if(LO_or_EM==0) // pure QCD   (Iwasaki)
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
                cout<<"QCD subtraction in Feynman gauge not implemented!"<<endl;
                exit(0);
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QCD to Z: Iwasaki/Landau"<<endl;
                
                Z_sub[0] = - 0.008872018466143912*a2p2 + (0.009354978553123072*a4p4)/a2p2 +
                            (0.0011093593071259786*a2p2 - (0.006098477715414468*a4p4)/a2p2)*log(a2p2);
                Z_sub[1] = 0.0021029591619405857*a2p2 + (0.015008383971666636*a4p4)/a2p2 +
                            (0.000852890061031311*a2p2 - (0.006098477715414467*a4p4)/a2p2)*log(a2p2);
                Z_sub[2] = - 0.0003146592683752582*a2p2 + (0.009354978553123072*a4p4)/a2p2 +
                            (0.0011093593071259786*a2p2 - (0.006098477715414468*a4p4)/a2p2)*log(a2p2);
                Z_sub[3] = - 0.002175720436943741*a2p2 + (0.015008383971666636*a4p4)/a2p2 +
                            (0.000852890061031311*a2p2 - (0.006098477715414467*a4p4)/a2p2)*log(a2p2);
                Z_sub[4] = 0.0014825948847944624*a2p2 + (0.016892856666230475*a4p4)/a2p2 +
                            (0.0007674003123330885*a2p2 - (0.006098477715414467*a4p4)/a2p2)*log(a2p2);
            }
        }
        if(LO_or_EM==1) // QCD+QED (Plaquette) <- we subtract the pure QED effects, since QCD is already factorized.
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
//                cout<<" - Subtracting QED to Z_EM: Plaquette/Feynman"<<endl;
                
                Z_sub[0] = - 0.002140808956604977*a2p2 + (0.012588884766879431*a4p4)/a2p2 +
                (- 0.002348329516710433*a2p2 - (0.005329916431185476*a4p4)/a2p2)*log(a2p2);
                Z_sub[1] = - 0.0015874433121423493*a2p2 + (0.01681060286283483*a4p4)/a2p2 +
                (0.0020053150929212685*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[2] = - 0.00010978783505045093*a2p2 + (0.012588884766879431*a4p4)/a2p2 +
                (0.0008179574721126228*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[3] = - 0.0026029475403456337*a2p2 + (0.01681060286283483*a4p4)/a2p2 +
                (0.0004221715985097406*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[4] = - 0.0024185113232466924*a2p2 + (0.018217840117295304*a4p4)/a2p2 +
                (0.001873388579244967*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QED to Z_EM: Plaquette/Landau"<<endl;

                Z_sub[0] = - 0.006938001612551163*a2p2 + (0.009290702927249964*a4p4)/a2p2 +
                (0.000026385724906858877*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[1] = 0.0034607181161415185*a2p2 + (0.015293457454418331*a4p4)/a2p2 +
                (0.0008179574721126228*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[2] = 0.004183353463026625*a2p2 + (0.009290702927249964*a4p4)/a2p2 +
                (0.000026385724906858877*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[3] = - 0.002099953089073398*a2p2 + (0.015293457454418331*a4p4)/a2p2 +
                (0.0008179574721126228*a2p2 - (0.005329916431185477*a4p4)/a2p2)*log(a2p2);
                Z_sub[4] = 0.001366263601052909*a2p2 + (0.01729437351928313*a4p4)/a2p2 +
                (0.0010818168320392034*a2p2 - (0.005329916431185478*a4p4)/a2p2)*log(a2p2);
            }
        }
        
    }
    return Z_sub[ibil];
}

double subZ4f(const int imom, const vector<double> a2p2_vec, const vector<double> a4p4_vec, const int iop1, const int iop2, const int LO_or_EM)
{
    double a2p2 = a2p2_vec[imom];
    double a4p4 = a4p4_vec[imom];
    
    double Z4f_sub[5][5]={0.0};
    
    if(free_analysis)  // pure QED
    {
        if(LO_or_EM==1) //  pure QED  (Plaquette)
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
//                cout<<" - Subtracting QED to Z4f_EM: Plaquette/Feynman"<<endl;
                
                Z4f_sub[0][0] = - (0.005017237851226561*a4p4)/a2p2 + a2p2*(-0.0021075326875911096 - 0.000784242379176081*log(a2p2))- (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[1][1] = - (0.004104872255397198*a4p4)/a2p2 + a2p2*(0.013796511549751856 - 0.0027631717471904905*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[2][2] = - (0.0017594733131997531*a4p4)/a2p2 + a2p2*(0.013257679866187257 - 0.0016637665427380406*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[3][3] = Z4f_sub[2][2];
                Z4f_sub[4][4] = - (0.0054949123497923785*a4p4)/a2p2 + a2p2*(0.0033734268908705207 - 0.0018103550760305853*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[0][1] = a2p2*(-0.00011283380313369834 - 0.00017590483271239196*log(a2p2));
                Z4f_sub[1][0] = Z4f_sub[0][1];
                Z4f_sub[0][2] = a2p2*(-0.00033850352025908774 - 0.0005277144981371759*log(a2p2));
                Z4f_sub[1][3] = Z4f_sub[0][2];
                Z4f_sub[0][3] = a2p2*(0.011964667751725307 - 0.007388002973920463*log(a2p2));
                Z4f_sub[1][2] = Z4f_sub[0][3];
                Z4f_sub[0][4] = a2p2*(0.0029911670434742263 - 0.0018470007434801157*log(a2p2));
                Z4f_sub[3][0] = Z4f_sub[0][4];
                Z4f_sub[2][1] = Z4f_sub[0][4];
                Z4f_sub[1][4] = a2p2*(0.0000846253523502738 + 0.00013192862453429398*log(a2p2));
                Z4f_sub[3][1] = Z4f_sub[1][4];
                Z4f_sub[2][0] = Z4f_sub[1][4];
                Z4f_sub[2][3] = a2p2*(0.00022566901350605842 + 0.0003518096654247839*log(a2p2));
                Z4f_sub[3][2] = Z4f_sub[2][3];
                Z4f_sub[3][4] = - (0.0000760326827200058*a4p4)/a2p2 + a2p2*(-0.0013253345922918892 + 0.00016491067512496786*log(a2p2));
                Z4f_sub[4][0] = a2p2*(0.03589400367734751 - 0.02216400892176139*log(a2p2));
                Z4f_sub[4][1] = a2p2*(0.0010154915630553318 + 0.0015831434944115277*log(a2p2));
                Z4f_sub[4][3] = - (0.003649563704501097*a4p4)/a2p2 + a2p2*(-0.06361610940191612 + 0.007915717472057639*log(a2p2));
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QED to Z4f_EM: Plaquette/Landau"<<endl;
                
                Z4f_sub[0][0] = - (0.0047937725734647245*a4p4)/a2p2 + a2p2*(-0.0021178665687984636 - 0.0012240044609570608*log(a2p2))  - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[1][1] = - (0.0038685349655967977*a4p4)/a2p2 + a2p2*(0.009981613661650554 - 0.0020155762081628245*log(a2p2))  - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[2][2] = - (0.0005336713393921483*a4p4)/a2p2 + a2p2*(0.011124783988481848 - 0.0015758141263818447*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[3][3] = Z4f_sub[2][2];
                Z4f_sub[4][4] = - (0.005596979369925417*a4p4)/a2p2 + a2p2*(0.0015342329797710143 - 0.0016344502433181935*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[0][1] = - 0.0006178523561349905*a2p2;
                Z4f_sub[1][0] = Z4f_sub[0][1];
                Z4f_sub[0][2] = - 0.0018535591792629643*a2p2;
                Z4f_sub[1][3] = Z4f_sub[0][2];
                Z4f_sub[0][3] = a2p2*(0.011860826625811465 - 0.006332573977646111*log(a2p2));
                Z4f_sub[1][2] = Z4f_sub[0][3];
                Z4f_sub[0][4] = a2p2*(0.002965206656452866 - 0.0015831434944115277*log(a2p2));
                Z4f_sub[3][0] = Z4f_sub[0][4];
                Z4f_sub[2][1] = Z4f_sub[0][4];
                Z4f_sub[1][4] = 0.00046338926710124296*a2p2;
                Z4f_sub[3][1] = Z4f_sub[1][4];
                Z4f_sub[2][0] = Z4f_sub[1][4];
                Z4f_sub[2][3] = 0.0012357061195086428*a2p2;
                Z4f_sub[3][2] = Z4f_sub[2][3];
                Z4f_sub[3][4] = - (0.00007710542075181905*a4p4)/a2p2 + a2p2*(-0.0010082883656986575 + 0.00006596420672424736*log(a2p2));
                Z4f_sub[4][0] = a2p2*(0.03558247987743439 - 0.018997721932938333*log(a2p2));
                Z4f_sub[4][1] = 0.005560675426930901*a2p2;
                Z4f_sub[4][3] = - (0.0037010517526553446*a4p4)/a2p2 + a2p2*(-0.04839790825664813 + 0.0031662869888230555*log(a2p2));
            }
        }
    }
    
    else if(inte_analysis)  // QCD+QED
    {
        if(LO_or_EM==0) // pure QCD   (Iwasaki)
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
                cout<<"QCD subtraction in Feynman gauge not implemented!"<<endl;
                exit(0);
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QCD to Z4f: Iwasaki/Landau"<<endl;
                
                Z4f_sub[0][0] = (0.015008383971666636*a4p4)/a2p2 + a2p2*(-0.00003638063750157662 + 0.0008528900610313113*log(a2p2)) - (0.006098477715414467*a4p4*log(a2p2))/a2p2;
                Z4f_sub[1][1] = Z4f_sub[0][0];
                Z4f_sub[2][2] = (0.009354978553123072*a4p4)/a2p2 + a2p2*(-0.0045933388672595825 + 0.0011093593071259786*log(a2p2))  - (0.006098477715414467*a4p4*log(a2p2))/a2p2;
                Z4f_sub[3][3] = Z4f_sub[2][2];
                Z4f_sub[4][4] = (0.016892856666230475*a4p4)/a2p2 + a2p2*(0.0014825948847944637 + 0.0007674003123330888*log(a2p2)) - (0.006098477715414467*a4p4*log(a2p2))/a2p2;
                Z4f_sub[0][1] = 0.002139339799442163*a2p2;
                Z4f_sub[1][0] = Z4f_sub[0][1];
                Z4f_sub[2][3] = - 0.004278679598884326*a2p2;
                Z4f_sub[3][2] = Z4f_sub[2][3];
            }
        }
        if(LO_or_EM==1) // QCD+QED (Plaquette) <- we subtract the pure QED effects, since QCD is already factorized.
        {
            if(strcmp(an_suffix.c_str(),"")==0) //Feynman
            {
//                cout<<" - Subtracting QED to Z4f_EM: Plaquette/Feynman"<<endl;
                
                Z4f_sub[0][0] = - (0.005017237851226561*a4p4)/a2p2 + a2p2*(-0.0021075326875911096 - 0.000784242379176081*log(a2p2))- (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[1][1] = - (0.004104872255397198*a4p4)/a2p2 + a2p2*(0.013796511549751856 - 0.0027631717471904905*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[2][2] = - (0.0017594733131997531*a4p4)/a2p2 + a2p2*(0.013257679866187257 - 0.0016637665427380406*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[3][3] = Z4f_sub[2][2];
                Z4f_sub[4][4] = - (0.0054949123497923785*a4p4)/a2p2 + a2p2*(0.0033734268908705207 - 0.0018103550760305853*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[0][1] = a2p2*(-0.00011283380313369834 - 0.00017590483271239196*log(a2p2));
                Z4f_sub[1][0] = Z4f_sub[0][1];
                Z4f_sub[0][2] = a2p2*(-0.00033850352025908774 - 0.0005277144981371759*log(a2p2));
                Z4f_sub[1][3] = Z4f_sub[0][2];
                Z4f_sub[0][3] = a2p2*(0.011964667751725307 - 0.007388002973920463*log(a2p2));
                Z4f_sub[1][2] = Z4f_sub[0][3];
                Z4f_sub[0][4] = a2p2*(0.0029911670434742263 - 0.0018470007434801157*log(a2p2));
                Z4f_sub[3][0] = Z4f_sub[0][4];
                Z4f_sub[2][1] = Z4f_sub[0][4];
                Z4f_sub[1][4] = a2p2*(0.0000846253523502738 + 0.00013192862453429398*log(a2p2));
                Z4f_sub[3][1] = Z4f_sub[1][4];
                Z4f_sub[2][0] = Z4f_sub[1][4];
                Z4f_sub[2][3] = a2p2*(0.00022566901350605842 + 0.0003518096654247839*log(a2p2));
                Z4f_sub[3][2] = Z4f_sub[2][3];
                Z4f_sub[3][4] = - (0.0000760326827200058*a4p4)/a2p2 + a2p2*(-0.0013253345922918892 + 0.00016491067512496786*log(a2p2));
                Z4f_sub[4][0] = a2p2*(0.03589400367734751 - 0.02216400892176139*log(a2p2));
                Z4f_sub[4][1] = a2p2*(0.0010154915630553318 + 0.0015831434944115277*log(a2p2));
                Z4f_sub[4][3] = - (0.003649563704501097*a4p4)/a2p2 + a2p2*(-0.06361610940191612 + 0.007915717472057639*log(a2p2));
            }
            else                                //Landau
            {
//                cout<<" - Subtracting QED to Z4f_EM: Plaquette/Landau"<<endl;
                
                Z4f_sub[0][0] = - (0.0047937725734647245*a4p4)/a2p2 + a2p2*(-0.0021178665687984636 - 0.0012240044609570608*log(a2p2))  - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[1][1] = - (0.0038685349655967977*a4p4)/a2p2 + a2p2*(0.009981613661650554 - 0.0020155762081628245*log(a2p2))  - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[2][2] = - (0.0005336713393921483*a4p4)/a2p2 + a2p2*(0.011124783988481848 - 0.0015758141263818447*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[3][3] = Z4f_sub[2][2];
                Z4f_sub[4][4] = - (0.005596979369925417*a4p4)/a2p2 + a2p2*(0.0015342329797710143 - 0.0016344502433181935*log(a2p2)) - (0.0014805323419959657*a4p4*log(a2p2))/a2p2;
                Z4f_sub[0][1] = - 0.0006178523561349905*a2p2;
                Z4f_sub[1][0] = Z4f_sub[0][1];
                Z4f_sub[0][2] = - 0.0018535591792629643*a2p2;
                Z4f_sub[1][3] = Z4f_sub[0][2];
                Z4f_sub[0][3] = a2p2*(0.011860826625811465 - 0.006332573977646111*log(a2p2));
                Z4f_sub[1][2] = Z4f_sub[0][3];
                Z4f_sub[0][4] = a2p2*(0.002965206656452866 - 0.0015831434944115277*log(a2p2));
                Z4f_sub[3][0] = Z4f_sub[0][4];
                Z4f_sub[2][1] = Z4f_sub[0][4];
                Z4f_sub[1][4] = 0.00046338926710124296*a2p2;
                Z4f_sub[3][1] = Z4f_sub[1][4];
                Z4f_sub[2][0] = Z4f_sub[1][4];
                Z4f_sub[2][3] = 0.0012357061195086428*a2p2;
                Z4f_sub[3][2] = Z4f_sub[2][3];
                Z4f_sub[3][4] = - (0.00007710542075181905*a4p4)/a2p2 + a2p2*(-0.0010082883656986575 + 0.00006596420672424736*log(a2p2));
                Z4f_sub[4][0] = a2p2*(0.03558247987743439 - 0.018997721932938333*log(a2p2));
                Z4f_sub[4][1] = 0.005560675426930901*a2p2;
                Z4f_sub[4][3] = - (0.0037010517526553446*a4p4)/a2p2 + a2p2*(-0.04839790825664813 + 0.0031662869888230555*log(a2p2));
            }
        }
        
    }
    return Z4f_sub[iop1][iop2];
}

oper_t oper_t::subOa2(const int b)
{
    cout<<endl;
    cout<<"----- subtraction of O(a2) effects -----"<<endl<<endl;
    
    oper_t out=(*this);
    
    double CF = 4.0/3.0;
    double g2 = 6.0/beta[b];
    double g2b = g2/plaquette;
    
    // Zq
    for(int imom=0;imom<out._linmoms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<out._nmr;mr++)
            {
                (out.jZq)[imom][ijack][mr] -= CF*g2b*subZq(imom,p2_tilde,p4_tilde,0);
                (out.jZq_EM)[imom][ijack][mr] -= subZq(imom,p2_tilde,p4_tilde,1);
            }
    
    // Zbil
    for(int imom=0;imom<out._bilmoms;imom++)
        for(int ibil=0;ibil<nbil;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr1=0;mr1<out._nmr;mr1++)
                    for(int mr2=0;mr2<out._nmr;mr2++)
                    {
                        (out.jZ)[imom][ibil][ijack][mr1][mr2] -= CF*g2b*subZ(imom,p2_tilde,p4_tilde,ibil,0);
                        (out.jZ_EM)[imom][ibil][ijack][mr1][mr2] -= subZ(imom,p2_tilde,p4_tilde,ibil,1);
                    }
    
    
    // ZV/ZA and ZP/ZS
    for(int imom=0;imom<out._bilmoms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0;mr1<out._nmr;mr1++)
                for(int mr2=0;mr2<out._nmr;mr2++)
                {
//                    (out.jZVoverZA)[imom][0][ijack][mr1][mr2] = jZVoverZA[imom][0][ijack][mr1][mr2];
//                    (out.jZPoverZS)[imom][0][ijack][mr1][mr2] = jZPoverZS[imom][0][ijack][mr1][mr2];
                    (out.jZVoverZA)[imom][0][ijack][mr1][mr2] = (out.jZ)[imom][1][ijack][mr1][mr2]/(out.jZ)[imom][3][ijack][mr1][mr2];
                    (out.jZPoverZS)[imom][0][ijack][mr1][mr2] = (out.jZ)[imom][2][ijack][mr1][mr2]/(out.jZ)[imom][0][ijack][mr1][mr2];
                }
    
    
    // Z4f
    for(int imom=0;imom<out._meslepmoms;imom++)
        for(int iop1=0;iop1<nbil;iop1++)
            for(int iop2=0;iop2<nbil;iop2++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr1=0;mr1<out._nmr;mr1++)
                        for(int mr2=0;mr2<out._nmr;mr2++)
                        {
                            (out.jZ_4f)[imom][iop1][iop2][ijack][mr1][mr2] -= CF*g2b*subZ4f(imom,p2_tilde,p4_tilde,iop1,iop2,0);
                            (out.jZ_4f_EM)[imom][iop1][iop2][ijack][mr1][mr2] -= subZ4f(imom,p2_tilde,p4_tilde,iop1,iop2,1);
                        }
    
    return out;
}

//vvd_t pr_bil_Oa2(const int LO_or_EM)
//{
//    vd_t c_v(3), c_a(3), c_s(3), c_p(3), c_t(3);
//
//    // Coefficients from the Mathematica file 'O(g2a2).nb'
//    
//    // We divide the following coefficients by 4 (or 6 for T)
//    // to account for the sum on Lorentz indices!
//    
//    // The coefficients c_v and c_a are used respectively on Gv and Ga,
//    // BUT in the physical basis ZA=Zq/Gv and ZV=Zq/Ga!!!
//    
//    if(action=="Iwa")
//    {
//        if(LO_or_EM==LO)
//        {
//            c_a={0.2881372/4.,-0.2095/4.,-0.516583/4.};     // Iwasaki Action with Landau gauge
//            c_v={0.9637998/4.,-0.2095/4.,-0.516583/4.};
//            c_s={2.02123300,-1./4.,0.376167};
//            c_p={0.66990790,-1./4.,0.376167};
//            c_t={0.3861012/6.,-0.196/6.,-0.814167/6.};
//        }
//        else if(LO_or_EM==EM)
//        {
//            c_v={0.3997992/4.,1./16./4.,-1./4./4.};         // Wilson Action with Feynman gauge
//            c_a={0.2394365/4.,-3./16./4.,-1./4./4.};
//            c_s={0.32682365,1./2.,5./12.};
//            c_p={0.00609817,0.,5./12.};
//            c_t={0.3706701/6.,-1./6./6.,-17./36./6.};
//        }
//    }
//    if (action=="Sym")
//    {
//        if(LO_or_EM==LO)
//        {
//            c_v={1.5240798/4.,-1./3./4.,-125./288./4.};     // Symanzik Action with Landau gauge
//            c_a={0.6999177/4.,-1./3./4.,-125./288./4.};
//            c_s={2.3547298,-1./4.,0.5};
//            c_p={0.70640549,-1./4.,0.5};
//            c_t={0.9724758/6.,-13./36./6.,-161./216./6.};
//        }
//        else if(LO_or_EM==EM)
//        {
//            c_v={0.3997992/4.,1./16./4.,-1./4./4.};         // Wilson Action with Feynman gauge
//            c_a={0.2394365/4.,-3./16./4.,-1./4./4.};
//            c_s={0.32682365,1./2.,5./12.};
//            c_p={0.00609817,0.,5./12.};
//            c_t={0.3706701/6.,-1./6./6.,-17./36./6.};
//        }
//    }
// 
//    vvd_t c={c_s,c_a,c_p,c_v,c_t};
//    
//    return c;
//}

//vd_t Zq_Oa2(const int LO_or_EM, const int imom, vector<p_t> &p_tilde)
//{
//    vd_t c_q(3);
//    
//    int Np;
//    int count=0;
//    
//    for(int igam=1;igam<5;igam++)
//        if(p_tilde[imom][igam%4]!=0.)
//            count++;
//    Np=count;
//    
//    // Coefficients from the Mathematica file 'O(g2a2).nb'
//    
//    // Nf=4 analysis
//    if(action=="Iwa")
//    {
//        if(LO_or_EM==LO)
//        {
//            c_q={0.6202244+1.8490436/(double)Np/*[imom]*/,-0.0748167-0.963033/(double)Np/*[imom]*/,0.0044};      //Iwasaki action
//        }
//        else if(LO_or_EM==EM)
//        {
//            c_q={-0.0112397+2.26296238/(double)Np/*[imom]*/,31./240.-101./120./(double)Np/*[imom]*/,17./120.};	   //Wilson action (QED)
//        }
//    }
//    // Nf=2 analysis
//    if (action=="Sym")
//    {
//        if(LO_or_EM==LO)
//        {
//            c_q={1.14716212+2.07733285/(double)Np/*[imom]*/,-73./360.-157./180./(double)Np/*[imom]*/,7./240.};   //Symanzik action
//        }
//        else if(LO_or_EM==EM)
//        {
//            c_q={-0.0112397+2.26296238/(double)Np/*[imom]*/,31./240.-101./120./(double)Np/*[imom]*/,17./120.};	   //Wilson action (QED)
//        }
//    }
//    
//    return c_q;
//}

//subtraction of O(a^2) effects to bilinears
//double oper_t::subtraction(const int imom, const int ibil, const int LO_or_EM)
//{
//    double sub=0.0;
//    
//    vvd_t c=pr_bil_Oa2(LO_or_EM);  // c[ibil][k]
//    
//    if(LO_or_EM==LO)
//        sub = g2_tilde*(p2_tilde[imom]*(c[ibil][0]+c[ibil][1]*log(p2_tilde[imom]))+c[ibil][2]*p4_tilde[imom]/p2_tilde[imom])/(12.*M_PI*M_PI);
//    else if(LO_or_EM==EM)
//        sub = (p2_tilde[imom]*(c[ibil][0]+c[ibil][1]*log(p2_tilde[imom]))+c[ibil][2]*p4_tilde[imom]/p2_tilde[imom])/(16.*M_PI*M_PI);
//    
//    
////    cout<<"p2t "<<p2_tilde[imom]<<endl;
////    cout<<"p4t "<<p4_tilde[imom]<<endl;
////    cout<<"c "<<c[ibil][0]<<" "<<c[ibil][1]<<" "<<c[ibil][2]<<endl;
////    cout<<"g2t "<<g2_tilde<<endl;
////    cout<<endl;
//    return sub;
//}

//subtraction of O(a^2) effects to Zq
//double oper_t::subtraction_q(const int imom, const int LO_or_EM)
//{
//    double sub=0.0;
//    
//    vd_t c=Zq_Oa2(LO_or_EM,imom,p_tilde);  // c[k]
//    
//    if(LO_or_EM==LO)
//        sub = g2_tilde*(p2_tilde[imom]*(c[0]+c[1]*log(p2_tilde[imom]))+c[2]*p4_tilde[imom]/p2_tilde[imom])/(12.*M_PI*M_PI);
//    else if(LO_or_EM==EM)
//        sub = (p2_tilde[imom]*(c[0]+c[1]*log(p2_tilde[imom]))+c[2]*p4_tilde[imom]/p2_tilde[imom])/(16.*M_PI*M_PI);
//    
//    return sub;
//}

