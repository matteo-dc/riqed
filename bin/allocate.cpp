#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"

void oper_t::allocate()
{
    vector<vector<vvd_t>> sig={sigma1_LO,sigma1_PH,sigma1_P,sigma1_S,
                               sigma2_LO,sigma2_PH,sigma2_P,sigma2_S,
                               sigma3_LO,sigma3_PH,sigma3_P,sigma3_S};
    
    vector<vector<vvd_t>>           zq={jZq,jZq_EM};
    vector<vector<jproj_t>>         gbil={jG_LO,jG_EM};
    vector<vector<jZbil_t>>         zbil={jZ,jZ_EM};
    vector<vector<jproj_t>>         gbil4f={jG_LO_4f,jG_EM_4f};
    vector<vector<jZ4f_t>>          z4f={jZ_4f,jZ_EM_4f};
    vector<vector<jproj_meslep_t>>  meslep={jpr_meslep_LO,jpr_meslep_EM,jpr_meslep_nasty};
    
    for(auto &s : sig)
    {
        s.resize(_linmoms);
        for(auto &ijack : s)
        {
            ijack.resize(njacks);
            for(auto &mr : ijack)
                mr.resize(_nmr);
        }
    }
    
    cout<<"sigma1_LO: "<<sigma1_LO.size();
    cout<<" "<<sigma1_LO[0].size()<<" "<<sigma1_LO[0][0].size()<<endl;
    cout<<"sigma1_PH: "<<sigma1_PH.size()<<" "<<sigma1_PH[0].size()<<" "<<sigma1_PH[0][0].size()<<endl;
    cout<<"sigma1_P: "<<sigma1_P.size()<<" "<<sigma1_P[0].size()<<" "<<sigma1_P[0][0].size()<<endl;
    cout<<"sigma1_S: "<<sigma1_S.size()<<" "<<sigma1_S[0].size()<<" "<<sigma1_S[0][0].size()<<endl;
    
    for(auto &z : zq)
    {
        z.resize(_linmoms);
        for(auto &ijack : z)
        {
            ijack.resize(njacks);
            for(auto &mr : ijack)
                mr.resize(_nmr);
        }
    }
    
    for(auto &g : gbil)
    {
        g.resize(_bilmoms);
        for(auto &ibil : g)
        {
            ibil.resize(nbil);
            for(auto &ijack : ibil)
            {
                ijack.resize(njacks);
                for(auto &mr1 : ijack)
                {
                    mr1.resize(_nmr);
                    for(auto &mr2 : mr1)
                        mr2.resize(_nmr);
                }
            }
        }
    }
    
    for(auto &z : zbil)
    {
        z.resize(_bilmoms);
        for(auto &ibil : z)
        {
            ibil.resize(nbil);
            for(auto &ijack : ibil)
            {
                ijack.resize(njacks);
                for(auto &mr1 : ijack)
                {
                    mr1.resize(_nmr);
                    for(auto &mr2 : mr1)
                        mr2.resize(_nmr);
                }
            }
        }
    }
    
    for(auto &g : gbil4f)
    {
        g.resize(_bilmoms);
        for(auto &ibil : g)
        {
            ibil.resize(nbil+1);
            for(auto &ijack : ibil)
            {
                ijack.resize(njacks);
                for(auto &mr1 : ijack)
                {
                    mr1.resize(_nmr);
                    for(auto &mr2 : mr1)
                        mr2.resize(_nmr);
                }
            }
        }
        
    }
    
    for(auto &m : meslep)
    {
        m.resize(_meslepmoms);
        for(auto &iop1 : m)
        {
            iop1.resize(nbil);
            for(auto &iop2 : iop1)
            {
                iop2.resize(nbil);
                for(auto &ijack : iop2)
                {
                    ijack.resize(njacks);
                    for(auto &mr1 : ijack)
                    {
                        mr1.resize(_nmr);
                        for(auto &mr2 : mr1)
                            mr2.resize(_nmr);
                    }
                }
            }
        }
    }
    
    for(auto &z : z4f)
    {
        z.resize(_meslepmoms);
        for(auto &iop1 : z)
        {
            iop1.resize(nbil);
            for(auto &iop2 : iop1)
            {
                iop2.resize(nbil);
                for(auto &ijack : iop2)
                {
                    ijack.resize(njacks);
                    for(auto &mr1 : ijack)
                    {
                        mr1.resize(_nmr);
                        for(auto &mr2 : mr1)
                            mr2.resize(_nmr);
                    }
                    
                }
            }
        }
    }
    
}
