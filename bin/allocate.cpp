#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"

void oper_t::allocate()
{
    jZq.resize(_linmoms);
    jZq_em.resize(_linmoms);
    
    jG_0.resize(_bilmoms);
    jG_em.resize(_bilmoms);
    
    jZ.resize(_bilmoms);
    jZ_em.resize(_bilmoms);
    
    jG_0_4f.resize(_bilmoms);
    jG_em_4f.resize(_bilmoms);
    
    jpr_meslep_0.resize(_meslepmoms);
    jpr_meslep_em.resize(_meslepmoms);
    jpr_meslep_nasty.resize(_meslepmoms);
    
    jZ_4f.resize(_meslepmoms);
    jZ_em_4f.resize(_meslepmoms);
    
    for(auto &ijack : jZq)
    {
        ijack.resize(njacks);
        for(auto &mr : ijack)
            mr.resize(_nmr);
    }
    
    for(auto &ijack : jZq_em)
    {
        ijack.resize(njacks);
        for(auto &mr : ijack)
            mr.resize(_nmr);
    }
    
    
    for(auto &ibil : jG_0)
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
    
    for(auto &ibil : jG_em)
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
    
    for(auto &ibil : jZ)
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
    
    for(auto &ibil : jZ_em)
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
    
    for(auto &ibil : jG_0_4f)
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
    
    for(auto &ibil : jG_em_4f)
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
    
    for(auto &iop1 : jpr_meslep_0)
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
    
    for(auto &iop1 : jpr_meslep_em)
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
    
    for(auto &iop1 : jpr_meslep_nasty)
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
    
    for(auto &iop1 : jZ_4f)
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
    
    for(auto &iop1 : jZ_em_4f)
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
