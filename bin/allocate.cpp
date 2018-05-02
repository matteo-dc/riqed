#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"

void oper_t::allocate()
{
    vector<vector<vvd_t>>           zq={jZq,jZq_EM};
    vector<vector<jproj_t>>         gbil={jG_LO,jG_EM};
    vector<vector<jZbil_t>>         zbil={jZ,jZ_EM};
    vector<vector<jproj_t>>         gbil4f={jG_LO_4f,jG_EM_4f};
    vector<vector<jZ4f_t>>          z4f={jZ_4f,jZ_EM_4f};
    vector<vector<jproj_meslep_t>>  meslep={jpr_meslep_LO,jpr_meslep_EM,jpr_meslep_nasty};
    
    // defining sigma allocation
#define ALLOCATE_SIGMA(NUM,KIND)                        \
    sigma ## NUM ## _ ## KIND.resize(_linmoms);         \
    for(auto &ijack : sigma ## NUM ## _ ## KIND)        \
    {                                                   \
        ijack.resize(njacks);                           \
        for(auto &mr : ijack)                           \
        {                                               \
            mr.resize(_nmr);                            \
            for(auto &i : mr)                           \
                i=0.0;                                  \
        }                                               \
    }
    // defining Zq allocation
#define ALLOCATE_ZQ(A)              \
    A.resize(_linmoms);             \
    for(auto &ijack : A)            \
    {                               \
        ijack.resize(njacks);       \
        for(auto &mr : ijack)       \
        {                           \
            mr.resize(_nmr);        \
            for(auto &i : mr)       \
                i=0.0;              \
        }                           \
    }
    
    // defining Gbil allocation
#define ALLOCATE_GBIL(A)                \
    A.resize(_bilmoms);                 \
    for(auto &ibil : A)                 \
    {                                   \
        ibil.resize(nbil);              \
        for(auto &ijack : ibil)         \
        {                               \
            ijack.resize(njacks);       \
            for(auto &mr1 : ijack)      \
            {                           \
                mr1.resize(_nmr);       \
                for(auto &mr2 : mr1)    \
                {                       \
                    mr2.resize(_nmr);   \
                    for(auto &i : mr2)  \
                        i=0.0;          \
                }                       \
            }                           \
        }                               \
    }
    
    // defining Zbil allocation
#define ALLOCATE_ZBIL(A)    \
    ALLOCATE_GBIL(A)
    
    // defining Gbil4f allocation
#define ALLOCATE_GBIL4f(A)              \
    A.resize(_bilmoms);                 \
    for(auto &ibil : A)                 \
    {                                   \
        ibil.resize(nbil+1);            \
        for(auto &ijack : ibil)         \
        {                               \
            ijack.resize(njacks);       \
            for(auto &mr1 : ijack)      \
            {                           \
                mr1.resize(_nmr);       \
                for(auto &mr2 : mr1)    \
                {                       \
                    mr2.resize(_nmr);   \
                    for(auto &i : mr2)  \
                        i=0.0;          \
                }                       \
            }                           \
        }                               \
    }
    // defining meslep allocation
#define ALLOCATE_MESLEP(A)                  \
    A.resize(_meslepmoms);                  \
    for(auto &iop1 : A)                     \
    {                                       \
        iop1.resize(nbil);                  \
        for(auto &iop2 : iop1)              \
        {                                   \
            iop2.resize(nbil);              \
            for(auto &ijack : iop2)         \
            {                               \
                ijack.resize(njacks);       \
                for(auto &mr1 : ijack)      \
                {                           \
                    mr1.resize(_nmr);       \
                    for(auto &mr2 : mr1)    \
                    {                       \
                        mr2.resize(_nmr);   \
                        for(auto &i : mr2)  \
                            i=0.0;          \
                    }                       \
                }                           \
            }                               \
        }                                   \
    }
    // defining Z4f allocation
#define ALLOCATE_Z4f(A)                     \
    A.resize(_meslepmoms);                  \
    for(auto &iop1 : A)                     \
    {                                       \
        iop1.resize(nbil);                  \
        for(auto &iop2 : iop1)              \
        {                                   \
            iop2.resize(nbil);              \
            for(auto &ijack : iop2)         \
            {                               \
                ijack.resize(njacks);       \
                for(auto &mr1 : ijack)      \
                {                           \
                    mr1.resize(_nmr);       \
                    for(auto &mr2 : mr1)    \
                    {                       \
                        mr2.resize(_nmr);   \
                        for(auto &i : mr2)  \
                            i=0.0;          \
                    }                       \
                }                           \
            }                               \
        }                                   \
    }

    // allocation
    
    ALLOCATE_SIGMA(1,LO);
    ALLOCATE_SIGMA(1,PH);
    ALLOCATE_SIGMA(1,P);
    ALLOCATE_SIGMA(1,S);
    ALLOCATE_SIGMA(2,LO);
    ALLOCATE_SIGMA(2,PH);
    ALLOCATE_SIGMA(2,P);
    ALLOCATE_SIGMA(2,S);
    ALLOCATE_SIGMA(3,LO);
    ALLOCATE_SIGMA(3,PH);
    ALLOCATE_SIGMA(3,P);
    ALLOCATE_SIGMA(3,S);

    ALLOCATE_ZQ(jZq);
    ALLOCATE_ZQ(jZq_EM);
    
    ALLOCATE_GBIL(jG_LO);
    ALLOCATE_GBIL(jG_EM);
    
    ALLOCATE_ZBIL(jZ);
    ALLOCATE_ZBIL(jZ_EM);
    
    ALLOCATE_GBIL4f(jG_LO_4f);
    ALLOCATE_GBIL4f(jG_EM_4f);
    
    ALLOCATE_MESLEP(jpr_meslep_LO);
    ALLOCATE_MESLEP(jpr_meslep_EM);
    ALLOCATE_MESLEP(jpr_meslep_nasty);
    
    ALLOCATE_Z4f(jZ_4f);
    ALLOCATE_Z4f(jZ_EM_4f);
    
#undef ALLOCATE_SIGMA
#undef ALLOCATE_ZQ
#undef ALLOCATE_GBIL
#undef ALLOCATE_ZBIL
#undef ALLOCATE_GBIL4f
#undef ALLOCATE_MESLEP
#undef ALLOCATE_Z4f
}
