#include "read_input.hpp"
#include "aliases.hpp"
#include "global.hpp"
#include "operations.hpp"
#include "read.hpp"

#define DEFAULT_STR_VAL ""
#define DEFAULT_INT_VAL -1
#define DEFAULT_DOUBLE_VAL 1.2345

// define global variables
int nconfs, njacks, nr, ntypes, nhits, Nf, Nc, UseSigma1, UseEffMass, nbeta, ntheta;
int clust_size, nbil, combo;
vector<double> beta;
vector<int> nm_Sea;
vector<vector<int>> SeaMasses_label; // SeaMasses_label[Nbeta][NSeaMass]
int L, T;
vector<double> ainv;
int conf_init, conf_step, nm, neq, neq2, nmr, delta_tmin, delta_tmax;
double kappa, mu_sea, plaquette, g2, g2_tilde, LambdaQCD, p2min, thresh;
vector<double> mass_val;
string mom_path, action, path_ensemble, scheme, BC;
vector<string> beta_label;  // beta_label[Nbeta]
vector<string> theta_label;  // theta_label[Ntheta]


char tok[128];

TK_t get_TK_glb(FILE *fin)
{
    //read a token
    int rc=fscanf(fin,"%s",tok);
    if(rc!=1)
    {
        if(feof(fin)) return FEOF_TK;
        else
        {
            fprintf(stderr,"Getting %d while reading token\n",rc);
            exit(FAILED_READ);
        }
    }
    
    //parse the token
    if(strcasecmp(tok,nconfs_tag)==0) return NCONFS_TK;
    if(strcasecmp(tok,njacks_tag)==0) return NJACKS_TK;
    if(strcasecmp(tok,act_tag)==0) return ACT_TK;
    if(strcasecmp(tok,path_ensemble_tag)==0) return PATH_TK;
    if(strcasecmp(tok,nbeta_tag)==0) return NBETA_TK;
    if(strcasecmp(tok,beta_tag)==0) return BETA_TK;
    if(strcasecmp(tok,beta_label_tag)==0) return BETA_LAB_TK;
    if(strcasecmp(tok,ntheta_tag)==0) return NTHETA_TK;
    if(strcasecmp(tok,theta_label_tag)==0) return THETA_LAB_TK;
    if(strcasecmp(tok,nm_Sea_tag)==0) return NM_SEA_TK;
    if(strcasecmp(tok,SeaMasses_label_tag)==0) return SEAMASSES_LAB_TK;
    if(strcasecmp(tok,nr_tag)==0) return NR_TK;
    if(strcasecmp(tok,ntypes_tag)==0) return NTYPES_TK;
    if(strcasecmp(tok,nhits_tag)==0) return NHITS_TK;
    if(strcasecmp(tok,UseSigma1_tag)==0) return USE_SIGMA_TK;
    if(strcasecmp(tok,UseEffMass_tag)==0) return USE_EFF_MASS_TK;
    if(strcasecmp(tok,scheme_tag)==0) return SCHEME_TK;
    if(strcasecmp(tok,Nc_tag)==0) return NC_TK;
    if(strcasecmp(tok,Nf_tag)==0) return NF_TK;
    if(strcasecmp(tok,ainv_tag)==0) return AINV_TK;
    if(strcasecmp(tok,LambdaQCD_tag)==0) return LAMBDAQCD_TK;
    if(strcasecmp(tok,BC_tag)==0) return BC_TK;
    if(strcasecmp(tok,p2min_tag)==0) return P2MIN_TK;
    if(strcasecmp(tok,thresh_tag)==0) return THRESH_TK;
    
    return VALUE_TK;
}

TK_t get_TK(FILE *fin)
{
    //read a token
    int rc=fscanf(fin,"%s",tok);
    if(rc!=1)
    {
        if(feof(fin)) return FEOF_TK;
        else
        {
            fprintf(stderr,"Getting %d while reading token\n",rc);
            exit(FAILED_READ);
        }
    }
    
    //parse the token
    if(strcasecmp(tok,mom_list_tag)==0) return MOM_LIST_TK;
    if(strcasecmp(tok,L_tag)==0) return L_TK;
    if(strcasecmp(tok,T_tag)==0) return T_TK;
    if(strcasecmp(tok,conf_init_tag)==0) return CONF_INIT_TK;
    if(strcasecmp(tok,conf_step_tag)==0) return CONF_STEP_TK;
    if(strcasecmp(tok,mu_sea_tag)==0) return MU_SEA_TK;
    if(strcasecmp(tok,kappa_tag)==0) return KAPPA_TK;
    if(strcasecmp(tok,nm_tag)==0) return NM_VAL_TK;
    if(strcasecmp(tok,mass_val_tag)==0) return VALMASSES_TK;
    if(strcasecmp(tok,plaquette_tag)==0) return PLAQ_TK;
    if(strcasecmp(tok,delta_tmin_tag)==0) return DELTA_TMIN_TK;
    if(strcasecmp(tok,delta_tmax_tag)==0) return DELTA_TMAX_TK;
    
    return VALUE_TK;
}


//parse the value string (glb input)
template <class T>
void get_value_glb(FILE *fin,T &ret,const char *t)
{
    TK_t tk=get_TK_glb(fin);
    if(tk!=VALUE_TK)
    {
        fprintf(stderr,"Getting token %s in the wrong place\n",tok);
        exit(MISPLACED_TK);
    }
    
    int rc=sscanf(tok,t,&ret);
    
    if(rc!=1)
    {
        fprintf(stderr,"Converting %s to %s failed\n",tok,t);
        exit(FAILED_CONVERSION);
    }
}

//parse the value string
template <class T>
void get_value(FILE *fin,T &ret,const char *t)
{
    TK_t tk=get_TK(fin);
    if(tk!=VALUE_TK)
    {
        fprintf(stderr,"Getting token %s in the wrong place\n",tok);
        exit(MISPLACED_TK);
    }
    
    int rc=sscanf(tok,t,&ret);
    
    if(rc!=1)
    {
        fprintf(stderr,"Converting %s to %s failed\n",tok,t);
        exit(FAILED_CONVERSION);
    }
}

//check
void check_str_par(const char *str,const char *name)
{
    if(strcasecmp(str,DEFAULT_STR_VAL)==0)
    {
        fprintf(stderr,"%s not initialized\n",name);
        exit(UNINITIALIZED_PAR);
    }
}

void check_int_par(const int val,const char *name)
{
    if(val==DEFAULT_INT_VAL)
    {
        fprintf(stderr,"%s not initialized\n",name);
        exit(UNINITIALIZED_PAR);
    }
}

void check_double_par(const double val,const char *name)
{
    if(val==DEFAULT_DOUBLE_VAL)
    {
        fprintf(stderr,"%s not initialized\n",name);
        exit(UNINITIALIZED_PAR);
    }
}

//factorial
int fact(int n)
{
    if(n > 1)
        return n * fact(n - 1);
    else
        return 1;
}

// reads the input file
void read_input_glb(const char path[])
{
    FILE *fin=fopen(path,"r");
    if(not fin)
    {
        fprintf(stderr,"Failed to open \"%s\"\n",path);
        exit(FAILED_OPEN);
    }
    
    printf("READING:\n");
    
    action=DEFAULT_STR_VAL;
    scheme=DEFAULT_STR_VAL;
    path_ensemble=DEFAULT_STR_VAL;
    BC=DEFAULT_STR_VAL;
    nconfs=DEFAULT_INT_VAL;
    njacks=DEFAULT_INT_VAL;
    nr=DEFAULT_INT_VAL;
    ntypes=DEFAULT_INT_VAL;
    nhits=DEFAULT_INT_VAL;
    Nc=DEFAULT_INT_VAL;
    Nf=DEFAULT_INT_VAL;
    UseSigma1=DEFAULT_INT_VAL;
    UseEffMass=DEFAULT_INT_VAL;
    LambdaQCD=DEFAULT_DOUBLE_VAL;
    p2min=DEFAULT_DOUBLE_VAL;
    thresh=DEFAULT_DOUBLE_VAL;
    
    printf("A\n");
    
//    for(auto &bl : beta_label) bl=DEFAULT_STR_VAL;
//    //        for(auto &l : L) l=DEFAULT_INT_VAL;
//    //        for(auto &t : T) t=DEFAULT_INT_VAL;
//    printf(" read C");
//    for(auto &m : nm_Sea) m=DEFAULT_INT_VAL;
//    for(auto &a : SeaMasses_label)
//        for(auto &b : a) b=DEFAULT_INT_VAL;
//    for(auto &t : theta_label) t=DEFAULT_STR_VAL;
//    for(auto &v : ainv) v=DEFAULT_DOUBLE_VAL;
    
    while(not feof(fin))
    {
        TK_t tk=get_TK_glb(fin);
        switch(tk)
        {
            case VALUE_TK:
                fprintf(stderr,"Invalid token %s found\n",tok);
                exit(1);
                break;
            case NCONFS_TK:
                get_value_glb(fin,nconfs,"%d");
                printf(" read 1 ");
                break;
            case NJACKS_TK:
                get_value_glb(fin,njacks,"%d");
                printf(" read 2 ");
                break;
            case BC_TK:
                printf(" read 3 ");
                get_value_glb(fin,BC,"%s");
                printf(" read 3 ");
                break;
            case NBETA_TK:
                get_value_glb(fin,nbeta,"%d");
                printf(" read 4 ");
                break;
            case BETA_TK:
                beta.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,beta[b],"%lf");
                printf(" read 5");
                break;
            case BETA_LAB_TK:
                beta_label.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,beta_label[b],"%s");
                break;
            case NM_SEA_TK:
                nm_Sea.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,nm_Sea[b],"%d");
                break;
            case SEAMASSES_LAB_TK:
                SeaMasses_label.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                {
                    SeaMasses_label[b].resize(nm_Sea[b]);
                    for(int m=0;m<nm_Sea[b];m++)
                        get_value_glb(fin,SeaMasses_label[b][m],"%d");
                }
                break;
            case NTHETA_TK:
                get_value_glb(fin,ntheta,"%d");
                break;
            case THETA_LAB_TK:
                theta_label.resize(ntheta);
                for(int t=0;t<ntheta;t++)
                    get_value_glb(fin,theta_label[t],"%s");
                break;
            case ACT_TK:
                get_value_glb(fin,action,"%s");
                break;
            case PATH_TK:
                get_value_glb(fin,path_ensemble,"%s");
                break;
            case NR_TK:
                get_value_glb(fin,nr,"%d");
                break;
            case NTYPES_TK:
                get_value_glb(fin,ntypes,"%d");
                break;
            case NHITS_TK:
                get_value_glb(fin,nhits,"%d");
                break;
            case USE_SIGMA_TK:
                get_value_glb(fin,UseSigma1,"%d");
                break;
            case USE_EFF_MASS_TK:
                get_value_glb(fin,UseEffMass,"%d");
                break;
            case SCHEME_TK:
                printf(" read 6 ");
                get_value_glb(fin,scheme,"%s");
                printf(" read 6 ");
                break;
            case NC_TK:
                get_value_glb(fin,Nc,"%d");
                break;
            case NF_TK:
                get_value_glb(fin,Nf,"%d");
                break;
            case AINV_TK:
                ainv.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,ainv[b],"%lf");
                break;
            case LAMBDAQCD_TK:
                get_value_glb(fin,LambdaQCD,"%lf");
                break;
            case P2MIN_TK:
                get_value_glb(fin,p2min,"%lf");
                break;
            case THRESH_TK:
                get_value_glb(fin,thresh,"%lf");
                break;
                
            case FEOF_TK:
                break;
        }
    }
    
    //check initialization
    check_int_par(nconfs,nconfs_tag);
    check_int_par(njacks,njacks_tag);
    //        for(auto &l : L) check_int_par(l,L_tag);
    //        for(auto &t : L) check_int_par(t,T_tag);
    check_str_par(path_ensemble.c_str(),path_ensemble_tag);
    check_str_par(scheme.c_str(),scheme_tag);
    check_str_par(BC.c_str(),BC_tag);
    check_str_par(action.c_str(),act_tag);
    check_int_par(Nc,Nc_tag);
    check_int_par(Nf,Nf_tag);
    check_int_par(nr,nr_tag);
    check_int_par(ntypes,ntypes_tag);
    check_int_par(nhits,nhits_tag);
    check_int_par(UseSigma1,UseSigma1_tag);
    check_int_par(UseEffMass,UseEffMass_tag);
    for(auto &b : beta) check_double_par(b,beta_tag);
    for(auto &bl : beta_label) check_str_par(bl.c_str(),beta_label_tag);
    for(auto &a : ainv) check_double_par(a,ainv_tag);
    for(auto &ms : nm_Sea) check_int_par(ms,nm_Sea_tag);
    for(auto &im : SeaMasses_label)
        for(auto &jm : im) check_int_par(jm,SeaMasses_label_tag);
    check_int_par(ntheta,ntheta_tag);
    for(auto &t : theta_label) check_str_par(t.c_str(),theta_label_tag);
    check_double_par(LambdaQCD,LambdaQCD_tag);
    check_double_par(p2min,p2min_tag);
    check_double_par(thresh,thresh_tag);
    
    fclose(fin);
    
    //print input parameters
    
    printf(".------------------------------------------------------.\n");
    printf("|                Global configuration                  |\n");
    printf(".------------------------------------------------------.\n\n");
    
    printf(" %s = %s\n",scheme_tag,scheme.c_str());
    printf("    with BC: %s \n\n",BC.c_str());
    
    printf(" %s = %.2lf\n",thresh_tag,thresh);
    printf(" Continuum limit range: a2p2 > %.1lf\n\n",p2min);
    
    printf(" %s = %s  --  %s = %d  -- %s = %d \n",act_tag,action.c_str(),Nf_tag,Nf,Nc_tag,Nc);
    printf(" %s = %d  (%d njacks) \n",nconfs_tag,nconfs,njacks);
    printf(" %s = %s \n\n",path_ensemble_tag,path_ensemble.c_str());
    
    printf("%s = %d\n",nr_tag,nr);
    printf("%s = %d\n",ntypes_tag,ntypes);
    printf("%s = %d\n\n",nhits_tag,nhits);
    
    printf("%s = %.3lf\n",LambdaQCD_tag,LambdaQCD);
    
    printf(" Working with %d beta: \n",nbeta);
    for(int b=0;b<nbeta;b++)
    {
        printf("   beta = %.2lf : ainv=%.2lf\n",beta[b],ainv[b]);
        printf("                  Ensembles: ");
        for(int m=0; m<nm_Sea[b]; m++)
            printf("%s%d, ",beta_label[b].c_str(),SeaMasses_label[b][m]);
        printf("\n");
        
    }
    
    
    // define global variables from input
    clust_size=nconfs/njacks;
    
    nbil=5;
    
    
    //    //factors enabling average over r
    //    int c1, c2;
    //    if(nr==2)
    //    {
    //        c1 = 1;
    //        c2 = 1;
    //    }
    
    //g2_tilde
    //        g2=6.0/beta;
    //        g2_tilde=g2/plaquette;
    //        printf("g2tilde = %lf\n",g2_tilde);
    
    //slightly increment thresh to include border
    thresh*=1+1e-10;
    
    printf("\n");
}

void read_input(const char path[])
{
    FILE *fin=fopen(path,"r");
    if(not fin)
    {
        fprintf(stderr,"Failed to open \"%s\"\n",path);
        exit(FAILED_OPEN);
    }
    
    
    char mom_list_path[128]=DEFAULT_STR_VAL;
    
    int L=DEFAULT_INT_VAL;
    int T=DEFAULT_INT_VAL;
    
    conf_init=DEFAULT_INT_VAL;
    conf_step=DEFAULT_INT_VAL;
    nm=DEFAULT_INT_VAL;
    delta_tmin=DEFAULT_INT_VAL;
    delta_tmax=DEFAULT_INT_VAL;
    //        kappa=DEFAULT_DOUBLE_VAL;
    mu_sea=DEFAULT_DOUBLE_VAL;
    plaquette=DEFAULT_DOUBLE_VAL;
    for(auto &m : mass_val) m=DEFAULT_DOUBLE_VAL;
    
    while(not feof(fin))
    {
        TK_t tk=get_TK(fin);
        switch(tk)
        {
            case VALUE_TK:
                fprintf(stderr,"Invalid token %s found\n",tok);
                exit(1);
                break;
            case MOM_LIST_TK:
                get_value(fin,mom_list_path,"%s");
                break;
            case L_TK:
                get_value(fin,L,"%d");
                break;
            case T_TK:
                get_value(fin,T,"%d");
                break;
            case CONF_INIT_TK:
                get_value(fin,conf_init,"%d");
                break;
            case CONF_STEP_TK:
                get_value(fin,conf_step,"%d");
                break;
            case PLAQ_TK:
                get_value(fin,plaquette,"%lf");
                break;
                //                case KAPPA_TK:
                //                    get_value(fin,kappa,"%lf");
                //                    break;
            case MU_SEA_TK:
                get_value(fin,mu_sea,"%lf");
                break;
                
            case NM_VAL_TK:
                get_value(fin,nm,"%d");
                break;
            case VALMASSES_TK:
                mass_val.resize(nm);
                for (int i=0; i<nm; i++)
                    get_value(fin,mass_val[i],"%lf");
                break;
                
            case DELTA_TMIN_TK:
                get_value(fin,delta_tmin,"%d");
                break;
            case DELTA_TMAX_TK:
                get_value(fin,delta_tmax,"%d");
                break;
                
                
                
            case FEOF_TK:
                break;
        }
    }
    
    check_str_par(mom_list_path,mom_list_tag);
    check_int_par(L,L_tag);
    check_int_par(T,T_tag);
    check_double_par(plaquette,plaquette_tag);
    check_double_par(kappa,kappa_tag);
    check_int_par(conf_init,conf_init_tag);
    check_int_par(conf_step,conf_step_tag);
    check_int_par(nm,nm_tag);
    for (int i=0; i<nm; i++)
        check_double_par(mass_val[i],mass_val_tag);
    check_int_par(delta_tmin,delta_tmin_tag);
    check_int_par(delta_tmax,delta_tmax_tag);
    
    fclose(fin);
    
//    if(plaquette==0.0)  plaquette=read_plaquette();
    
    mom_path = string(mom_list_path);
    
    coords_t size={T,L,L,L};
    
    printf(" %s = \"%s\"\n",mom_list_tag,mom_list_path);
    printf("%s = %d\n",L_tag,L);
    printf("%s = %d\n",T_tag,T);
    printf("%s = %lf\n",plaquette_tag,plaquette);
    
    
    printf(" %s = %d  [from %d to %d]\n",nconfs_tag,nconfs,conf_init,conf_init+(nconfs-1)*conf_step);
    
    printf("Fit range for deltam_cr: [%d,%d]\n",delta_tmin,delta_tmax);
    
    printf("%s = %.6lf\n",kappa_tag,kappa);
    printf("%s = %.4lf\n",mu_sea_tag,mu_sea);
    printf("%s = %d\n",nm_tag,nm);
    printf("%s = ",mass_val_tag);
    for (int i=0; i<nm; i++)
        printf("%.4lf  ",mass_val[i]);
    printf("\n");
    
    
    nmr=nm*nr;
    combo=nm*nr*ntypes*nhits*nconfs;
    neq=fact(nm+nr-1)/fact(nr)/fact(nm-1);
    neq2=nm;
    
}
