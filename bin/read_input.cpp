#include "read_input.hpp"
#include "aliases.hpp"
#include "global.hpp"
#include "operations.hpp"
#include "read.hpp"

#define DEFAULT_STR_VAL "null"
#define DEFAULT_INT_VAL -1
#define DEFAULT_DOUBLE_VAL 1.2345

// define global variables
int nconfs, njacks, nr, ntypes, nhits, Nf, Nc, UseSigma1, UseEffMass, nbeta, ntheta, compute_4f,only_basic, compute_mpcac, load_ave, load_chir, load, QCD_on_the_right;
int clust_size, nbil, combo, combo_lep, ntypes_lep;
vector<double> beta;
vector<int> nm_Sea;
int nm_Sea_max;
vector<vector<int>> SeaMasses_label; // SeaMasses_label[Nbeta][NSeaMass]
int L, T;
vector<double> ainv;
int conf_init, conf_step, nm, neq, neq2, nmr, delta_tmin, delta_tmax;
double kappa, mu_sea, plaquette, LambdaQCD, p2min, p2max, thresh, p2ref;
vector<double> mass_val;
string mom_path, action, path_folder, scheme, BC, out_hadr, out_lep, analysis, path_ensemble, an_suffix;
vector<string> path_analysis;
vector<string> beta_label;  // beta_label[Nbeta]
vector<string> theta_label;  // theta_label[Ntheta]
bool free_analysis;
bool inte_analysis;
bool eta_analysis;
bool recompute_basic;

coords_t size;

char tok[128];

TK_glb_t get_TK_glb(FILE *fin)
{
    //read a token
    int rc=fscanf(fin,"%s",tok);
    if(rc!=1)
    {
        if(feof(fin)) return FEOF_GLB_TK;
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
    if(strcasecmp(tok,path_folder_tag)==0) return PATH_TK;
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
    if(strcasecmp(tok,p2max_tag)==0) return P2MAX_TK;
    if(strcasecmp(tok,thresh_tag)==0) return THRESH_TK;
    if(strcasecmp(tok,compute_meslep_tag)==0) return COMPUTE_MESLEP_TK;
    if(strcasecmp(tok,out_hadr_tag)==0) return OUT_HADR_TK;
    if(strcasecmp(tok,out_lep_tag)==0) return OUT_LEP_TK;
    if(strcasecmp(tok,only_basic_tag)==0) return ONLY_BASIC_TK;
    if(strcasecmp(tok,compute_mpcac_tag)==0) return COMPUTE_MPCAC_TK;
    if(strcasecmp(tok,analysis_tag)==0) return ANALYSIS_TK;
    if(strcasecmp(tok,p2ref_tag)==0) return P2REF_TK;
    if(strcasecmp(tok,load_ave_tag)==0) return LOAD_AVE_TK;
    if(strcasecmp(tok,load_chir_tag)==0) return LOAD_CHIR_TK;
    if(strcasecmp(tok,an_suffix_tag)==0) return SUFFIX_TK;
    if(strcasecmp(tok,QCD_on_the_right_tag)==0) return QCDONTHERIGHT_TK;

    return VALUE_GLB_TK;
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
void _get_value_glb(FILE *fin,T &ret,const char *t)
{
    TK_glb_t tk=get_TK_glb(fin);
    if(tk!=VALUE_GLB_TK)
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

void get_value_glb(FILE *fin,double &out)
{
    return _get_value_glb(fin,out,"%lg");
}

void get_value_glb(FILE *fin,int &out)
{
    return _get_value_glb(fin,out,"%d");
}

void get_value_glb(FILE *fin,string &out)
{
    char temp[1024];
    _get_value_glb(fin,temp,"%s");
    out=string(temp);

}

//parse the value string
template <class T>
void _get_value(FILE *fin,T &ret,const char *t)
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

void get_value(FILE *fin,double &out)
{
    return _get_value(fin,out,"%lg");
}

void get_value(FILE *fin,int &out)
{
    return _get_value(fin,out,"%d");
}

void get_value(FILE *fin,string &out)
{
    char temp[1024];
    _get_value(fin,temp,"%s");
    out=string(temp);
    
}


//check
void check_str_par(const string str,const char *name)
{
    if(str.compare(DEFAULT_STR_VAL)==0)
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
        
    action=DEFAULT_STR_VAL;
    scheme=DEFAULT_STR_VAL;
    path_folder=DEFAULT_STR_VAL;
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
    p2max=DEFAULT_DOUBLE_VAL;
    thresh=DEFAULT_DOUBLE_VAL;
    compute_4f=DEFAULT_INT_VAL;
    out_hadr=DEFAULT_STR_VAL;
    out_lep=DEFAULT_STR_VAL;
    only_basic=DEFAULT_INT_VAL;
    compute_mpcac=DEFAULT_INT_VAL;
    analysis=DEFAULT_STR_VAL;
    p2ref=DEFAULT_DOUBLE_VAL;
    load_ave=DEFAULT_INT_VAL;
    load_chir=DEFAULT_INT_VAL;
    an_suffix=DEFAULT_STR_VAL;
    QCD_on_the_right=DEFAULT_INT_VAL;

    
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
        TK_glb_t tk=get_TK_glb(fin);
        switch(tk)
        {
            case VALUE_GLB_TK:
                fprintf(stderr,"Invalid token %s found\n",tok);
                exit(1);
                break;
            case NCONFS_TK:
                get_value_glb(fin,nconfs);
                break;
            case NJACKS_TK:
                get_value_glb(fin,njacks);
                break;
            case BC_TK:
                get_value_glb(fin,BC);
                break;
            case NBETA_TK:
                get_value_glb(fin,nbeta);
                break;
            case BETA_TK:
                beta.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,beta[b]);
                break;
            case BETA_LAB_TK:
                beta_label.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,beta_label[b]);
                break;
            case NM_SEA_TK:
                nm_Sea.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,nm_Sea[b]);
                break;
            case SEAMASSES_LAB_TK:
                SeaMasses_label.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                {
                    SeaMasses_label[b].resize(nm_Sea[b]);
                    for(int m=0;m<nm_Sea[b];m++)
                        get_value_glb(fin,SeaMasses_label[b][m]);
                }
                break;
            case NTHETA_TK:
                get_value_glb(fin,ntheta);
                break;
            case THETA_LAB_TK:
                theta_label.resize(ntheta);
                for(int t=0;t<ntheta;t++)
                    get_value_glb(fin,theta_label[t]);
                break;
            case ACT_TK:
                get_value_glb(fin,action);
                break;
            case PATH_TK:
                get_value_glb(fin,path_folder);
                break;
            case NR_TK:
                get_value_glb(fin,nr);
                break;
            case NTYPES_TK:
                get_value_glb(fin,ntypes);
                break;
            case NHITS_TK:
                get_value_glb(fin,nhits);
                break;
            case USE_SIGMA_TK:
                get_value_glb(fin,UseSigma1);
                break;
            case USE_EFF_MASS_TK:
                get_value_glb(fin,UseEffMass);
                break;
            case SCHEME_TK:
                get_value_glb(fin,scheme);
                break;
            case NC_TK:
                get_value_glb(fin,Nc);
                break;
            case NF_TK:
                get_value_glb(fin,Nf);
                break;
            case AINV_TK:
                ainv.resize(nbeta);
                for(int b=0;b<nbeta;b++)
                    get_value_glb(fin,ainv[b]);
                break;
            case LAMBDAQCD_TK:
                get_value_glb(fin,LambdaQCD);
                break;
            case P2MIN_TK:
                get_value_glb(fin,p2min);
                break;
            case P2MAX_TK:
                get_value_glb(fin,p2max);
                break;
            case THRESH_TK:
                get_value_glb(fin,thresh);
                break;
            case COMPUTE_MESLEP_TK:
                get_value_glb(fin,compute_4f);
                break;
            case OUT_HADR_TK:
                get_value_glb(fin,out_hadr);
                break;
            case OUT_LEP_TK:
                get_value_glb(fin,out_lep);
                break;
            case ONLY_BASIC_TK:
                get_value_glb(fin,only_basic);
                break;
            case COMPUTE_MPCAC_TK:
                get_value_glb(fin,compute_mpcac);
                break;
            case ANALYSIS_TK:
                get_value_glb(fin,analysis);
                break;
            case P2REF_TK:
                get_value(fin,p2ref);
                break;
            case LOAD_AVE_TK:
                get_value(fin,load_ave);
                break;
            case LOAD_CHIR_TK:
                get_value(fin,load_chir);
                break;
            case SUFFIX_TK:
                get_value(fin,an_suffix);
                break;
            case QCDONTHERIGHT_TK:
                get_value(fin,QCD_on_the_right);
                break;
                
            case FEOF_GLB_TK:
                break;
        }
    }
    
    //check initialization
    check_int_par(nconfs,nconfs_tag);
    check_int_par(njacks,njacks_tag);
    check_str_par(path_folder,path_folder_tag);
    check_str_par(scheme,scheme_tag);
    check_str_par(BC,BC_tag);
    check_str_par(action,act_tag);
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
    check_double_par(p2max,p2max_tag);
    check_double_par(thresh,thresh_tag);
    check_int_par(compute_4f,compute_meslep_tag);
    check_str_par(out_hadr,out_hadr_tag);
    check_str_par(out_lep,out_lep_tag);
    check_int_par(only_basic,only_basic_tag);
    check_int_par(compute_mpcac,compute_mpcac_tag);
    check_str_par(analysis,analysis_tag);
    check_double_par(p2ref,p2ref_tag);
    check_int_par(load_ave,load_ave_tag);
    check_int_par(load_chir,load_chir_tag);
    check_str_par(an_suffix,an_suffix_tag);
    check_int_par(QCD_on_the_right,QCD_on_the_right_tag);

    fclose(fin);
    
    free_analysis=false;
    inte_analysis=false;
    eta_analysis=false;
    
    if(strcmp(analysis.c_str(),"inte" )==0)
    {
        path_analysis={"Nf4"+an_suffix};
        
        inte_analysis=true;
    }
    else if(strcmp(analysis.c_str(),"free" )==0)
    {
        path_analysis={"free_matching"+an_suffix};
        
        free_analysis=true;
    }
    else if(strcmp(analysis.c_str(),"eta")==0)
    {
        path_analysis={"Rat"+an_suffix,"Nf4"+an_suffix,"free_matching"+an_suffix};
        
        eta_analysis=true;
    }
    else {cout<<"Choose the analysis: 'inte', 'free' or 'eta'."<<endl; exit(0);}
    
    if(strcmp(analysis.c_str(),"free")==0 and nr>1)
    {
        cout<<"Nr must be 1 in free theory. Setting Nr=1 instead of Nr="<<nr<<"."<<endl;
        nr=1;
    }
    
    load = (load_ave or load_chir);
    
    if(load and only_basic)
    {
        cout<<"Cannot load saved quantities in the only_basic mode."<<endl;
        exit(0);
    }
    
    // this is the path to the directory which contains 'print', 'plots', ecc.
    string full_path = path_folder+path_analysis[0]+"/";
    
    // evaluate max number of sea masses for the ensembles
    nm_Sea_max = *max_element(nm_Sea.begin(),nm_Sea.end());
    
    //print input parameters
    printf("*------------------------------------------------------*\n");
    printf("|                Global configuration                  |\n");
    printf("*------------------------------------------------------*\n\n");
    
    printf(" %s = %s\n\n",analysis_tag,analysis.c_str());  //free, inte, ratio
    
    printf(" %s = %s\n",scheme_tag,scheme.c_str());
    printf("    with BC: %s \n\n",BC.c_str());
    
    printf(" %s = %.2lf\n",thresh_tag,thresh);
    printf(" Continuum limit range: a2p2 = [%.1lf,%.1lf]\n\n",p2min,p2max);
    
    printf(" %s = %s  --  %s = %d  -- %s = %d -- %s = %.3lf \n",act_tag,action.c_str(),Nf_tag,Nf,Nc_tag,Nc,LambdaQCD_tag,LambdaQCD);
    printf(" %s = %d  (%d njacks) \n",nconfs_tag,nconfs,njacks);
    printf(" %s = %s \n\n",path_folder_tag,full_path.c_str());
    
    printf(" %s = %d\n",nr_tag,nr);
    printf(" %s = %d\n",ntypes_tag,ntypes);
    printf(" %s = %d\n\n",nhits_tag,nhits);
    
    printf(" Working with %d beta: \n",nbeta);
    for(int b=0;b<nbeta;b++)
    {
        printf("    beta = %.2lf : ainv = %.2lf\n",beta[b],ainv[b]);
        printf("                  Ensembles: ");
        for(int m=0; m<nm_Sea[b]; m++)
            printf("%s%d ",beta_label[b].c_str(),SeaMasses_label[b][m]);
        printf("\n");
    }
    
    printf(" Using Z^{QCD} factorized on the ");
    if(QCD_on_the_right) printf("RIGHT. \n");
    else printf("LEFT. \n");
    
    if(only_basic)
        printf(" Computing only basic quantities. \n");
    
    if(load_ave)
        printf(" Loading averaged quantities. \n");
    
    // define global variables from input
    clust_size=nconfs/njacks;
    
    nbil=5;
    
    //slightly increment thresh to include border
    thresh*=1+1e-10;
    
    printf("\n\n");
}

void read_input(const string &path_to_ens, const string &name)
{
    string path_to_input = path_to_ens + "input.txt";
    
    FILE *fin=fopen(path_to_input.c_str(),"r");
    if(not fin)
    {
        fprintf(stderr,"Failed to open \"%s\"\n",path_to_input.c_str());
        exit(FAILED_OPEN);
    }
    
    mom_path=DEFAULT_STR_VAL;
    
    int L=DEFAULT_INT_VAL;
    int T=DEFAULT_INT_VAL;
    
    conf_init=DEFAULT_INT_VAL;
    conf_step=DEFAULT_INT_VAL;
    nm=DEFAULT_INT_VAL;
    delta_tmin=DEFAULT_INT_VAL;
    delta_tmax=DEFAULT_INT_VAL;
    kappa=DEFAULT_DOUBLE_VAL;
    mu_sea=DEFAULT_DOUBLE_VAL;
    plaquette=DEFAULT_DOUBLE_VAL;
//    for(auto &m : mass_val) m=DEFAULT_DOUBLE_VAL;
    
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
                get_value(fin,mom_path);
                break;
            case L_TK:
                get_value(fin,L);
                break;
            case T_TK:
                get_value(fin,T);
                break;
            case CONF_INIT_TK:
                get_value(fin,conf_init);
                break;
            case CONF_STEP_TK:
                get_value(fin,conf_step);
                break;
            case PLAQ_TK:
                get_value(fin,plaquette);
                break;
            case KAPPA_TK:
                get_value(fin,kappa);
                break;
            case MU_SEA_TK:
                get_value(fin,mu_sea);
                break;
            case NM_VAL_TK:
                get_value(fin,nm);
                break;
            case VALMASSES_TK:
                mass_val.resize(nm);
                for (int i=0; i<nm; i++)
                    get_value(fin,mass_val[i]);
                break;
            case DELTA_TMIN_TK:
                get_value(fin,delta_tmin);
                break;
            case DELTA_TMAX_TK:
                get_value(fin,delta_tmax);
                break;
                
            case FEOF_TK:
                break;
        }
    }
    
    check_str_par(mom_path.c_str(),mom_list_tag);
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
    
    if(plaquette==0.0)  plaquette=read_plaquette(path_to_ens);
    
    size={T,L,L,L};
    
    printf("*------------------------------------------------------*\n");
    printf("|                     Ensemble %s                     |\n",name.c_str());
    printf("*------------------------------------------------------*\n\n");
    
    
    printf(" %s = \"%s\"\n",mom_list_tag,mom_path.c_str());
    printf(" Dimensions = %dc%d\n",L,T);
    printf(" %s = %lf -- %s = %.6lf -- %s = %.4lf \n",plaquette_tag,plaquette,kappa_tag,kappa,mu_sea_tag,mu_sea);
    
    
    printf(" %s = %d  [from %d to %d]\n",nconfs_tag,nconfs,conf_init,conf_init+(nconfs-1)*conf_step);
    
    printf(" Fit range for deltam_cr: [%d,%d]\n",delta_tmin,delta_tmax);
    
    printf(" %s = ",mass_val_tag);
    for (int i=0; i<nm; i++)
        printf("%.4lf  ",mass_val[i]);
    printf("\n\n");
    
    
    ntypes_lep = 2;
    
    nmr=nm*nr;
    combo=nm*nr*ntypes*nhits*nconfs;
    combo_lep=ntypes_lep*nhits*nconfs;
    neq=fact(nm+nr-1)/fact(nr)/fact(nm-1);
    neq2=nm;
    
}
