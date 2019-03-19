#ifndef INPUT_HPP
#define INPUT_HPP

#include "global.hpp"
#include "aliases.hpp"

enum ERR_t{NO_FAIL,FAILED_READ,FAILED_CONVERSION,FAILED_OPEN,MISPLACED_TK,UNINITIALIZED_PAR};

enum TK_glb_t{FEOF_GLB_TK,VALUE_GLB_TK,NCONFS_TK,NJACKS_TK,ACT_TK,PATH_TK,NBETA_TK,BETA_TK,BETA_LAB_TK,NTHETA_TK,THETA_LAB_TK,NM_SEA_TK,SEAMASSES_LAB_TK,NR_TK,NTYPES_TK,NHITS_TK,USE_SIGMA_TK,USE_EFF_MASS_TK,SCHEME_TK,NF_TK,NC_TK,AINV_TK,LAMBDAQCD_TK,BC_TK,P2MIN_TK,P2MAX_TK,THRESH_TK,COMPUTE_MESLEP_TK,OUT_HADR_TK,OUT_LEP_TK,ONLY_BASIC_TK,COMPUTE_MPCAC_TK,ANALYSIS_TK,P2REF_TK,LOAD_AVE_TK,LOAD_CHIR_TK,SUFFIX_TK,QCDONTHERIGHT_TK};

enum TK_t{FEOF_TK,VALUE_TK,MOM_LIST_TK,L_TK,T_TK,CONF_INIT_TK,CONF_STEP_TK,PLAQ_TK,MU_SEA_TK,KAPPA_TK,NM_VAL_TK,VALMASSES_TK,DELTA_TMIN_TK,DELTA_TMAX_TK};

const char nconfs_tag[]="NConfs";
const char mom_list_tag[]="MomList";
const char njacks_tag[]="NJacks";
const char L_tag[]="L";
const char T_tag[]="T";
const char conf_init_tag[]="ConfInit";
const char conf_step_tag[]="ConfStep";
const char act_tag[]="Action";
const char path_folder_tag[]="PathToEnsembles";
const char mu_sea_tag[]="MuSea";
const char nbeta_tag[]="NBeta";
const char beta_tag[]="Beta";
const char beta_label_tag[]="Beta_label";
const char kappa_tag[]="Kappa";
const char nm_Sea_tag[]="NSeaMasses";
const char SeaMasses_label_tag[]="SeaMasses_label";
const char ntheta_tag[]="NTheta";
const char theta_label_tag[]="Theta_label";
const char nm_tag[]="NValMasses";
const char mass_val_tag[]="ValMasses";
const char nr_tag[]="Nr";
const char plaquette_tag[]="Plaquette";
const char ntypes_tag[]="NTypes";
const char nhits_tag[]="NHits";
const char UseSigma1_tag[]="UseSigma1";
const char UseEffMass_tag[]="UseEffMass";
const char scheme_tag[]="Scheme";
const char Nc_tag[]="Nc";
const char Nf_tag[]="Nf";
const char ainv_tag[]="ainv";
const char LambdaQCD_tag[]="LambdaQCD";
const char delta_tmin_tag[]="tmin(deltam_cr)";
const char delta_tmax_tag[]="tmax(deltam_cr)";
const char BC_tag[]="BC";
const char p2min_tag[]="p2min";
const char p2max_tag[]="p2max";
const char thresh_tag[]="Thresh";
const char compute_meslep_tag[]="ComputeMesLep";
const char out_hadr_tag[]="OutHadrons";
const char out_lep_tag[]="OutLeptons";
const char only_basic_tag[]="OnlyBasic";
const char compute_mpcac_tag[]="ComputeMPCAC";
const char analysis_tag[]="Analysis";
const char p2ref_tag[]="RefP2";
const char load_ave_tag[]="LoadAveRave";
const char load_chir_tag[]="LoadChir";
const char an_suffix_tag[]="AnSuffix";
const char QCD_on_the_right_tag[]="QCDOnTheRight";

// parse the value string
template <class T>
void get_value(FILE *fin,T &ret,const char *t);

// check
void check_str_par(const string str,const char *name);
void check_int_par(const int val,const char *name);
void check_double_par(const double val,const char *name);

// read global input file
void read_input_glb(const char input_path[]);

// read input file relative to single ensembles
void read_input(const string &input, const string &name);

#endif
