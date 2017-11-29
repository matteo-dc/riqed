#ifdef HAVE_CONFIG_H
#include <config.hpp>
#endif

#include <complex>
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <valarray>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <sstream>
#include <iomanip>

#define lambdaQCD 0.250
#define Z3 1.2020569031595942
#define Nc 3.0

using namespace std;
using namespace Eigen;
using namespace std::chrono;

//coordinates in the lattice
using coords_t=array<int,4>;

//complex double
using dcompl=complex<double>;

//propagator (12X12)
using prop_t=Matrix<dcompl,12,12>;

//list of propagators
using vprop_t=valarray<prop_t>;
using vvprop_t=valarray< valarray<prop_t> >;
using vvvprop_t=valarray< valarray< valarray<prop_t> > >;

//list of gamma for a given momentum
using qline_t=valarray<prop_t>;
using vqline_t=valarray<qline_t>;
using vvqline_t=valarray<vqline_t>;
using vert_t = vvqline_t;

//list of jackknife propagators
using jprop_t=valarray< valarray<prop_t> >;

//list of jackknife vertices
using jvert_t=valarray< vert_t >;

//valarray of complex double
using vd_t=valarray<double>;

//valarray of valarray of complex double
using vvd_t=valarray< valarray<double> > ;

//valarray of valarray of valarray of complex double
using vvvd_t=valarray< valarray< valarray<double> > >;
using vvvvd_t=valarray<vvvd_t>;
using vvvvvd_t=valarray<vvvvd_t>;

//valarray of complex double
using vdcompl_t=valarray<dcompl>;
using vvdcompl_t=valarray< vdcompl_t >;
using vvvdcompl_t=valarray< vvdcompl_t >;
using vvvvdcompl_t=valarray< vvvdcompl_t >;

//valarray of Eigen Vectors
using vXd_t=valarray<VectorXd>;

//useful notation
using jZ_t=vvd_t;
using jZbil_t=vvvvd_t;
using jproj_t=vvvvd_t;

//list of momenta
vector<coords_t> mom_list;

//list of N(p)
vector<int> Np;

int nr,nm,nmr;


//factorial
int fact(int n)
{
    if(n > 1)
        return n * fact(n - 1);
    else
        return 1;
}

//to string with precision
template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
    ostringstream out;
    out << fixed;
    out << setprecision(n) << a_value;
    return out.str();
}

//read a file
void read_mom_list(const string &path)
{
    //file
    ifstream input(path);
    if(!input.good())
    {
        cerr<<"Error opening "<<path<<endl;
        exit(1);
    }
    
    //loop until end of file
    while(!input.eof())
    {
        coords_t c;
        for(int mu=0;mu<4;mu++) input>>c[mu];
        if(input.good()) mom_list.push_back(c);
    }
}

//create the path-string to the configuration
string path_to_conf(const string &string_path, int i_conf,const string &name)
{
    char path[1024];
    sprintf(path,"%sout/%04d/fft_%s",string_path.c_str(),i_conf,name.c_str());
    //sprintf(path,"/marconi_work/INF17_lqcd123_0/sanfo/RIQED/3.90_24_0.0100/out/%04d/fft_%s",i_conf,name.c_str());
    // sprintf(path,"out/%04d/fft_%s",i_conf,name.c_str());
    return path;
}

//returns the linearized spin color index
size_t isc(size_t is,size_t ic)
{return ic+3*is;}

//read a propagator file
prop_t read_prop(ifstream &input, const string &path)
{
    prop_t out(prop_t::Zero());
    
    for(int id_so=0;id_so<4;id_so++)
        for(int ic_so=0;ic_so<3;ic_so++)
            for(int id_si=0;id_si<4;id_si++)
                for(int ic_si=0;ic_si<3;ic_si++)
                {
                    double temp[2];
                    if(not input.good())
                    {
                        cerr<<"Bad before reading"<<endl;
                        exit(1);
                    }
                    input.read((char*)&temp,sizeof(double)*2);
                    if(not input.good())
                    {
                        cerr<<"Unable to read from "<<path<<" id_so: "<<id_so<<", ic_so: "<<ic_so<<", id_si: "<<id_si<<", ic_si:"<<ic_si<<endl;
                        exit(1);
                    }
                    out(isc(id_si,ic_si),isc(id_so,ic_so))=dcompl(temp[0],temp[1]); //store
                }
    
    return out;
}

//create Dirac Gamma matrices
vprop_t make_gamma()
{
    int NGamma=16;
    
    vprop_t gam(NGamma);
    vector< vector<int> > col(NGamma, vector<int>(4));
    vector< vector<int> > real_part(NGamma, vector<int>(4));
    vector< vector<int> > im_part(NGamma, vector<int>(4));
    
    //Identity=gamma0
    col[0]={0,1,2,3};
    real_part[0]={1,1,1,1};
    im_part[0]={0,0,0,0};
    //gamma1
    col[1]={3,2,1,0};
    real_part[1]={0,0,0,0};
    im_part[1]={-1,-1,1,1};
    //gamma2
    col[2]={3,2,1,0};
    real_part[2]={-1,1,1,-1};
    im_part[2]={0,0,0,0};
    //gamma3
    col[3]={2,3,0,1};
    real_part[3]={0,0,0,0};
    im_part[3]={-1,1,1,-1};
    //gamma4
    col[4]={2,3,0,1};
    real_part[4]={-1,-1,-1,-1};
    im_part[4]={0,0,0,0};
    //gamma5
    col[5]={0,1,2,3};
    real_part[5]={1,1,-1,-1};
    im_part[5]={0,0,0,0};
    
    for(int i_gam=0;i_gam<6;i_gam++)
        for(int i_row=0;i_row<4;i_row++)
            for(int ic=0;ic<3;ic++)
            {
                gam[i_gam](isc(i_row,ic),isc(col[i_gam][i_row],ic))=dcompl(real_part[i_gam][i_row],im_part[i_gam][i_row] );
            }
    
    //gamma_mu*gamma5
    for(int j=0;j<4;j++)
    {
        gam[6+j]=gam[1+j]*gam[5];
    }
    //sigma
    size_t ind1[6]={2,3,1,4,4,4};
    size_t ind2[6]={3,1,2,1,2,3};
    for(int i=0;i<6;i++)
        gam[10+i]=0.5*(gam[ind1[i]]*gam[ind2[i]]-gam[ind2[i]]*gam[ind1[i]]);
    
    return gam;
}

//calculate the vertex function in a given configuration for the given equal momenta
prop_t make_vertex(const prop_t &prop1, const prop_t &prop2, const int mu, const vprop_t &gamma)
{
    
    prop_t vert=prop1*gamma[mu]*gamma[5]*prop2.adjoint()*gamma[5];  /*it has to be "jackknifed"*/
    
    return vert;
}

//create the path-string to the contraction
string path_to_contr(int i_conf,const int mr1, const string &T1, const int mr2, const string &T2, const string &string_path)
{
    
    int r1 = mr1%nr;
    int m1 = (mr1-r1)/nr;
    int r2 = mr2%nr;
    int m2 = (mr2-r2)/nr;
    
    char path[1024];
    sprintf(path,"%sout/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",string_path.c_str(),i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());
    //sprintf(path,"/marconi_work/INF17_lqcd123_0/sanfo/RIQED/3.90_24_0.0100/out/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());
    //sprintf(path,"out/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());
    
    // cout<<path<<endl;
    
    return path;
}

//jackknife double
vvd_t jackknife_double(vvd_t &jd, int size, int nconf, int clust_size )
{
    vd_t jSum(0.0,size);
    
    //sum of jd
    for(size_t j=0;j<jd.size();j++) jSum+= jd[j];
    //jackknife fluctuation
    for(size_t j=0;j<jd.size();j++)
    {
        jd[j]=jSum-jd[j];
        for(auto &it : jd[j])
            it/=(nconf-clust_size);
    }
    
    return jd;
}

//jackknife Propagator
jprop_t jackknife_prop(jprop_t &jS, const int nconf, const int clust_size, const size_t nhits)
{
    int nmr = jS[0].size();
    int nj = jS.size();
    valarray<prop_t> jSum(prop_t::Zero(),nmr);
    
    //sum of jS
    
    for(int j=0;j<nj;j++)
    {
#pragma omp parallel for
        for(int mr=0;mr<nmr;mr++)
            jSum[mr]+= jS[j][mr];
    }
    
    //jackknife fluctuation
#pragma omp parallel for collapse(2)
    for(int j=0;j<nj;j++)
        for(int mr=0;mr<nmr;mr++)
            jS[j][mr]=(jSum[mr]-jS[j][mr])/((nconf-clust_size)/*/nhits*/);
    
    return jS;
}

//jackknife Vertex
jvert_t jackknife_vertex(jvert_t &jVert, const int nconf, const int clust_size, const size_t nhits)
{
    int nmr = jVert[0].size();
    vert_t jSum(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr);
    
    //sum of the jVert
    for(size_t j=0;j<jVert.size();j++)
    {
#pragma omp parallel for collapse(3)
        for(int mrA=0;mrA<nmr;mrA++)
            for(int mrB=0;mrB<nmr;mrB++)
                for(int igam=0;igam<16;igam++)
                    jSum[mrA][mrB][igam] += jVert[j][mrA][mrB][igam];
    }
    //jackknife fluctuation
#pragma omp parallel for collapse(4)
    for(size_t j=0;j<jVert.size();j++)
        for(int mrA=0;mrA<nmr;mrA++)
            for(int mrB=0;mrB<nmr;mrB++)
                for(int igam=0;igam<16;igam++)
                    jVert[j][mrA][mrB][igam] = (jSum[mrA][mrB][igam]-jVert[j][mrA][mrB][igam])/((nconf-clust_size)/**nhits*/);
    
    return jVert;
}

//invert the propagator
jprop_t invert_jprop( const jprop_t &jprop){
    
    int njacks=jprop.size();
    int nmr=jprop[0].size();
    
    jprop_t jprop_inv(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
    
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
            jprop_inv[ijack][mr]=jprop[ijack][mr].inverse();
    
    return jprop_inv;
}

//amputate external legs

prop_t amputate( const prop_t  &prop1_inv, const prop_t &V, const prop_t  &prop2_inv, vprop_t GAMMA){
    
    prop_t Lambda=prop1_inv*V*GAMMA[5]*prop2_inv.adjoint()*GAMMA[5];
    
    return Lambda;
}

//compute jZq
vvd_t compute_jZq(vprop_t GAMMA, jprop_t jS_inv, double L, double T, int imom)
{
    double V=L*L*L*T;
    int njacks=jS_inv.size();
    int nmr=jS_inv[0].size();
    
    //compute p_slash as a vector of prop-type matrices
    vd_t p(0.0,4);
    vd_t p_tilde(0.0,4);
    prop_t p_slash(prop_t::Zero());
    double p2_not_tilde=0.0 , p2=0.0;
    
    // vvdcompl_t jZq(vdcompl_t(0.0,nmr),njacks);
    vvd_t jZq_real(vd_t(0.0,nmr),njacks);
    dcompl I(0.0,1.0);
    
    int count=0;
    
    p={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
    p_tilde={sin(p[0]),sin(p[1]),sin(p[2]),sin(p[3])};
    
    for(int igam=1;igam<5;igam++)
    {
        p_slash+=GAMMA[igam]*p_tilde[igam-1];
        
        if(p_tilde[igam-1]!=0.)
            count++;
    }
    
    Np.push_back(count);
    
    /*  Note that: p_slash*p_slash=p2*GAMMA[0]  */
    
    //compute p^2
    for(int coord=0;coord<4;coord++)
        p2_not_tilde+=p[coord]*p[coord];
    for(int coord=0;coord<4;coord++)
        p2+=p_tilde[coord]*p_tilde[coord];
    
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
            jZq_real[ijack][mr]=-(I*(p_slash*jS_inv[ijack][mr]).trace()).real()/p2/12./V;
    
    return jZq_real;
    
}

//compute jSigma1
vvd_t compute_jSigma1(vprop_t GAMMA, jprop_t jS_inv, double L, double T, int imom)
{
    double V=L*L*L*T;
    int njacks=jS_inv.size();
    int nmr=jS_inv[0].size();
    
    //compute p_slash as a vector of prop-type matrices
    vd_t p(0.0,4);
    vd_t p_tilde(0.0,4);
    prop_t p_slash(prop_t::Zero());
    
    // vvdcompl_t jSigma1(vdcompl_t(0.0,nmr),njacks);
    vvd_t jSigma1_real(vd_t(0.0,nmr),njacks);
    dcompl I(0.0,1.0);
    
    vvprop_t A(vprop_t(prop_t::Zero(),nmr),njacks);
    
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
        {
            int count=0;
            
            p={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
            p_tilde={sin(p[0]),sin(p[1]),sin(p[2]),sin(p[3])};
            
            for(int igam=1;igam<5;igam++)
                if(p_tilde[igam-1]!=0.)
                {
                    A[ijack][mr]+=GAMMA[igam]*jS_inv[ijack][mr]/p_tilde[igam-1];
                    count++;
                }
            A[ijack][mr]/=(double)count;
            jSigma1_real[ijack][mr]=-(I*A[ijack][mr].trace()).real()/12./V;
        }
    
    return jSigma1_real;
}


//project the amputated green function
jproj_t project(vprop_t GAMMA, const jvert_t &jLambda)
{
    const int njacks=jLambda.size();
    const int nmr=jLambda[0].size();
    
    //L_proj has 5 components: S(0), V(1), P(2), A(3), T(4)
    jvert_t L_proj(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),5),nmr),nmr),njacks);
    jproj_t jG_real(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
    vprop_t P(prop_t::Zero(),16);
    
    //create projectors such that tr(GAMMA*P)=Identity
    P[0]=GAMMA[0]; //scalar
    for(int igam=1;igam<5;igam++)  //vector
        P[igam]=GAMMA[igam].adjoint()/4.;
    P[5]=GAMMA[5];  //pseudoscalar
    for(int igam=6;igam<10;igam++)  //axial
        P[igam]=GAMMA[igam].adjoint()/4.;
    for(int igam=10;igam<16;igam++)  //tensor
        P[igam]=GAMMA[igam].adjoint()/6.;
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            {
                L_proj[ijack][mr_fw][mr_bw][0]=jLambda[ijack][mr_fw][mr_bw][0]*P[0];
                for(int igam=1;igam<5;igam++)
                    L_proj[ijack][mr_fw][mr_bw][1]+=jLambda[ijack][mr_fw][mr_bw][igam]*P[igam];
                L_proj[ijack][mr_fw][mr_bw][2]=jLambda[ijack][mr_fw][mr_bw][5]*P[5];
                for(int igam=6;igam<10;igam++)
                    L_proj[ijack][mr_fw][mr_bw][3]+=jLambda[ijack][mr_fw][mr_bw][igam]*P[igam];
                for(int igam=10;igam<16;igam++)
                    L_proj[ijack][mr_fw][mr_bw][4]+=jLambda[ijack][mr_fw][mr_bw][igam]*P[igam];
                
                for(int j=0;j<5;j++)
                    jG_real[ijack][mr_fw][mr_bw][j]=L_proj[ijack][mr_fw][mr_bw][j].trace().real()/12.0;
            }
    
    return jG_real;
}

//subtraction of O(a^2) effects
double subtract(vector<double> c, double f, double p2, double p4, double g2_tilde)
{
    double f_new;
    
    f_new = f - g2_tilde*(p2*(c[0]+c[1]*log(p2))+c[2]*p4/p2)/(12.*M_PI*M_PI);
    
    return f_new;
}

valarray<VectorXd> fit_par_jackknife(const vvd_t &coord, vd_t &error, const vvd_t &y, const int range_min, const int range_max)
{
    int n_par = coord.size();
    int njacks = y.size();
    
    MatrixXd S(n_par,n_par);
    valarray<VectorXd> Sy(VectorXd(n_par),njacks);
    valarray<VectorXd> jpars(VectorXd(n_par),njacks);
    
    //initialization
    S=MatrixXd::Zero(n_par,n_par);
    for(int ijack=0; ijack<njacks; ijack++)
    {
        Sy[ijack]=VectorXd::Zero(n_par);
        jpars[ijack]=VectorXd::Zero(n_par);
    }
    
    //definition
    for(int i=range_min; i<=range_max; i++)
    {
        if(error[i]<1.0e-20) error[i]+=1.0e-20;
        
        for(int j=0; j<n_par; j++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
        
        for(int ijack=0; ijack<njacks; ijack++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]);
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
    
    for(int i=range_min; i<=range_max; i++)
        cout<<"(x,y) [ijack=0] = "<<coord[1][i]<<" "<<y[0][i]<<" "<<error[i]<<endl;
    cout<<"Extrapolation: "<<jpars[0](0)<<endl;
    
    return jpars;
    
}

//print of file
void print_internal(double t,ofstream& outfile)
{
    outfile.write((char*) &t,sizeof(double));
}
//template <class T>
void print_internal(VectorXd &V, ofstream& outfile)
{
    for(int i=0; i<V.size();i++) print_internal(V(i),outfile);
}
template <class T>
void print_internal(valarray<T> &v, ofstream& outfile)
{
    for(auto &i : v) print_internal(i,outfile);
}
template <class T>
void print_vec( T &vec, const char* path)
{
    ofstream outfile(path,ofstream::binary);
    
    if (outfile.is_open())
    {
        for(auto &i : vec)
            print_internal(i,outfile);
        
        outfile.close();
        
    }
    else cout << "Unable to open the output file "<<path<<endl;
}

vvvd_t average_Zq(vector<jZ_t> &jZq)
{
    int moms=jZq.size();
    int njacks=jZq[0].size();
    int nmr=jZq[0][0].size();
    
    vvd_t Zq_ave(vd_t(0.0,nmr),moms), sqr_Zq_ave(vd_t(0.0,nmr),moms), Zq_err(vd_t(0.0,nmr),moms);
    vvvd_t Zq_ave_err(vvd_t(vd_t(0.0,nmr),moms),2);
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int mr=0;mr<nmr;mr++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                Zq_ave[imom][mr]+=jZq[imom][ijack][mr]/njacks;
                sqr_Zq_ave[imom][mr]+=jZq[imom][ijack][mr]*jZq[imom][ijack][mr]/njacks;
            }
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int mr=0;mr<nmr;mr++)
        {
            Zq_ave_err[0][imom][mr]=Zq_ave[imom][mr];
            Zq_ave_err[1][imom][mr]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[imom][mr]-Zq_ave[imom][mr]*Zq_ave[imom][mr]));
        }
    
    return Zq_ave_err;
}

vvd_t average_Zq_chiral(vector<vd_t> &jZq)
{
    int moms=jZq.size();
    int njacks=jZq[0].size();
    
    vd_t Zq_ave(0.0,moms), sqr_Zq_ave(0.0,moms), Zq_err(0.0,moms);
    vvd_t Zq_ave_err(vd_t(0.0,moms),2);
    
    for(int imom=0;imom<moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            Zq_ave[imom]+=jZq[imom][ijack]/njacks;
            sqr_Zq_ave[imom]+=jZq[imom][ijack]*jZq[imom][ijack]/njacks;
        }
    for(int imom=0;imom<moms;imom++)
    {
        Zq_ave_err[0][imom]=Zq_ave[imom];
        Zq_ave_err[1][imom]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[imom]-Zq_ave[imom]*Zq_ave[imom]));
    }
    
    return Zq_ave_err;
}

vvvvvd_t average_Z(vector<jZbil_t> &jZ)
{
    int moms=jZ.size();
    int njacks=jZ[0].size();
    int nmr=jZ[0][0].size();
    
    vvvvd_t Z_ave(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms), sqr_Z_ave(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms), Z_err(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms);
    vvvvvd_t Z_ave_err(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms),2);
    
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<moms;imom++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int k=0;k<5;k++)
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
                        Z_ave[imom][mr_fw][mr_bw][k]+=jZ[imom][ijack][mr_fw][mr_bw][k]/njacks;
                        sqr_Z_ave[imom][mr_fw][mr_bw][k]+=jZ[imom][ijack][mr_fw][mr_bw][k]*jZ[imom][ijack][mr_fw][mr_bw][k]/njacks;
                    }
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<moms;imom++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int k=0;k<5;k++)
                {
                    Z_ave_err[0][imom][mr_fw][mr_bw][k]=Z_ave[imom][mr_fw][mr_bw][k];
                    Z_ave_err[1][imom][mr_fw][mr_bw][k]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_ave[imom][mr_fw][mr_bw][k]-Z_ave[imom][mr_fw][mr_bw][k]*Z_ave[imom][mr_fw][mr_bw][k]));
                }
    
    return Z_ave_err;
}

double alphas(int Nf,double ainv,double mu2)
{
    
    int CF;
    CF = (Nc*Nc-1.)/(2.*Nc);
    
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
    
    lam0=lambdaQCD/ainv;
    
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
double q_evolution_to_RIp_ainv(int Nf,double ainv,double mu_2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=a2*p2 (adimensional quantity)
    //mu0_2=a2*(1/a^2)=1
    double mu0_2=1;
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,ainv,mu_2)/(4*M_PI);
    al0=alphas(Nf,ainv,mu0_2)/(4*M_PI);
    
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
double S_evolution_to_RIp_ainv(int Nf,double ainv,double mu_2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=a2*p2 (adimensional quantity)
    //mu0_2=a2*(1/a^2)=1
    double mu0_2=1;
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,ainv,mu_2)/(4*M_PI);
    al0=alphas(Nf,ainv,mu0_2)/(4*M_PI);
    
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

double P_evolution_to_RIp_ainv(int Nf,double ainv,double mu_2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=a2*p2 (adimensional quantity)
    //mu0_2=a2*(1/a^2)=1
    double mu0_2=1;
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,ainv,mu_2)/(4*M_PI);
    al0=alphas(Nf,ainv,mu0_2)/(4*M_PI);
    
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
double T_evolution_to_RIp_ainv(int Nf,double ainv,double mu_2)
{
    double cmu=0.0, cmu0=0.0; // c=cmu/cmu0
    //mu_2=a2*p2 (adimensional quantity)
    //mu0_2=a2*(1/a^2)=1
    double mu0_2=1;
    
    // alphas @ NNLO
    double alm, al0;
    alm=alphas(Nf,ainv,mu_2)/(4*M_PI);
    al0=alphas(Nf,ainv,mu0_2)/(4*M_PI);
    
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





/***********************************************************************************/
/*************************************** main **************************************/
/***********************************************************************************/


int main(int narg,char **arg)
{
#pragma omp parallel
#pragma omp master
    system("clear");
    cout<<"Using "<<omp_get_num_threads()<<" threads"<<endl;
    
    high_resolution_clock::time_point t_START=high_resolution_clock::now();
    
    high_resolution_clock::time_point t0=high_resolution_clock::now();
    
    if (narg!=14){
//        cerr<<"Number of arguments not valid: <mom file> <nconfs> <njacks> <L> <T> <initial conf_id> <step conf_id> <p2fit min> <p2fit max> <action=sym/iwa/free> <path before ensemble directory: /marconi_work/.../ > <beta> <sea mass> <c1> <c2>"<<endl;
        cerr<<"Number of arguments not valid: <mom file> <nconfs> <njacks> <L> <T> <initial conf_id> <step conf_id> <action=sym/iwa/free> <path before ensemble directory: /marconi_work/.../ > <beta> <sea mass> <c1> <c2>"<<endl;
        exit(0);
    }
  
    cout<<endl<<endl;
    
    int nconfs=stoi(arg[2]);
    int njacks=stoi(arg[3]);
    int clust_size=nconfs/njacks;
    int conf_id[nconfs];
    
    double L=0.0, T=0.0;
    
    //Check on L and T
    if((strcmp(arg[4],"24")==0 && strcmp(arg[5],"48")==0) || (strcmp(arg[4],"32")==0 && strcmp(arg[5],"64")==0))
    {
        L=stod(arg[4]);
        T=stod(arg[5]);
    }
    else
    {
        cerr<<"WARNING: wrong L and T arguments. Please write '24 48' or '32 64'.";
        exit(0);
    }
    
    
    size_t nhits=1; //!
    
    nm = 4;  //! to be passed from command line
    nr = 2;
    
    nmr=nm*nr;
    
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=stoi(arg[6])+iconf*stoi(arg[7]);
    
//    double p2fit_min=stod(arg[8]);  //!
//    double p2fit_max=stod(arg[9]);  //!
    
//     const double use_tad = 1.0;  //!
    
    string string_L = arg[4];
    
    double beta=0.0, plaquette=0.0;
    vector<double> c_v(3), c_a(3), c_s(3), c_p(3), c_t(3);
    vector<double> c_v_em(3), c_a_em(3), c_s_em(3), c_p_em(3), c_t_em(3);
    
    string string_action;
    
    //beta & plaquette
    if(strcmp(arg[8],"iwa")==0)  //Nf=4 (Iwasaki)
    {
        beta=1.90;
        plaquette=0.574872;
        
        c_v={0.2881372,-0.2095,-0.516583};
        c_a={0.9637998,-0.2095,-0.516583};
        c_s={2.02123300,-1./4.,0.376167};
        c_p={0.66990790,-1./4.,0.376167};
        c_t={0.3861012,-0.196,-0.814167};
        
        string_action = "Iwasaki";
    }
    else if(strcmp(arg[8],"sym")==0)  //Nf=2 (Symanzik)
    {
        //beta=3.90;
        plaquette=0.582591;
        
        //WARNING: we divide the following coefficients by 4 (or 6 for T) to account for the sum on Lorentz indices!
        
        c_a={1.5240798/4.,-1./3./4.,-125./288./4.};     // Symanzik Action with Landau gauge (from the Mathematica file 'O(g2a2).nb')
        c_v={0.6999177/4.,-1./3./4.,-125./288./4.};
        c_s={2.3547298,-1./4.,0.5};
        c_p={0.70640549,-1./4.,0.5};
        c_t={0.9724758/6.,-13./36./6.,-161./216./6.};
        
        c_a_em={0.3997992/4.,1./16./4.,-1./4./4.};      // Wilson Action with Feynman gauge
        c_v_em={0.2394365/4.,-3./16./4.,-1./4./4.};
        c_s_em={0.32682365,1./2.,5./12.};
        c_p_em={0.00609817,0.,5./12.};
        c_t_em={0.3706701/6.,-1./6./6.,-17./36./6.};
        
        string_action = "Symanzik";
    }
    else
    {
        cerr<<"WARNING: wrong action argument. Please write 'sym' for Symanzik action or 'iwa' for Iwasaki action.";
        exit(0);
    }
    
    string string_path_to_ensemble = arg[9];
    string string_beta;
    string string_sea_mass;
    
    double sea_mass=0.0;
    
    //Check on beta
    if(strcmp(arg[10],"3.80")==0 || strcmp(arg[10],"3.90")==0 || strcmp(arg[10],"4.05")==0)
    {
        beta = stod(arg[10]);
        string_beta = arg[10];
    }
    else
    {
        cerr<<"WARNING: wrong beta argument. Please choice among: 3.80, 3.90 and 4.05.";
        exit(0);
    }
    
    //Check on sea mass
    if(strcmp(arg[10],"3.80")==0 && (strcmp(arg[11],"0.0080")==0 || strcmp(arg[11],"0.0110")==0 || strcmp(arg[11],"0.0165")==0))
    {
        sea_mass = stod(arg[11]);
        string_sea_mass = arg[11];
    }
    else if(strcmp(arg[10],"3.90")==0 && (strcmp(arg[11],"0.0040")==0 || strcmp(arg[11],"0.0064")==0 || strcmp(arg[11],"0.0085")==0 || strcmp(arg[11],"0.0100")==0))
    {
        sea_mass = stod(arg[11]);
        string_sea_mass = arg[11];
    }
    else if(strcmp(arg[10],"4.05")==0 && (strcmp(arg[11],"0.0030")==0 || strcmp(arg[11],"0.0060")==0 || strcmp(arg[11],"0.0080")==0))
    {
        sea_mass = stod(arg[11]);
        string_sea_mass = arg[11];
    }
    else
    {
        if(strcmp(arg[10],"3.80")==0) cerr<<"WARNING: wrong sea mass argument. Please choice among: 0.0080, 0.0100 and 0.0165.";
        if(strcmp(arg[10],"3.90")==0) cerr<<"WARNING: wrong sea mass argument. Please choice among: 0.0040, 0.0064, 0.0085 and 0.0100.";
        if(strcmp(arg[10],"4.05")==0) cerr<<"WARNING: wrong sea mass argument. Please choice among: 0.0030, 0.0060 and 0.0080.";
        
        exit(0);
    }
    
    
    int c1 = stoi(arg[12]);
    int c2 = stoi(arg[13]);
    
    string string_path = string_path_to_ensemble+string_beta+"_"+string_L+"_"+string_sea_mass+"/";
    cout<<string_path<<endl;
    
    //g2_tilde
    double g2=6.0/beta;
    double g2_tilde=g2/plaquette;
    
    cout<<"N confs = "<<nconfs<<endl;
    cout<<"N jacks = "<<njacks<<endl;
    cout<<"Clust size = "<<clust_size<<endl;
    cout<<"L = "<<L<<"\t T = "<<T<<endl;
    //    cout<<"Fit range = ["<<p2fit_min<<":"<<p2fit_max<<"]"<<endl;
    cout<<"Action: "<<string_action<<endl;
    cout<<"Beta = "<<beta<<endl;
    cout<<"Sea mass = "<<sea_mass<<endl;
    cout<<"Plaquette = "<<plaquette<<endl;
    cout<<"g2_tilde = "<<g2_tilde<<endl<<endl;
    
    high_resolution_clock::time_point t1=high_resolution_clock::now();
    duration<double> t_span = duration_cast<duration<double>>(t1-t0);
    cout<<"***** Assigned input values in  "<<t_span.count()<<" s ******"<<endl<<endl;
    
    //Delta m_cr
    
    cout<<"Reading deltam_cr. "<<endl;
    
    t0=high_resolution_clock::now();
    
    vvvd_t deltam_cr(vvd_t(vd_t(0.0,nm),nm),njacks);
    
    ifstream input_deltam;
    input_deltam.open("deltam_cr_array",ios::binary);
    
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
            {
                double temp;
                input_deltam.read((char*)&temp,sizeof(double));
                if(not input_deltam.good())
                {
                    cerr<<"Unable to read from deltam_cr_array mr_fw: "<<m_fw<<", mr_bw: "<<m_bw<<", ijack: "<<ijack<<endl;
                    exit(1);
                }
                deltam_cr[ijack][m_fw][m_bw]=temp; //store
            }
    
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int ijack=0;ijack<njacks;ijack++)
                if(m_bw==m_fw)
                    cout<<"mA "<<m_fw<<" mB "<<m_bw<<" ijack "<<ijack<<"  "<<deltam_cr[ijack][m_fw][m_bw]<<endl;
    
    vvd_t deltam_cr_ave(vd_t(0.0,nm),nm);
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int ijack=0;ijack<njacks;ijack++)
                deltam_cr_ave[m_fw][m_bw] += deltam_cr[ijack][m_fw][m_bw]/njacks;
    
    cout<<endl;
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            if(m_bw==m_fw)
                cout<<"mA "<<m_fw<<" mB "<<m_bw<<"  "<<deltam_cr_ave[m_fw][m_bw]<<endl;
    
    t1=high_resolution_clock::now();
    t_span = duration_cast<duration<double>>(t1-t0);
    cout<<"***** Read Deltam_cr in  "<<t_span.count()<<" s ******"<<endl<<endl;
    
    
    //Quark Masses array
    vd_t mass_array={0.0040,0.0060,0.0080,0.0100};
//    vd_t mass_array={0.0040,0.0064,0.0085,0.0100,0.0150};

    
    //Effective Mass
    
    cout<<"Reading effective mass. "<<endl;
    
    t0=high_resolution_clock::now();
    
    vvvd_t eff_mass_array(vvd_t(vd_t(0.0,2),nmr),nmr);
    
    ifstream input_effmass;
    input_effmass.open("eff_mass_array",ios::binary);
    
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            for(int i=0;i<2;i++)
            {
                double temp;
                input_effmass.read((char*)&temp,sizeof(double));
                if(not input_effmass.good())
                {
                    cerr<<"Unable to read from eff_mass_array mr_fw: "<<mr_fw<<", mr_bw: "<<mr_bw<<", i: "<<i<<endl;
                    exit(1);
                }
                eff_mass_array[mr_fw][mr_bw][i]=temp; //store
            }
    
    vvd_t eff_mass(vd_t(0.0,nmr),nmr);
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            eff_mass[mr_fw][mr_bw] = eff_mass_array[mr_fw][mr_bw][0];
    
    t1=high_resolution_clock::now();
    t_span = duration_cast<duration<double>>(t1-t0);
    cout<<endl;
    cout<<"***** Read Effective Mass in "<<t_span.count()<<" s ******"<<endl<<endl;
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    read_mom_list(arg[1]);
    
    cout<<"Read: "<<mom_list.size()<<" momenta."<<endl<<endl;
    
    //create gamma matrices
    vprop_t GAMMA=make_gamma();
    
    vector<string> Mass={"M0_","M1_","M2_","M3_"};
    vector<string> R={"R0_","R1_"};
    vector<string> Type={"0","F","FF","T","P"/*,"S"*/};
    
    vector<double> p2_vector_allmoms, p4_vector;
    vector<int> tag_vector;
    
    // int nm=Mass.size();
    // int nr=R.size();
    int nt=Type.size();
    
    int combo=nm*nr*nt*nhits*nconfs;
    
    int neq  = fact(nm+nr-1)/fact(nr)/fact(nm-1);
    int neq2=nm;
    
    int moms=mom_list.size();
    
    //array of input files to be read in a given conf
    ifstream input[combo];
    
    //Vector of interesting quantities (ALL MOMS)
    vector<jZ_t> jZq_allmoms(moms,vvd_t(vd_t(0.0,nmr),njacks)), jSigma1_allmoms(moms,vvd_t(vd_t(0.0,nmr),njacks)), jZq_em_allmoms(moms,vvd_t(vd_t(0.0,nmr),njacks)), jSigma1_em_allmoms(moms,vvd_t(vd_t(0.0,nmr),njacks));
    
    vector<vd_t> jZq_sub_allmoms(moms,vd_t(0.0,njacks)), jSigma1_sub_allmoms(moms,vd_t(0.0,njacks)), jZq_em_sub_allmoms(moms,vd_t(0.0,njacks)), jSigma1_em_sub_allmoms(moms,vd_t(0.0,njacks));
    
    vector<vvd_t> jZq_equivalent_allmoms(moms,vvd_t(vd_t(0.0,neq2),njacks)), jSigma1_equivalent_allmoms(moms,vvd_t(vd_t(0.0,neq2),njacks)), jZq_em_equivalent_allmoms(moms,vvd_t(vd_t(0.0,neq2),njacks)), jSigma1_em_equivalent_allmoms(moms,vvd_t(vd_t(0.0,neq2),njacks));
    
    vector<jZbil_t> jZ_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks)), jZ1_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks)), jZ_em_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks)), jZ1_em_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks));
    
    vector<vvd_t> jZ_sub_allmoms(moms,vvd_t(vd_t(0.0,5),njacks)), jZ1_sub_allmoms(moms,vvd_t(vd_t(0.0,5),njacks)), jZ_em_sub_allmoms(moms,vvd_t(vd_t(0.0,5),njacks)), jZ1_em_sub_allmoms(moms,vvd_t(vd_t(0.0,5),njacks)), jZ_em_sub_strong_allmoms(moms,vvd_t(vd_t(0.0,5),njacks)), jZ1_em_sub_strong_allmoms(moms,vvd_t(vd_t(0.0,5),njacks));
    
    vector<vvd_t> jGv_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGa_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGt_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks));
    vector<vvd_t> jGp_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGp_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks));
    
    vector<vvd_t> jGp_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGp_em_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_em_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks));
    vector<vvd_t> jGv_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGa_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)),jGt_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks));
    
    vector<vd_t> jGp_0_chiral_allmoms(moms,vd_t(njacks)),jGa_0_chiral_allmoms(moms,vd_t(njacks)),jGv_0_chiral_allmoms(moms,vd_t(njacks)), jGs_0_chiral_allmoms(moms,vd_t(njacks)),jGt_0_chiral_allmoms(moms,vd_t(njacks));
    vector<vd_t> jGp_em_a_b_chiral_allmoms(moms,vd_t(njacks)),jGa_em_a_b_chiral_allmoms(moms,vd_t(njacks)),jGv_em_a_b_chiral_allmoms(moms,vd_t(njacks)), jGs_em_a_b_chiral_allmoms(moms,vd_t(njacks)),jGt_em_a_b_chiral_allmoms(moms,vd_t(njacks));
    
    vector<vd_t> jGp_0_sub_allmoms(moms,vd_t(njacks)), jGa_0_sub_allmoms(moms,vd_t(njacks)), jGv_0_sub_allmoms(moms,vd_t(njacks)), jGs_0_sub_allmoms(moms,vd_t(njacks)), jGt_0_sub_allmoms(moms,vd_t(njacks));
    vector<vd_t> jGp_em_a_b_sub_allmoms(moms,vd_t(njacks)), jGa_em_a_b_sub_allmoms(moms,vd_t(njacks)), jGv_em_a_b_sub_allmoms(moms,vd_t(njacks)), jGs_em_a_b_sub_allmoms(moms,vd_t(njacks)), jGt_em_a_b_sub_allmoms(moms,vd_t(njacks));
    
    vector<vd_t> jZq_chiral_allmoms(moms,vd_t(0.0,njacks)),jSigma1_chiral_allmoms(moms,vd_t(0.0,njacks));
    vector<vd_t> jZq_em_chiral_allmoms(moms,vd_t(0.0,njacks)),jSigma1_em_chiral_allmoms(moms,vd_t(0.0,njacks));
    
    vector<vvd_t> jZ_chiral_allmoms(moms,vvd_t(vd_t(0.0,5),njacks)),jZ1_chiral_allmoms(moms,vvd_t(vd_t(0.0,5),njacks));
    vector<vvd_t> jZ_em_chiral_allmoms(moms,vvd_t(vd_t(0.0,5),njacks)),jZ1_em_chiral_allmoms(moms,vvd_t(vd_t(0.0,5),njacks));
    
    
    vector<vd_t> jSigma1_RIp_ainv_allmoms(moms,vd_t(0.0,njacks)),jSigma1_em_RIp_ainv_allmoms(moms,vd_t(0.0,njacks));
    vector<vvd_t> jZO_RIp_ainv_allmoms(moms,vvd_t(vd_t(5),njacks)),jZO_em_RIp_ainv_allmoms(moms,vvd_t(vd_t(5),njacks));

    vector<vd_t> jSigma1_sub_RIp_ainv_allmoms(moms,vd_t(0.0,njacks)),jSigma1_em_sub_RIp_ainv_allmoms(moms,vd_t(0.0,njacks));
    vector<vvd_t> jZO_sub_RIp_ainv_allmoms(moms,vvd_t(vd_t(5),njacks)),jZO_em_sub_RIp_ainv_allmoms(moms,vvd_t(vd_t(5),njacks));
    vector<vvd_t> jZO_em_sub_strong_RIp_ainv_allmoms(moms,vvd_t(vd_t(5),njacks));
    
    vector< vXd_t > jGp_pars_allmoms(moms,vXd_t(VectorXd(3),njacks)), jGs_pars_allmoms(moms,vXd_t(VectorXd(3),njacks)), jGp_em_pars_allmoms(moms,vXd_t(VectorXd(3),njacks)), jGs_em_pars_allmoms(moms,vXd_t(VectorXd(3),njacks));
    vector< vXd_t > jGv_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGa_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGt_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGv_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGa_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGt_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks));
    vector< vXd_t > jZq_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jSigma1_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jZq_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jSigma1_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks));
    
    vd_t m_eff_equivalent(0.0,neq);
    vd_t m_eff_equivalent_Zq(0.0,neq2);
    
    t0=high_resolution_clock::now();
    
#pragma omp parallel for collapse(5)
    for(int iconf=0;iconf<nconfs;iconf++)
        for(size_t ihit=0;ihit<nhits;ihit++)
            for(int t=0;t<nt;t++)
                for(int m=0;m<nm;m++)
                    for(int r=0;r<nr;r++)
                    {
                        string hit_suffix = "";
                        if(nhits>1) hit_suffix = "_hit_" + to_string(ihit);
                        
                        int icombo=r + nr*m + nr*nm*t + nr*nm*nt*ihit + nr*nm*nt*nhits*iconf;
                        string path = path_to_conf(string_path,conf_id[iconf],"S_"+Mass[m]+R[r]+Type[t]+hit_suffix);
                        
                        input[icombo].open(path,ios::binary);
                        
                        printf("  Opening file %s\n",path.c_str());
                        
                        if(!input[icombo].good())
                        {
                            cerr<<"Unable to open file "<<path<<" combo "<<icombo<<"/"<<combo<<endl;
                            exit(1);
                        }
                    }
    
    t1=high_resolution_clock::now();
    t_span = duration_cast<duration<double>>(t1-t0);
    cout<<"***** Opened all the files to read props in "<<t_span.count()<<" s ******"<<endl<<endl;
    
    
    int tag=0, tag_aux=0;
    double eps=1.0e-15;
    tag_vector.push_back(0);
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ mom loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    for(size_t imom=0; imom<mom_list.size(); imom++)
    {
        // put to zero jackknife props and verts
        jprop_t jS_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        
        jvert_t jVert_0 (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        jvert_t jVert_11_self_tad (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        jvert_t jVert_p (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        
        jprop_t jS_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
        jvert_t jVert_em (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        
        
        t0=high_resolution_clock::now();
        
        // initialize props
        vvvprop_t S(vvprop_t(vprop_t(prop_t::Zero(),nmr),nt),njacks);  // S[iconf][type][mr]
        vvprop_t S_em(vprop_t(prop_t::Zero(),nmr),njacks);
        
        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
            for(size_t ihit=0;ihit<nhits;ihit++)
            {
                string hit_suffix = "";
                if(nhits>1) hit_suffix = "_hit_" + to_string(ihit);
                
#pragma omp parallel for collapse(4)
                for(int t=0;t<nt;t++)
                    for(int m=0;m<nm;m++)
                        for(int r=0;r<nr;r++)
                            for(int ijack=0;ijack<njacks;ijack++)
                            {
                                int iconf=clust_size*ijack+i_in_clust;
                                
                                int icombo=r + nr*m + nr*nm*t + nr*nm*nt*ihit + nr*nm*nt*nhits*iconf;
                                string path = path_to_conf(string_path,conf_id[iconf],"S_"+Mass[m]+R[r]+Type[t]+hit_suffix);
                                
                                int mr = r + nr*m; // M0R0,M0R1,M1R0,M1R1,M2R0,M2R1,M3R0,M3R1
                                
                                //DEBUG
                                //printf("  Reading propagator from %s\n",path.c_str());
                                //DEBUG
                                
                                //create all the propagators in a given conf and a given mom
                                S[ijack][t][mr] = read_prop(input[icombo],path);
                               
                                if(t==4) S[ijack][t][mr]*=dcompl(0.0,1.0);      // i*(pseudoscalar insertion)
                                //if(t==5) S[ijack][t][mr]*=dcompl(1.0,0.0);    // (minus sign?)
                            }
                
#pragma omp parallel for collapse(3)
                for(int m=0;m<nm;m++)
                    for(int r=0;r<nr;r++)
                        for(int ijack=0;ijack<njacks;ijack++)
                        {
                            int mr = r + nr*m;
                            
                            // Electromagnetic correction:  S_em = S_self + S_tad -+ deltam_cr*S_P
                            if(r==0) S_em[ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] + deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=0
                            if(r==1) S_em[ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] - deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=1
                        }
         
#pragma omp parallel for collapse (2)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr=0;mr<nmr;mr++)
                    {
                        //int iconf=clust_size*ijack+i_in_clust;
                        
                        jS_0[ijack][mr] += S[ijack][0][mr];
                        jS_em[ijack][mr] += S_em[ijack][mr];
                    }
                
#pragma omp parallel for collapse (4)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                            for(int igam=0;igam<16;igam++)
                            {
                                //int iconf=clust_size*ijack+i_in_clust;
                                
                                jVert_0[ijack][mr_fw][mr_bw][igam] += make_vertex(S[ijack][0][mr_fw], S[ijack][0][mr_bw],igam,GAMMA);
                                jVert_em[ijack][mr_fw][mr_bw][igam] += make_vertex(S[ijack][0][mr_fw],S_em[ijack][mr_bw],igam,GAMMA) + make_vertex(S_em[ijack][mr_fw],S[ijack][0][mr_bw],igam,GAMMA) + make_vertex(S[ijack][1][mr_fw],S[ijack][1][mr_bw],igam,GAMMA) ;
                            }
                
            } //close hits&in_i_clust loop
        
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Read propagators and created vertices (and jackknives) in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        t0=high_resolution_clock::now();
        
        //jackknife of propagators
        jS_0 = jackknife_prop(jS_0,nconfs,clust_size,nhits);
        jS_em = jackknife_prop(jS_em,nconfs,clust_size,nhits);
        
        // //jackknife of vertices
        jVert_0 = jackknife_vertex(jVert_0,nconfs,clust_size,nhits);
        jVert_em = jackknife_vertex(jVert_em,nconfs,clust_size,nhits);
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Computed jackknives averages (prop&vert) in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        
        //inverse of the propagators
        t0=high_resolution_clock::now();
        
        jprop_t jS_0_inv = invert_jprop(jS_0);         //jS_0_inv[ijack][mr]
        jprop_t jS_em_inv = jS_0_inv*jS_em*jS_0_inv;
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Inverted propagators in "<<t_span.count()<<" s ******"<<endl<<endl;
       
        
//        for(int ijack=0;ijack<njacks;ijack++)
//            for(int r=0;r<nr;r++)
//                for(int m=0;m<nm;m++)
//                {
//                    cout<<"jS_em[0,0] (ijack "<<ijack<<" r"<<r<<" m"<<m<<" imom "<<imom<<") = "<<jS_em[ijack][r+nr*m](0,0)<<endl;
//                }
//        cout<<endl;
//        
//        for(int ijack=0;ijack<njacks;ijack++)
//            for(int r=0;r<nr;r++)
//                for(int m=0;m<nm;m++)
//                {
//                    cout<<"jS_em_inv[0,0] (ijack "<<ijack<<" r"<<r<<" m"<<m<<" imom "<<imom<<") = "<<jS_em_inv[ijack][r+nr*m](0,0)<<endl;
//                    cout<<"jS_em_inv[11,11] (ijack "<<ijack<<" r"<<r<<" m"<<m<<" imom "<<imom<<") = "<<jS_em_inv[ijack][r+nr*m](11,11)<<endl;
//                }
//        cout<<endl;
        
        //amputate external legs
        t0=high_resolution_clock::now();
        
        jvert_t jLambda_0(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);    //jLambda_0[ijack][mr_fw][mr_bw][gamma]
        jvert_t jLambda_em(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        jvert_t jLambda_a(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        jvert_t jLambda_b(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
        
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                    for(int igam=0;igam<16;igam++)
                    {
                        jLambda_0[ijack][mr_fw][mr_bw][igam] = amputate(jS_0_inv[ijack][mr_fw], jVert_0[ijack][mr_fw][mr_bw][igam], jS_0_inv[ijack][mr_bw], GAMMA);  //jLambda_0[ijack][mr_fw][mr_bw][igam]
                        jLambda_em[ijack][mr_fw][mr_bw][igam] = amputate(jS_0_inv[ijack][mr_fw], jVert_em[ijack][mr_fw][mr_bw][igam], jS_0_inv[ijack][mr_bw], GAMMA);
                        jLambda_a[ijack][mr_fw][mr_bw][igam] = amputate(jS_em_inv[ijack][mr_fw], jVert_0[ijack][mr_fw][mr_bw][igam], jS_0_inv[ijack][mr_bw], GAMMA);
                        jLambda_b[ijack][mr_fw][mr_bw][igam] = amputate(jS_0_inv[ijack][mr_fw], jVert_0[ijack][mr_fw][mr_bw][igam], jS_em_inv[ijack][mr_bw], GAMMA);
                    }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Amputated external legs in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        t0=high_resolution_clock::now();
        
        //compute Zq according to RI'-MOM, one for each momentum
        jZ_t jZq = compute_jZq(GAMMA,jS_0_inv,L,T,imom);            //jZq[ijack][mr]
        jZ_t jZq_em = - compute_jZq(GAMMA,jS_em_inv,L,T,imom);
        
        //compute Zq according to Sigma1-way
        jZ_t jSigma1 = compute_jSigma1(GAMMA,jS_0_inv,L,T,imom);
        jZ_t jSigma1_em = - compute_jSigma1(GAMMA,jS_em_inv,L,T,imom);
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Computed Zq&Sigma1 in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        t0=high_resolution_clock::now();

        //compute the projected green function as a vector (S,V,P,A,T)
        
        jproj_t jG_0 = project(GAMMA,jLambda_0);        //jG_0[ijack][mr_fw][mr_bw][i] (i=S,V,P,A,T)
        jproj_t jG_em = project(GAMMA,jLambda_em);
        jproj_t jG_a = project(GAMMA,jLambda_a);
        jproj_t jG_b = project(GAMMA,jLambda_b);
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Projected Green Functions in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        t0=high_resolution_clock::now();
        
        //compute Z's according to RI-MOM and to Sigma1-way, one for each momentum
        
        jZbil_t jZ(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
        jZbil_t jZ1(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
        
        jZbil_t jZ_em(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
        jZbil_t jZ1_em(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
        
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)  // according to 'riqed.pdf'
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                    for(int k=0;k<5;k++)
                    {
                        jZ[ijack][mr_fw][mr_bw][k] = sqrt(jZq[ijack][mr_fw])*sqrt(jZq[ijack][mr_bw])/jG_0[ijack][mr_fw][mr_bw][k];
                        jZ1[ijack][mr_fw][mr_bw][k] = sqrt(jSigma1[ijack][mr_fw])*sqrt(jSigma1[ijack][mr_bw])/jG_0[ijack][mr_fw][mr_bw][k];
                        
                        jZ_em[ijack][mr_fw][mr_bw][k] = (-jG_em[ijack][mr_fw][mr_bw][k]+jG_a[ijack][mr_fw][mr_bw][k]+jG_b[ijack][mr_fw][mr_bw][k])/jG_0[ijack][mr_fw][mr_bw][k] + \
                        0.5*(jZq_em[ijack][mr_fw]/jZq[ijack][mr_fw] + jZq_em[ijack][mr_bw]/jZq[ijack][mr_bw]);
                        jZ1_em[ijack][mr_fw][mr_bw][k] = (-jG_em[ijack][mr_fw][mr_bw][k]+jG_a[ijack][mr_fw][mr_bw][k]+jG_b[ijack][mr_fw][mr_bw][k])/jG_0[ijack][mr_fw][mr_bw][k] + \
                        0.5*(jSigma1_em[ijack][mr_fw]/jSigma1[ijack][mr_fw] + jSigma1_em[ijack][mr_bw]/jSigma1[ijack][mr_bw]);
                    }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Computed Zbil in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        //create p_tilde vector
        
        vd_t p(0.0,4);
        vd_t p_tilde(0.0,4);
        double p2=0.0;
        double p2_not_tilde=0.0;
        double p2_space=0.0;
        double p4=0.0;  //for the democratic filter
        
        p={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
        p_tilde={sin(p[0]),sin(p[1]),sin(p[2]),sin(p[3])};
        
        for(int coord=0;coord<4;coord++)
            p2_not_tilde+=p[coord]*p[coord];
        for(int coord=0;coord<4;coord++)
            p2+=p_tilde[coord]*p_tilde[coord];
        for(int coord=0;coord<3;coord++)
            p2_space+=p_tilde[coord]*p_tilde[coord];
        for(int coord=0;coord<4;coord++)
            p4+=p_tilde[coord]*p_tilde[coord]*p_tilde[coord]*p_tilde[coord]; //for the democratic filter
        
        p2_vector_allmoms.push_back(p2);
        p4_vector.push_back(p4);
        
        vector<double> c_q(3), c_q_em(3);
        
        if(strcmp(arg[8],"sym")==0)
        {
            c_q={1.14716212+2.07733285/(double)Np[imom],-73./360.-157./180./(double)Np[imom],7./240.};   //Symanzik action
            
            c_q_em={-0.0112397+2.26296238/(double)Np[imom],31./240.-101./120./(double)Np[imom],17./120.};	   //Wilson action (QED)
        }
        else if(strcmp(arg[8],"iwa")==0)
        {
            c_q={0.6202244+1.8490436/(double)Np[imom],-0.0748167-0.963033/(double)Np[imom],0.0044};      //Iwasaki action

        }
        else if(strcmp(arg[8],"free")==0) //Free action
        {
            c_q={0.0,0.0,0.0};
            
            c_q_em={0.0,0.0,0.0};
        }
        
        jproj_t jG_em_a_b = -jG_em + jG_a + jG_b;  //minus sign w.r.t. eq. (3.42) thesis.
        
        
        //Sum of quark masses for the extrapolation
        vd_t mass_sum(0.0,10);
        int i_sum = 0;
        
        for (int i=0; i<nm; i++)
            for(int j=i;j<nm;j++)
            {
                mass_sum[i_sum] = mass_array[i]+mass_array[j];
                
                cout<<mass_sum[i_sum]<<endl;
                
                i_sum++;
            } // isum=0,1,2,3,4,5,6,7,8,9

        
        
        //Goldstone pole subtraction from jG_p and jG_s & chiral extrapolation of jG_p and jG_s
        t0=high_resolution_clock::now();
        
        vvd_t jGp_equivalent(vd_t(0.0,neq),njacks);
        vvd_t jGs_equivalent(vd_t(0.0,neq),njacks);
        
        vvd_t jGp_em_equivalent(vd_t(0.0,neq),njacks);
        vvd_t jGs_em_equivalent(vd_t(0.0,neq),njacks);
        
        int ieq=0;
        
        if(imom==0)
        {
            for(int mA=0; mA<nm; mA++)
                for(int mB=mA; mB<nm; mB++)
                {
                    m_eff_equivalent[ieq] = mass_sum[ieq]; //charged channel
                    
                    ieq++;
                }
            
        }  //ieq={00,01,02,03,11,12,13,22,23,33}
        
        ieq=0;
        
        for(int mA=0; mA<nm; mA++)
            for(int mB=mA; mB<nm; mB++)
            {
                for(int r=0; r<nr; r++)
                {
                    //ieq=-(mA*mA/2)+mB+mA*(nm-0.5);
                    
                    for(int ijack=0;ijack<njacks;ijack++) jGp_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_0[ijack][r+nr*mA][r+nr*mB][2]+jG_0[ijack][r+nr*mB][r+nr*mA][2])/(2.0*nr);
                    for(int ijack=0;ijack<njacks;ijack++) jGs_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_0[ijack][r+nr*mA][r+nr*mB][0]+jG_0[ijack][r+nr*mB][r+nr*mA][0])/(2.0*nr);
                    
                    for(int ijack=0;ijack<njacks;ijack++) jGp_em_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_em_a_b[ijack][r+nr*mA][r+nr*mB][2]+jG_em_a_b[ijack][r+nr*mB][r+nr*mA][2])/(2.0*nr);
                    for(int ijack=0;ijack<njacks;ijack++) jGs_em_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_em_a_b[ijack][r+nr*mA][r+nr*mB][0]+jG_em_a_b[ijack][r+nr*mB][r+nr*mA][0])/(2.0*nr);
                }
                ieq++;
            }   //ieq={00,01,02,03,11,12,13,22,23,33}
        
        
        vd_t Gp_ave(0.0,neq), sqr_Gp_ave(0.0,neq), Gp_err(0.0,neq);
        vd_t Gs_ave(0.0,neq), sqr_Gs_ave(0.0,neq), Gs_err(0.0,neq);
        
        vd_t Gp_em_ave(0.0,neq), sqr_Gp_em_ave(0.0,neq), Gp_em_err(0.0,neq);
        vd_t Gs_em_ave(0.0,neq), sqr_Gs_em_ave(0.0,neq), Gs_em_err(0.0,neq);
        
        //#pragma omp parallel for
        for(int i=0;i<neq;i++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                Gp_ave[i]+=jGp_equivalent[ijack][i]/njacks;
                sqr_Gp_ave[i]+=jGp_equivalent[ijack][i]*jGp_equivalent[ijack][i]/njacks;
                
                Gs_ave[i]+=jGs_equivalent[ijack][i]/njacks;
                sqr_Gs_ave[i]+=jGs_equivalent[ijack][i]*jGs_equivalent[ijack][i]/njacks;
                
                Gp_em_ave[i]+=jGp_em_equivalent[ijack][i]/njacks;
                sqr_Gp_em_ave[i]+=jGp_em_equivalent[ijack][i]*jGp_em_equivalent[ijack][i]/njacks;
                
                Gs_em_ave[i]+=jGs_em_equivalent[ijack][i]/njacks;
                sqr_Gs_em_ave[i]+=jGs_em_equivalent[ijack][i]*jGs_em_equivalent[ijack][i]/njacks;
            }
        
        for(int i=0;i<neq;i++)
        {
            Gp_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gp_ave[i]-Gp_ave[i]*Gp_ave[i]));
            Gs_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gs_ave[i]-Gs_ave[i]*Gs_ave[i]));
            
            Gp_em_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gp_em_ave[i]-Gp_em_ave[i]*Gp_em_ave[i]));
            Gs_em_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gs_em_ave[i]-Gs_em_ave[i]*Gs_em_ave[i]));
        }
        
        //range for the fit
        int t_min=0;
        int t_max=neq-1;
        
        vvd_t coord(vd_t(0.0,neq),3);
        for(int i=0; i<neq; i++)
        {
            coord[0][i] = 1.0;  //costante
            //coord[1][i] = m_eff_equivalent[i]*m_eff_equivalent[i];   //M^2
            //coord[2][i] = 1.0/(m_eff_equivalent[i]*m_eff_equivalent[i]);  //1/M^2
            coord[1][i] = mass_sum[i];   // (am1+am2)
            coord[2][i] = 1.0/mass_sum[i];  // 1/(am1+am2)
        }
        
        vXd_t jGp_pars=fit_par_jackknife(coord,Gp_err,jGp_equivalent,t_min,t_max);  //jGp_pars[ijack](par)
        vXd_t jGs_pars=fit_par_jackknife(coord,Gs_err,jGs_equivalent,t_min,t_max);
        vXd_t jGp_em_pars=fit_par_jackknife(coord,Gp_em_err,jGp_em_equivalent,t_min,t_max);
        vXd_t jGs_em_pars=fit_par_jackknife(coord,Gs_em_err,jGs_em_equivalent,t_min,t_max);
        
        vd_t C_p(njacks), C_s(njacks), jGp_0_chiral(njacks), jGs_0_chiral(njacks);
        vd_t C_p_em(njacks), C_s_em(njacks), jGp_em_a_b_chiral(njacks), jGs_em_a_b_chiral(njacks);
        vvd_t jGp_subpole(vd_t(neq),njacks), jGs_subpole(vd_t(neq),njacks);
        vvd_t jGp_em_subpole(vd_t(neq),njacks), jGs_em_subpole(vd_t(neq),njacks);
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            C_p[ijack]=jGp_pars[ijack](2);
            C_s[ijack]=jGs_pars[ijack](2);
            C_p_em[ijack]=jGp_em_pars[ijack](2);
            C_s_em[ijack]=jGs_em_pars[ijack](2);
        }
        
        //  cout<<"----- Goldstone fit parameters (for each jackknife) ------"<<endl;
        //  for(int ijack=0;ijack<njacks;ijack++) cout<<jGp_pars[ijack][0]<<"  "<<jGp_pars[ijack][1]<<"  "<<C_p[ijack]<<endl;
        //  cout<<endl;
        
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int i=0; i<neq; i++)
            {
                jGp_subpole[ijack][i] = jGp_equivalent[ijack][i] - C_p[ijack]/mass_sum[i];
                jGs_subpole[ijack][i] = jGs_equivalent[ijack][i] - C_s[ijack]/mass_sum[i];
                
                jGp_em_subpole[ijack][i] = jGp_em_equivalent[ijack][i] - C_p_em[ijack]/mass_sum[i];
                jGs_em_subpole[ijack][i] = jGs_em_equivalent[ijack][i] - C_s_em[ijack]/mass_sum[i];
            }
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jGp_0_chiral[ijack]=jGp_pars[ijack](0);
            jGs_0_chiral[ijack]=jGs_pars[ijack](0);
            
            jGp_em_a_b_chiral[ijack]=jGp_em_pars[ijack](0);
            jGs_em_a_b_chiral[ijack]=jGs_em_pars[ijack](0);
        }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Goldstone pole subtraction in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        
        
        //chiral extrapolation of Gv,Ga,Gt
        t0=high_resolution_clock::now();
        
        vvd_t jGv_equivalent(vd_t(0.0,neq),njacks);
        vvd_t jGa_equivalent(vd_t(0.0,neq),njacks);
        vvd_t jGt_equivalent(vd_t(0.0,neq),njacks);
        
        vvd_t jGv_em_equivalent(vd_t(0.0,neq),njacks);
        vvd_t jGa_em_equivalent(vd_t(0.0,neq),njacks);
        vvd_t jGt_em_equivalent(vd_t(0.0,neq),njacks);
        
        ieq=0;
        
        for(int mA=0; mA<nm; mA++)
            for(int mB=mA; mB<nm; mB++)
            {
                for(int r=0; r<nr; r++)
                {
                    //ieq=-(mA*mA/2)+mB+mA*(nm-0.5);
                    
                    for(int ijack=0;ijack<njacks;ijack++) jGv_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_0[ijack][r+nr*mA][r+nr*mB][1]+jG_0[ijack][r+nr*mB][r+nr*mA][1])/(2.0*nr);
                    for(int ijack=0;ijack<njacks;ijack++) jGa_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_0[ijack][r+nr*mA][r+nr*mB][3]+jG_0[ijack][r+nr*mB][r+nr*mA][3])/(2.0*nr);
                    for(int ijack=0;ijack<njacks;ijack++) jGt_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_0[ijack][r+nr*mA][r+nr*mB][4]+jG_0[ijack][r+nr*mB][r+nr*mA][4])/(2.0*nr);
                    
                    for(int ijack=0;ijack<njacks;ijack++) jGv_em_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_em_a_b[ijack][r+nr*mA][r+nr*mB][1]+jG_em_a_b[ijack][r+nr*mB][r+nr*mA][1])/(2.0*nr);
                    for(int ijack=0;ijack<njacks;ijack++) jGa_em_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_em_a_b[ijack][r+nr*mA][r+nr*mB][3]+jG_em_a_b[ijack][r+nr*mB][r+nr*mA][3])/(2.0*nr);
                    for(int ijack=0;ijack<njacks;ijack++) jGt_em_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*(jG_em_a_b[ijack][r+nr*mA][r+nr*mB][4]+jG_em_a_b[ijack][r+nr*mB][r+nr*mA][4])/(2.0*nr);
                }
                ieq++;
            } //ieq={00,01,02,03,11,12,13,22,23,33}
        
//        for(int mA=0; mA<nm; mA++)
//            for(int mB=mA; mB<nm; mB++)
//                for(int r=0; r<nr; r++)
//                    for(int ijack=0;ijack<njacks;ijack++)
//                        cout<<"mA "<<mA<<" mB "<<mB<<" r "<<r<<" ijack "<<ijack<<"  "<<jG_0[ijack][r+nr*mA][r+nr*mB][1]<<"  "<<jG_0[ijack][r+nr*mB][r+nr*mA][1]<<endl;
        
        
        vd_t Gv_ave(0.0,neq),sqr_Gv_ave(0.0,neq),Gv_err(0.0,neq),Ga_ave(0.0,neq),sqr_Ga_ave(0.0,neq),Ga_err(0.0,neq),Gt_ave(0.0,neq),sqr_Gt_ave(0.0,neq),Gt_err(0.0,neq);
        vd_t Gv_em_ave(0.0,neq),sqr_Gv_em_ave(0.0,neq),Gv_em_err(0.0,neq),Ga_em_ave(0.0,neq),sqr_Ga_em_ave(0.0,neq),Ga_em_err(0.0,neq),Gt_em_ave(0.0,neq),sqr_Gt_em_ave(0.0,neq),Gt_em_err(0.0,neq);
        
#pragma omp parallel for
        for(int i=0;i<neq;i++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                Gv_ave[i]+=jGv_equivalent[ijack][i]/njacks;
                sqr_Gv_ave[i]+=jGv_equivalent[ijack][i]*jGv_equivalent[ijack][i]/njacks;
                
                Ga_ave[i]+=jGa_equivalent[ijack][i]/njacks;
                sqr_Ga_ave[i]+=jGa_equivalent[ijack][i]*jGa_equivalent[ijack][i]/njacks;
                
                Gt_ave[i]+=jGt_equivalent[ijack][i]/njacks;
                sqr_Gt_ave[i]+=jGt_equivalent[ijack][i]*jGt_equivalent[ijack][i]/njacks;
                
                Gv_em_ave[i]+=jGv_em_equivalent[ijack][i]/njacks;
                sqr_Gv_em_ave[i]+=jGv_em_equivalent[ijack][i]*jGv_em_equivalent[ijack][i]/njacks;
                
                Ga_em_ave[i]+=jGa_em_equivalent[ijack][i]/njacks;
                sqr_Ga_em_ave[i]+=jGa_em_equivalent[ijack][i]*jGa_em_equivalent[ijack][i]/njacks;
                
                Gt_em_ave[i]+=jGt_em_equivalent[ijack][i]/njacks;
                sqr_Gt_em_ave[i]+=jGt_em_equivalent[ijack][i]*jGt_em_equivalent[ijack][i]/njacks;
            }
#pragma omp parallel for
        for(int i=0;i<neq;i++)
        {
            Gv_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gv_ave[i]-Gv_ave[i]*Gv_ave[i]));
            Ga_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Ga_ave[i]-Ga_ave[i]*Ga_ave[i]));
            Gt_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gt_ave[i]-Gt_ave[i]*Gt_ave[i]));
            
            Gv_em_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gv_em_ave[i]-Gv_em_ave[i]*Gv_em_ave[i]));      //Green function with gamma_mu         --> Renorm. with ZA
            Ga_em_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Ga_em_ave[i]-Ga_em_ave[i]*Ga_em_ave[i]));      //Green function with gamma_mu*gamma_5 --> Renorm. with ZV
            Gt_em_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Gt_em_ave[i]-Gt_em_ave[i]*Gt_em_ave[i]));
        }
        
        //range for the fit
        t_min=0;
        t_max=neq-1;
        
        vvd_t coord_linear(vd_t(0.0,neq),2);
        for(int i=0; i<neq; i++)
        {
            coord_linear[0][i] = 1.0;  //costante
            //coord_linear[1][i] = m_eff_equivalent[i]*m_eff_equivalent[i];   //M^2
            coord_linear[1][i] = mass_sum[i];
        }
        
        vXd_t jGv_pars(VectorXd(2),njacks), jGa_pars(VectorXd(2),njacks), jGt_pars(VectorXd(2),njacks);
        vXd_t jGv_em_pars(VectorXd(2),njacks), jGa_em_pars(VectorXd(2),njacks), jGt_em_pars(VectorXd(2),njacks);
        
        jGv_pars=fit_par_jackknife(coord_linear,Gv_err,jGv_equivalent,t_min,t_max);  //jGv_pars[ijack][par]
        jGa_pars=fit_par_jackknife(coord_linear,Ga_err,jGa_equivalent,t_min,t_max);
        jGt_pars=fit_par_jackknife(coord_linear,Gt_err,jGt_equivalent,t_min,t_max);
        
        jGv_em_pars=fit_par_jackknife(coord_linear,Gv_em_err,jGv_em_equivalent,t_min,t_max);  //jGv_pars[ijack][par]
        jGa_em_pars=fit_par_jackknife(coord_linear,Ga_em_err,jGa_em_equivalent,t_min,t_max);
        jGt_em_pars=fit_par_jackknife(coord_linear,Gt_em_err,jGt_em_equivalent,t_min,t_max);
        
        vd_t jGv_0_chiral(njacks), jGa_0_chiral(njacks), jGt_0_chiral(njacks);
        vd_t jGv_em_a_b_chiral(njacks), jGa_em_a_b_chiral(njacks), jGt_em_a_b_chiral(njacks);
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jGv_0_chiral[ijack]=jGv_pars[ijack](0);
            jGa_0_chiral[ijack]=jGa_pars[ijack](0);
            jGt_0_chiral[ijack]=jGt_pars[ijack](0);
            
            jGv_em_a_b_chiral[ijack]=jGv_em_pars[ijack](0);
            jGa_em_a_b_chiral[ijack]=jGa_em_pars[ijack](0);
            jGt_em_a_b_chiral[ijack]=jGt_em_pars[ijack](0);
        }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Chiral extrapolation of Gv,Ga&Gt in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        //chiral extrapolation of Zq
        t0=high_resolution_clock::now();
        
        vvd_t jZq_equivalent(vd_t(0.0,neq2),njacks), jSigma1_equivalent(vd_t(0.0,neq2),njacks);
        vvd_t jZq_em_equivalent(vd_t(0.0,neq2),njacks), jSigma1_em_equivalent(vd_t(0.0,neq2),njacks);
        
        for(int m=0; m<nm; m++)
            m_eff_equivalent_Zq[m]=mass_array[m];
        
        for(int m=0; m<nm; m++)
            for(int r=0; r<nr; r++)
            {
                ieq=m;
                //if(imom==0)  m_eff_equivalent_Zq[ieq] += eff_mass[r+nr*m][r+nr*m]/nr; //charged channel
                
                //LO
                for(int ijack=0;ijack<njacks;ijack++) jZq_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*jZq[ijack][r+nr*m]/nr;
                for(int ijack=0;ijack<njacks;ijack++) jSigma1_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*jSigma1[ijack][r+nr*m]/nr;
                //EM
                for(int ijack=0;ijack<njacks;ijack++) jZq_em_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*jZq_em[ijack][r+nr*m]/nr;
                for(int ijack=0;ijack<njacks;ijack++) jSigma1_em_equivalent[ijack][ieq] += (c1*((r+1)%2)+c2*(r%2))*jSigma1_em[ijack][r+nr*m]/nr;
            }
        
        for(int ijack=0; ijack<njacks;ijack++)
            for(int m=0; m<nm; m++)
                for(int r=0; r<nr; r++)
                {
                    cout<<"Zq_em (ijack "<<ijack<<" r "<<r<<" m "<<m<<" imom "<<imom<<") = "<<jZq_em[ijack][r+nr*m]<<endl;
                }
        for(int ijack=0; ijack<njacks;ijack++)
            for(int m=0; m<nm; m++)
                for(int r=0; r<nr; r++)
                {
                    cout<<"Sigma1_em (ijack "<<ijack<<" r "<<r<<" m "<<m<<" imom "<<imom<<") = "<<jSigma1_em[ijack][r+nr*m]<<endl;
                }
    
        vd_t Zq_ave(0.0,neq2), sqr_Zq_ave(0.0,neq2), Zq_err(0.0,neq2);
        vd_t Sigma1_ave(0.0,neq2), sqr_Sigma1_ave(0.0,neq2), Sigma1_err(0.0,neq2);
        
        vd_t Zq_em_ave(0.0,neq2), sqr_Zq_em_ave(0.0,neq2), Zq_em_err(0.0,neq2);
        vd_t Sigma1_em_ave(0.0,neq2), sqr_Sigma1_em_ave(0.0,neq2), Sigma1_em_err(0.0,neq2);
        
        //#pragma omp parallel for
        for(int i=0;i<neq2;i++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                Zq_ave[i]+=jZq_equivalent[ijack][i]/njacks;
                sqr_Zq_ave[i]+=jZq_equivalent[ijack][i]*jZq_equivalent[ijack][i]/njacks;
                
                Sigma1_ave[i]+=jSigma1_equivalent[ijack][i]/njacks;
                sqr_Sigma1_ave[i]+=jSigma1_equivalent[ijack][i]*jSigma1_equivalent[ijack][i]/njacks;
                
                Zq_em_ave[i]+=jZq_em_equivalent[ijack][i]/njacks;
                sqr_Zq_em_ave[i]+=jZq_em_equivalent[ijack][i]*jZq_em_equivalent[ijack][i]/njacks;
                
                Sigma1_em_ave[i]+=jSigma1_em_equivalent[ijack][i]/njacks;
                sqr_Sigma1_em_ave[i]+=jSigma1_em_equivalent[ijack][i]*jSigma1_em_equivalent[ijack][i]/njacks;
            }
        
        for(int i=0;i<neq2;i++)
        {
            Zq_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[i]-Zq_ave[i]*Zq_ave[i]));
            Sigma1_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Sigma1_ave[i]-Sigma1_ave[i]*Sigma1_ave[i]));
            
            Zq_em_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_em_ave[i]-Zq_em_ave[i]*Zq_em_ave[i]));
            Sigma1_em_err[i]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Sigma1_em_ave[i]-Sigma1_em_ave[i]*Sigma1_em_ave[i]));
        }
        
        for(int i=0;i<neq2;i++)
            cout<<"Sigma1 average (eq. mass "<<i<<" - imom "<<imom<<") = "<<Sigma1_ave[i]<<endl;
        
        //linear fit
        t_min=0;
        t_max=neq2-1;
        
        for(int i=0; i<neq2; i++)
        {
            coord_linear[0][i] = 1.0;  //costante
            //coord_linear[1][i] = m_eff_equivalent_Zq[i]*m_eff_equivalent_Zq[i];   //M^2
            coord_linear[1][i] = mass_array[i]; //quark mass
        }
        
        vXd_t jZq_pars=fit_par_jackknife(coord_linear,Zq_err,jZq_equivalent,t_min,t_max);  //jZq_pars[ijack][par]
        vXd_t jSigma1_pars=fit_par_jackknife(coord_linear,Sigma1_err,jSigma1_equivalent,t_min,t_max);  //jZq_pars[ijack][par]
        
        vXd_t jZq_em_pars=fit_par_jackknife(coord_linear,Zq_em_err,jZq_em_equivalent,t_min,t_max);
        vXd_t jSigma1_em_pars=fit_par_jackknife(coord_linear,Sigma1_em_err,jSigma1_em_equivalent,t_min,t_max);
        
        vd_t jZq_chiral(0.0,njacks), jSigma1_chiral(0.0,njacks), jZq_em_chiral(0.0,njacks), jSigma1_em_chiral(0.0,njacks);
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZq_chiral[ijack]=jZq_pars[ijack](0);
            jSigma1_chiral[ijack]=jSigma1_pars[ijack](0);
            
            jZq_em_chiral[ijack]=jZq_em_pars[ijack](0);
            jSigma1_em_chiral[ijack]=jSigma1_em_pars[ijack](0);
        }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Chiral extrapolation of Zq in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        
        //chiral jackknife Z's
        t0=high_resolution_clock::now();
        vvd_t jZ_chiral(vd_t(0.0,5),njacks), jZ1_chiral(vd_t(0.0,5),njacks);
        vvd_t jZ_em_chiral(vd_t(0.0,5),njacks), jZ1_em_chiral(vd_t(0.0,5),njacks);
        
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZ_chiral[ijack][0] = jZq_chiral[ijack]/jGs_0_chiral[ijack];  //ZS     Z_QCD
            jZ_chiral[ijack][1] = jZq_chiral[ijack]/jGv_0_chiral[ijack];  //ZA
            jZ_chiral[ijack][2] = jZq_chiral[ijack]/jGp_0_chiral[ijack];  //ZP
            jZ_chiral[ijack][3] = jZq_chiral[ijack]/jGa_0_chiral[ijack];  //ZV
            jZ_chiral[ijack][4] = jZq_chiral[ijack]/jGt_0_chiral[ijack];  //ZT
            
            jZ1_chiral[ijack][0] = jSigma1_chiral[ijack]/jGs_0_chiral[ijack]; //ZS
            jZ1_chiral[ijack][1] = jSigma1_chiral[ijack]/jGv_0_chiral[ijack]; //ZA
            jZ1_chiral[ijack][2] = jSigma1_chiral[ijack]/jGp_0_chiral[ijack];  //ZP
            jZ1_chiral[ijack][3] = jSigma1_chiral[ijack]/jGa_0_chiral[ijack];  //ZV
            jZ1_chiral[ijack][4] = jSigma1_chiral[ijack]/jGt_0_chiral[ijack];  //ZT
            
            jZ_em_chiral[ijack][0] = jGs_em_a_b_chiral[ijack]/jGs_0_chiral[ijack] + jZq_em_chiral[ijack]/jZq_chiral[ijack];  //ZS       deltaZ
            jZ_em_chiral[ijack][1] = jGv_em_a_b_chiral[ijack]/jGv_0_chiral[ijack] + jZq_em_chiral[ijack]/jZq_chiral[ijack];  //ZA
            jZ_em_chiral[ijack][2] = jGp_em_a_b_chiral[ijack]/jGp_0_chiral[ijack] + jZq_em_chiral[ijack]/jZq_chiral[ijack];  //ZP
            jZ_em_chiral[ijack][3] = jGa_em_a_b_chiral[ijack]/jGa_0_chiral[ijack] + jZq_em_chiral[ijack]/jZq_chiral[ijack];  //ZV
            jZ_em_chiral[ijack][4] = jGt_em_a_b_chiral[ijack]/jGt_0_chiral[ijack] + jZq_em_chiral[ijack]/jZq_chiral[ijack];  //ZT
            
            jZ1_em_chiral[ijack][0] = jGs_em_a_b_chiral[ijack]/jGs_0_chiral[ijack] + jSigma1_em_chiral[ijack]/jSigma1_chiral[ijack];  //ZS
            jZ1_em_chiral[ijack][1] = jGv_em_a_b_chiral[ijack]/jGv_0_chiral[ijack] + jSigma1_em_chiral[ijack]/jSigma1_chiral[ijack];  //ZA
            jZ1_em_chiral[ijack][2] = jGp_em_a_b_chiral[ijack]/jGp_0_chiral[ijack] + jSigma1_em_chiral[ijack]/jSigma1_chiral[ijack];  //ZP
            jZ1_em_chiral[ijack][3] = jGa_em_a_b_chiral[ijack]/jGa_0_chiral[ijack] + jSigma1_em_chiral[ijack]/jSigma1_chiral[ijack];  //ZV
            jZ1_em_chiral[ijack][4] = jGt_em_a_b_chiral[ijack]/jGt_0_chiral[ijack] + jSigma1_em_chiral[ijack]/jSigma1_chiral[ijack];  //ZT
        }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Chiral Z's computation in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        
        
        //Subtraction of O(a^2) effects through perturbation theory
        
        vd_t jZq_sub(0.0,njacks), jSigma1_sub(0.0,njacks), jZq_em_sub(0.0,njacks), jSigma1_em_sub(0.0,njacks);
        vd_t jGv_0_sub(njacks), jGa_0_sub(njacks), jGt_0_sub(njacks);
        vd_t jGv_em_a_b_sub(njacks), jGa_em_a_b_sub(njacks), jGt_em_a_b_sub(njacks);
        vd_t jGp_0_sub(njacks), jGs_0_sub(njacks);
        vd_t jGp_em_a_b_sub(njacks), jGs_em_a_b_sub(njacks);
        
        vd_t jZv_sub(njacks), jZa_sub(njacks), jZp_sub(njacks), jZs_sub(njacks), jZt_sub(njacks);
        vd_t jZv_em_sub(njacks), jZa_em_sub(njacks), jZp_em_sub(njacks), jZs_em_sub(njacks), jZt_em_sub(njacks);
        vd_t jZv_em_sub_strong(njacks), jZa_em_sub_strong(njacks), jZp_em_sub_strong(njacks), jZs_em_sub_strong(njacks), jZt_em_sub_strong(njacks);

        vd_t jZ1v_sub(njacks), jZ1a_sub(njacks), jZ1p_sub(njacks), jZ1s_sub(njacks), jZ1t_sub(njacks);
        vd_t jZ1v_em_sub(njacks), jZ1a_em_sub(njacks), jZ1p_em_sub(njacks), jZ1s_em_sub(njacks), jZ1t_em_sub(njacks);
        vd_t jZ1v_em_sub_strong(njacks), jZ1a_em_sub_strong(njacks), jZ1p_em_sub_strong(njacks), jZ1s_em_sub_strong(njacks), jZ1t_em_sub_strong(njacks);

        
        t0=high_resolution_clock::now();
        
#pragma omp parallel for
        for(int ijack=0;ijack<njacks;ijack++)
        {
            //subtraction of O(g^2a^2) effects
            jZq_sub[ijack]=subtract(c_q,jZq_chiral[ijack],p2,p4,g2_tilde);
            jSigma1_sub[ijack]=subtract(c_q,jSigma1_chiral[ijack],p2,p4,g2_tilde);
            
            //subtraction of O(e^2a^2) effects
            jZq_em_sub[ijack]=subtract(c_q_em,jZq_em_chiral[ijack],p2,p4,-3./4.);          //Wilson Action //The coupling gets a - sign due to the definition of S^{-1}(QCD+QED)
            jSigma1_em_sub[ijack]=subtract(c_q_em,jSigma1_em_chiral[ijack],p2,p4,-3./4.);
        }
#pragma omp parallel for
        for(int ijack=0;ijack<njacks;ijack++)
        {
            //subtraction of O(g^2a^2) effects
            jGs_0_sub[ijack]=subtract(c_s,jGs_0_chiral[ijack],p2,p4,g2_tilde); //ZS
            jGv_0_sub[ijack]=subtract(c_a,jGv_0_chiral[ijack],p2,p4,g2_tilde); //ZA
            jGp_0_sub[ijack]=subtract(c_p,jGp_0_chiral[ijack],p2,p4,g2_tilde); //ZP
            jGa_0_sub[ijack]=subtract(c_v,jGa_0_chiral[ijack],p2,p4,g2_tilde); //ZV
            jGt_0_sub[ijack]=subtract(c_t,jGt_0_chiral[ijack],p2,p4,g2_tilde); //ZT
            
            //subtraction of O(e^2a^2) effects
            jGs_em_a_b_sub[ijack]=subtract(c_s_em,jGs_em_a_b_chiral[ijack],p2,p4,3./4.);   ///!!!!!  with Wilson Action
            jGv_em_a_b_sub[ijack]=subtract(c_a_em,jGv_em_a_b_chiral[ijack],p2,p4,3./4.);
            jGp_em_a_b_sub[ijack]=subtract(c_p_em,jGp_em_a_b_chiral[ijack],p2,p4,3./4.);
            jGa_em_a_b_sub[ijack]=subtract(c_v_em,jGa_em_a_b_chiral[ijack],p2,p4,3./4.);
            jGt_em_a_b_sub[ijack]=subtract(c_t_em,jGt_em_a_b_chiral[ijack],p2,p4,3./4.);
        }
#pragma omp parallel for // collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            //subtraction of O(g^2a^2) effects from QCD RCs
            jZs_sub[ijack] = jZq_sub[ijack]/jGs_0_sub[ijack];
            jZa_sub[ijack] = jZq_sub[ijack]/jGv_0_sub[ijack];
            jZp_sub[ijack] = jZq_sub[ijack]/jGp_0_sub[ijack];
            jZv_sub[ijack] = jZq_sub[ijack]/jGa_0_sub[ijack];
            jZt_sub[ijack] = jZq_sub[ijack]/jGt_0_sub[ijack];
            
            jZ1s_sub[ijack] = jSigma1_sub[ijack]/jGs_0_sub[ijack];
            jZ1a_sub[ijack] = jSigma1_sub[ijack]/jGv_0_sub[ijack];
            jZ1p_sub[ijack] = jSigma1_sub[ijack]/jGp_0_sub[ijack];
            jZ1v_sub[ijack] = jSigma1_sub[ijack]/jGa_0_sub[ijack];
            jZ1t_sub[ijack] = jSigma1_sub[ijack]/jGt_0_sub[ijack];
            
            //subtraction of O(e^2a^2) effects from QED RCs
            jZs_em_sub[ijack] = jGs_em_a_b_sub[ijack]/jGs_0_sub[ijack] + jZq_em_sub[ijack]/jZq_sub[ijack];
            jZa_em_sub[ijack] = jGv_em_a_b_sub[ijack]/jGv_0_sub[ijack] + jZq_em_sub[ijack]/jZq_sub[ijack];
            jZp_em_sub[ijack] = jGp_em_a_b_sub[ijack]/jGp_0_sub[ijack] + jZq_em_sub[ijack]/jZq_sub[ijack];
            jZv_em_sub[ijack] = jGa_em_a_b_sub[ijack]/jGa_0_sub[ijack] + jZq_em_sub[ijack]/jZq_sub[ijack];
            jZt_em_sub[ijack] = jGt_em_a_b_sub[ijack]/jGt_0_sub[ijack] + jZq_em_sub[ijack]/jZq_sub[ijack];
            
            jZ1s_em_sub[ijack] = jGs_em_a_b_sub[ijack]/jGs_0_sub[ijack] + jSigma1_em_sub[ijack]/jSigma1_sub[ijack];
            jZ1a_em_sub[ijack] = jGv_em_a_b_sub[ijack]/jGv_0_sub[ijack] + jSigma1_em_sub[ijack]/jSigma1_sub[ijack];
            jZ1p_em_sub[ijack] = jGp_em_a_b_sub[ijack]/jGp_0_sub[ijack] + jSigma1_em_sub[ijack]/jSigma1_sub[ijack];
            jZ1v_em_sub[ijack] = jGa_em_a_b_sub[ijack]/jGa_0_sub[ijack] + jSigma1_em_sub[ijack]/jSigma1_sub[ijack];
            jZ1t_em_sub[ijack] = jGt_em_a_b_sub[ijack]/jGt_0_sub[ijack] + jSigma1_em_sub[ijack]/jSigma1_sub[ijack];
            
            //subtraction of O(g^2a^2) effects from QED RCs
            jZs_em_sub_strong[ijack] = jGs_em_a_b_chiral[ijack]/jGs_0_sub[ijack] + jZq_em_chiral[ijack]/jZq_sub[ijack];
            jZa_em_sub_strong[ijack] = jGv_em_a_b_chiral[ijack]/jGv_0_sub[ijack] + jZq_em_chiral[ijack]/jZq_sub[ijack];
            jZp_em_sub_strong[ijack] = jGp_em_a_b_chiral[ijack]/jGp_0_sub[ijack] + jZq_em_chiral[ijack]/jZq_sub[ijack];
            jZv_em_sub_strong[ijack] = jGa_em_a_b_chiral[ijack]/jGa_0_sub[ijack] + jZq_em_chiral[ijack]/jZq_sub[ijack];
            jZt_em_sub_strong[ijack] = jGt_em_a_b_chiral[ijack]/jGt_0_sub[ijack] + jZq_em_chiral[ijack]/jZq_sub[ijack];
            
            jZ1s_em_sub_strong[ijack] = jGs_em_a_b_chiral[ijack]/jGs_0_sub[ijack] + jSigma1_em_chiral[ijack]/jSigma1_sub[ijack];
            jZ1a_em_sub_strong[ijack] = jGv_em_a_b_chiral[ijack]/jGv_0_sub[ijack] + jSigma1_em_chiral[ijack]/jSigma1_sub[ijack];
            jZ1p_em_sub_strong[ijack] = jGp_em_a_b_chiral[ijack]/jGp_0_sub[ijack] + jSigma1_em_chiral[ijack]/jSigma1_sub[ijack];
            jZ1v_em_sub_strong[ijack] = jGa_em_a_b_chiral[ijack]/jGa_0_sub[ijack] + jSigma1_em_chiral[ijack]/jSigma1_sub[ijack];
            jZ1t_em_sub_strong[ijack] = jGt_em_a_b_chiral[ijack]/jGt_0_sub[ijack] + jSigma1_em_chiral[ijack]/jSigma1_sub[ijack];
        }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Subtraction O(g2a2) and O(e2a2) in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        
        vvd_t jZ_sub(vd_t(0.0,5),njacks), jZ1_sub(vd_t(0.0,5),njacks), jZ_em_sub(vd_t(0.0,5),njacks), jZ1_em_sub(vd_t(0.0,5),njacks), jZ_em_sub_strong(vd_t(0.0,5),njacks),jZ1_em_sub_strong(vd_t(0.0,5),njacks);
        
#pragma omp parallel for
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZ_sub[ijack][0] = jZs_sub[ijack];
            jZ_sub[ijack][1] = jZa_sub[ijack];
            jZ_sub[ijack][2] = jZp_sub[ijack];
            jZ_sub[ijack][3] = jZv_sub[ijack];
            jZ_sub[ijack][4] = jZt_sub[ijack];
            
            jZ1_sub[ijack][0] = jZ1s_sub[ijack];
            jZ1_sub[ijack][1] = jZ1a_sub[ijack];
            jZ1_sub[ijack][2] = jZ1p_sub[ijack];
            jZ1_sub[ijack][3] = jZ1v_sub[ijack];
            jZ1_sub[ijack][4] = jZ1t_sub[ijack];
            
            jZ_em_sub[ijack][0] = jZs_em_sub[ijack];
            jZ_em_sub[ijack][1] = jZa_em_sub[ijack];
            jZ_em_sub[ijack][2] = jZp_em_sub[ijack];
            jZ_em_sub[ijack][3] = jZv_em_sub[ijack];
            jZ_em_sub[ijack][4] = jZt_em_sub[ijack];
            
            jZ1_em_sub[ijack][0] = jZ1s_em_sub[ijack];
            jZ1_em_sub[ijack][1] = jZ1a_em_sub[ijack];
            jZ1_em_sub[ijack][2] = jZ1p_em_sub[ijack];
            jZ1_em_sub[ijack][3] = jZ1v_em_sub[ijack];
            jZ1_em_sub[ijack][4] = jZ1t_em_sub[ijack];
            
            jZ_em_sub_strong[ijack][0] = jZs_em_sub_strong[ijack];
            jZ_em_sub_strong[ijack][1] = jZa_em_sub_strong[ijack];
            jZ_em_sub_strong[ijack][2] = jZp_em_sub_strong[ijack];
            jZ_em_sub_strong[ijack][3] = jZv_em_sub_strong[ijack];
            jZ_em_sub_strong[ijack][4] = jZt_em_sub_strong[ijack];
            
            jZ1_em_sub_strong[ijack][0] = jZ1s_em_sub_strong[ijack];
            jZ1_em_sub_strong[ijack][1] = jZ1a_em_sub_strong[ijack];
            jZ1_em_sub_strong[ijack][2] = jZ1p_em_sub_strong[ijack];
            jZ1_em_sub_strong[ijack][3] = jZ1v_em_sub_strong[ijack];
            jZ1_em_sub_strong[ijack][4] = jZ1t_em_sub_strong[ijack];
        }
        
        
        
        //-------------------------------------------------------------------------------------//
        //---------------  p -> 1/a  evolution ---------------- (from Nuria) ------------------//
        //-------------------------------------------------------------------------------------//
        
        double ainv=2.3; //ok?
        int Nf=2;
        
        vd_t jSigma1_RIp_ainv(0.0,njacks),jSigma1_em_RIp_ainv(0.0,njacks);
        vvd_t jZO_RIp_ainv(vd_t(0.0,5),njacks),jZO_em_RIp_ainv(vd_t(0.0,5),njacks);
        
        vd_t jSigma1_sub_RIp_ainv(0.0,njacks),jSigma1_em_sub_RIp_ainv(0.0,njacks);
        vvd_t jZO_sub_RIp_ainv(vd_t(0.0,5),njacks),jZO_em_sub_RIp_ainv(vd_t(0.0,5),njacks);
        vvd_t jZO_em_sub_strong_RIp_ainv(vd_t(0.0,5),njacks);
        
        double cq;
        vd_t cO(0.0,5);
        
        // Note that ZV  ZA are RGI because they're protected by the WIs
        cq=q_evolution_to_RIp_ainv(Nf,ainv,p2_not_tilde);
        cO[0]=S_evolution_to_RIp_ainv(Nf,ainv,p2_not_tilde); //S
        cO[1]=1.0; //A
        cO[2]=P_evolution_to_RIp_ainv(Nf,ainv,p2_not_tilde); //P
        cO[3]=1.0; //V
        cO[4]=T_evolution_to_RIp_ainv(Nf,ainv,p2_not_tilde); //T
        
        jSigma1_RIp_ainv = jSigma1_chiral/cq;
        jSigma1_em_RIp_ainv = jSigma1_em_chiral/cq;
        
        jSigma1_sub_RIp_ainv = jSigma1_sub/cq;
        jSigma1_em_sub_RIp_ainv = jSigma1_em_sub/cq;
        
        for(int ijack=0;ijack<njacks;ijack++)
            for (int ibil=0; ibil<5; ibil++)
            {
                jZO_RIp_ainv[ijack][ibil]=jZ1_chiral[ijack][ibil]/cO[ibil];           //jZO_RIp_ainv = {S, A, P, V, T}
                jZO_em_RIp_ainv[ijack][ibil]=jZ1_em_chiral[ijack][ibil]/cO[ibil];
                
                jZO_sub_RIp_ainv[ijack][ibil]=jZ1_sub[ijack][ibil]/cO[ibil];
                jZO_em_sub_RIp_ainv[ijack][ibil]=jZ1_em_sub[ijack][ibil]/cO[ibil];
                
                jZO_em_sub_strong_RIp_ainv[ijack][ibil]=jZ1_em_sub_strong[ijack][ibil]/cO[ibil];
            }
        cout<<"ZQ evolver ______"<<endl;
        cout<<p2_not_tilde<<"  "<<q_evolution_to_RIp_ainv(Nf,ainv,p2_not_tilde)<<endl;
        cout<<"ZS evolver ______"<<endl;
        cout<<p2_not_tilde<<"  "<<S_evolution_to_RIp_ainv(Nf,ainv,p2_not_tilde)<<endl;
        cout<<"ZT evolver ______"<<endl;
        cout<<p2_not_tilde<<"  "<<T_evolution_to_RIp_ainv(Nf,ainv,p2_not_tilde)<<endl;
        //-------------------------------------------------------------------------------------//
        //-------------------------------------------------------------------------------------//
        //-------------------------------------------------------------------------------------//
        
        
        //Tag assignment
        t0=high_resolution_clock::now();
        size_t count_no=0;
        for(size_t i=0;i<imom;i++)
        {
            if(abs(p2_vector_allmoms[i]-p2_vector_allmoms[imom])<eps*p2_vector_allmoms[i] &&	\
               abs( abs(mom_list[i][1])*abs(mom_list[i][2])*abs(mom_list[i][3])-(abs(mom_list[imom][1])*abs(mom_list[imom][2])*abs(mom_list[imom][3])))<eps ) //equivalence
            {
                tag_aux=tag_vector[i];
            }else count_no++;
            
            if(count_no==imom)
            {
                tag++;
                tag_vector.push_back(tag);
            }else if(i==imom-1) tag_vector.push_back(tag_aux);
        }
        
        t1=high_resolution_clock::now();
        t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"***** Found equivalent momenta in "<<t_span.count()<<" s ******"<<endl<<endl;
        
        //pushback allmoms-vectors component
#pragma omp parallel for collapse(2)
        for(int ijack=0; ijack<njacks;ijack++)
            for(int mr=0; mr<nmr; mr++)
            {
                jZq_allmoms[imom][ijack][mr]=jZq[ijack][mr];
                jSigma1_allmoms[imom][ijack][mr]=jSigma1[ijack][mr];
                jZq_em_allmoms[imom][ijack][mr]=jZq_em[ijack][mr];
                jSigma1_em_allmoms[imom][ijack][mr]=jSigma1_em[ijack][mr];
            }
#pragma omp parallel for
        for(int ijack=0; ijack<njacks;ijack++)
        {
            jZq_sub_allmoms[imom][ijack]=jZq_sub[ijack];
            jSigma1_sub_allmoms[imom][ijack]=jSigma1_sub[ijack];
            jZq_em_sub_allmoms[imom][ijack]=jZq_em_sub[ijack];
            jSigma1_em_sub_allmoms[imom][ijack]=jSigma1_em_sub[ijack];
        }
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                    for(int k=0;k<5;k++)
                    {
                        jZ_allmoms[imom][ijack][mr_fw][mr_bw][k]=jZ[ijack][mr_fw][mr_bw][k];
                        jZ1_allmoms[imom][ijack][mr_fw][mr_bw][k]=jZ1[ijack][mr_fw][mr_bw][k];
                        jZ_em_allmoms[imom][ijack][mr_fw][mr_bw][k]=jZ_em[ijack][mr_fw][mr_bw][k];
                        jZ1_em_allmoms[imom][ijack][mr_fw][mr_bw][k]=jZ1_em[ijack][mr_fw][mr_bw][k];
                    }
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int k=0;k<5;k++)
            {
                jZ_sub_allmoms[imom][ijack][k]=jZ_sub[ijack][k];
                jZ1_sub_allmoms[imom][ijack][k]=jZ1_sub[ijack][k];
                jZ_em_sub_allmoms[imom][ijack][k]=jZ_em_sub[ijack][k];
                jZ1_em_sub_allmoms[imom][ijack][k]=jZ1_em_sub[ijack][k];
                
                jZ_em_sub_strong_allmoms[imom][ijack][k]=jZ_em_sub_strong[ijack][k];
                jZ1_em_sub_strong_allmoms[imom][ijack][k]=jZ1_em_sub_strong[ijack][k];
            }
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int ieq=0; ieq<neq; ieq++)
            {
                jGp_equivalent_allmoms[imom][ijack][ieq]=jGp_equivalent[ijack][ieq];
                jGs_equivalent_allmoms[imom][ijack][ieq]=jGs_equivalent[ijack][ieq];
                jGp_subpole_allmoms[imom][ijack][ieq]=jGp_subpole[ijack][ieq];
                jGs_subpole_allmoms[imom][ijack][ieq]=jGs_subpole[ijack][ieq];
                
                jGv_equivalent_allmoms[imom][ijack][ieq]=jGv_equivalent[ijack][ieq];
                jGa_equivalent_allmoms[imom][ijack][ieq]=jGa_equivalent[ijack][ieq];
                jGt_equivalent_allmoms[imom][ijack][ieq]=jGt_equivalent[ijack][ieq];
                
                jGp_em_equivalent_allmoms[imom][ijack][ieq]=jGp_em_equivalent[ijack][ieq];
                jGs_em_equivalent_allmoms[imom][ijack][ieq]=jGs_em_equivalent[ijack][ieq];
                jGp_em_subpole_allmoms[imom][ijack][ieq]=jGp_em_subpole[ijack][ieq];
                jGs_em_subpole_allmoms[imom][ijack][ieq]=jGs_em_subpole[ijack][ieq];
                
                jGv_em_equivalent_allmoms[imom][ijack][ieq]=jGv_em_equivalent[ijack][ieq];
                jGa_em_equivalent_allmoms[imom][ijack][ieq]=jGa_em_equivalent[ijack][ieq];
                jGt_em_equivalent_allmoms[imom][ijack][ieq]=jGt_em_equivalent[ijack][ieq];
            }
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int ieq=0; ieq<neq2; ieq++)
            {
                jZq_equivalent_allmoms[imom][ijack][ieq]=jZq_equivalent[ijack][ieq];
                jSigma1_equivalent_allmoms[imom][ijack][ieq]=jSigma1_equivalent[ijack][ieq];
                jZq_em_equivalent_allmoms[imom][ijack][ieq]=jZq_em_equivalent[ijack][ieq];
                jSigma1_em_equivalent_allmoms[imom][ijack][ieq]=jSigma1_em_equivalent[ijack][ieq];
            }
#pragma omp parallel for
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jGp_0_chiral_allmoms[imom][ijack]=jGp_0_chiral[ijack];
            jGv_0_chiral_allmoms[imom][ijack]=jGv_0_chiral[ijack];
            jGs_0_chiral_allmoms[imom][ijack]=jGs_0_chiral[ijack];
            jGa_0_chiral_allmoms[imom][ijack]=jGa_0_chiral[ijack];
            jGt_0_chiral_allmoms[imom][ijack]=jGt_0_chiral[ijack];
            
            jGp_0_sub_allmoms[imom][ijack]=jGp_0_sub[ijack];
            jGv_0_sub_allmoms[imom][ijack]=jGv_0_sub[ijack];
            jGs_0_sub_allmoms[imom][ijack]=jGs_0_sub[ijack];
            jGa_0_sub_allmoms[imom][ijack]=jGa_0_sub[ijack];
            jGt_0_sub_allmoms[imom][ijack]=jGt_0_sub[ijack];
            
            jGp_em_a_b_chiral_allmoms[imom][ijack]=jGp_em_a_b_chiral[ijack];
            jGv_em_a_b_chiral_allmoms[imom][ijack]=jGv_em_a_b_chiral[ijack];
            jGs_em_a_b_chiral_allmoms[imom][ijack]=jGs_em_a_b_chiral[ijack];
            jGa_em_a_b_chiral_allmoms[imom][ijack]=jGa_em_a_b_chiral[ijack];
            jGt_em_a_b_chiral_allmoms[imom][ijack]=jGt_em_a_b_chiral[ijack];
            
            jGp_em_a_b_sub_allmoms[imom][ijack]=jGp_em_a_b_sub[ijack];
            jGv_em_a_b_sub_allmoms[imom][ijack]=jGv_em_a_b_sub[ijack];
            jGs_em_a_b_sub_allmoms[imom][ijack]=jGs_em_a_b_sub[ijack];
            jGa_em_a_b_sub_allmoms[imom][ijack]=jGa_em_a_b_sub[ijack];
            jGt_em_a_b_sub_allmoms[imom][ijack]=jGt_em_a_b_sub[ijack];
            
            jZq_chiral_allmoms[imom][ijack]=jZq_chiral[ijack];
            jSigma1_chiral_allmoms[imom][ijack]=jSigma1_chiral[ijack];
            jZ_chiral_allmoms[imom][ijack]=jZ_chiral[ijack];
            jZ1_chiral_allmoms[imom][ijack]=jZ1_chiral[ijack];
            
            jZq_em_chiral_allmoms[imom][ijack]=jZq_em_chiral[ijack];
            jSigma1_em_chiral_allmoms[imom][ijack]=jSigma1_em_chiral[ijack];
            jZ_em_chiral_allmoms[imom][ijack]=jZ_em_chiral[ijack];
            jZ1_em_chiral_allmoms[imom][ijack]=jZ1_em_chiral[ijack];
            
            jSigma1_RIp_ainv_allmoms[imom][ijack]=jSigma1_RIp_ainv[ijack];
            jSigma1_em_RIp_ainv_allmoms[imom][ijack]=jSigma1_em_RIp_ainv[ijack];
            
            jZO_RIp_ainv_allmoms[imom][ijack]=jZO_RIp_ainv[ijack];
            jZO_em_RIp_ainv_allmoms[imom][ijack]=jZO_em_RIp_ainv[ijack];
            
            jSigma1_sub_RIp_ainv_allmoms[imom][ijack]=jSigma1_sub_RIp_ainv[ijack];
            jSigma1_em_sub_RIp_ainv_allmoms[imom][ijack]=jSigma1_em_sub_RIp_ainv[ijack];
            
            jZO_sub_RIp_ainv_allmoms[imom][ijack]=jZO_sub_RIp_ainv[ijack];
            jZO_em_sub_RIp_ainv_allmoms[imom][ijack]=jZO_em_sub_RIp_ainv[ijack];
            jZO_em_sub_strong_RIp_ainv_allmoms[imom][ijack]=jZO_em_sub_strong_RIp_ainv[ijack];
        }
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(size_t k=0; k<3; k++)
            {
                jGp_pars_allmoms[imom][ijack](k)=jGp_pars[ijack](k);
                jGs_pars_allmoms[imom][ijack](k)=jGs_pars[ijack](k);
                jGp_em_pars_allmoms[imom][ijack](k)=jGp_em_pars[ijack](k);
                jGs_em_pars_allmoms[imom][ijack](k)=jGs_em_pars[ijack](k);
            }
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(size_t k=0; k<2; k++)
            {
                jGv_pars_allmoms[imom][ijack](k)=jGv_pars[ijack](k);
                jGa_pars_allmoms[imom][ijack](k)=jGa_pars[ijack](k);
                jGt_pars_allmoms[imom][ijack](k)=jGt_pars[ijack](k);
                jGv_em_pars_allmoms[imom][ijack](k)=jGv_em_pars[ijack](k);
                jGa_em_pars_allmoms[imom][ijack](k)=jGa_em_pars[ijack](k);
                jGt_em_pars_allmoms[imom][ijack](k)=jGt_em_pars[ijack](k);
                
                jZq_pars_allmoms[imom][ijack](k)=jZq_pars[ijack](k);
                jSigma1_pars_allmoms[imom][ijack](k)=jSigma1_pars[ijack](k);
                jZq_em_pars_allmoms[imom][ijack](k)=jZq_em_pars[ijack](k);
                jSigma1_em_pars_allmoms[imom][ijack](k)=jSigma1_em_pars[ijack](k);
            }
    }//moms loop
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    cout<<endl;
    cout<<"***Exit loop on momenta***"<<endl;
    cout<<endl;
    
    t0=high_resolution_clock::now();
    
    int neq_moms = tag+1;
    
    vector<int> count_tag_vector(neq_moms);
    int count=0;
    for(int tag=0;tag<neq_moms;tag++)
    {
        count=0;
        for(int imom=0;imom<moms;imom++)
        {
            if(tag_vector[imom]==tag) count++;
        }
        count_tag_vector[tag]=count;
    }
    
    vector<double> p2_vector_eqmoms(neq_moms);
    for(int tag=0;tag<neq_moms;tag++)
        for(int imom=0;imom<moms;imom++)
        {
            if(tag_vector[imom]==tag)  p2_vector_eqmoms[tag] = p2_vector_allmoms[imom];
        }
    
    
    //Vector of interesting quantities (EQUIVALENT MOMS)
    vector<jZ_t> jZq_eqmoms(neq_moms,vvd_t(vd_t(0.0,nmr),njacks)), jSigma1_eqmoms(neq_moms,vvd_t(vd_t(0.0,nmr),njacks)), \
    jZq_em_eqmoms(neq_moms,vvd_t(vd_t(0.0,nmr),njacks)), jSigma1_em_eqmoms(neq_moms,vvd_t(vd_t(0.0,nmr),njacks));
    
    vector<jZbil_t> jZ_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks)), jZ1_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks)), \
    jZ_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks)), jZ1_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks));
    
    vector<vd_t> jZq_sub_eqmoms(neq_moms,vd_t(0.0,njacks)), jSigma1_sub_eqmoms(neq_moms,vd_t(0.0,njacks)), \
    jZq_em_sub_eqmoms(neq_moms,vd_t(0.0,njacks)), jSigma1_em_sub_eqmoms(neq_moms,vd_t(0.0,njacks));
    
    vector<vvd_t> jZ_sub_eqmoms(neq_moms,vvd_t(vd_t(0.0,5),njacks)), jZ1_sub_eqmoms(neq_moms,vvd_t(vd_t(0.0,5),njacks)), \
    jZ_em_sub_eqmoms(neq_moms,vvd_t(vd_t(0.0,5),njacks)), jZ1_em_sub_eqmoms(neq_moms,vvd_t(vd_t(0.0,5),njacks)), jZ_em_sub_strong_eqmoms(neq_moms,vvd_t(vd_t(0.0,5),njacks)), jZ1_em_sub_strong_eqmoms(neq_moms,vvd_t(vd_t(0.0,5),njacks));
    
    vector<vvd_t> jGp_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)),\
    jGp_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));
    vector<vvd_t> jGp_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), \
    jGp_em_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_em_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));
    
    vector<vvd_t> jGv_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGa_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)),\
    jGt_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));
    vector<vvd_t> jGv_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGa_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)),\
    jGt_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));
    
    vector<vvd_t> jZq_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks)), jSigma1_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks)),\
    jZq_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks)), jSigma1_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks));
    
    vector<vd_t> jGp_0_chiral_eqmoms(neq_moms,vd_t(njacks)),jGa_0_chiral_eqmoms(neq_moms,vd_t(njacks)),jGv_0_chiral_eqmoms(neq_moms,vd_t(njacks)),\
    jGs_0_chiral_eqmoms(neq_moms,vd_t(njacks)),jGt_0_chiral_eqmoms(neq_moms,vd_t(njacks));
    vector<vd_t> jGp_0_sub_eqmoms(neq_moms,vd_t(njacks)),jGa_0_sub_eqmoms(neq_moms,vd_t(njacks)),jGv_0_sub_eqmoms(neq_moms,vd_t(njacks)),\
    jGs_0_sub_eqmoms(neq_moms,vd_t(njacks)),jGt_0_sub_eqmoms(neq_moms,vd_t(njacks));

    vector<vd_t> jGp_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)),jGa_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)),jGv_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)), \
    jGs_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)),jGt_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks));
    vector<vd_t> jGp_em_a_b_sub_eqmoms(neq_moms,vd_t(njacks)),jGa_em_a_b_sub_eqmoms(neq_moms,vd_t(njacks)),jGv_em_a_b_sub_eqmoms(neq_moms,vd_t(njacks)), \
    jGs_em_a_b_sub_eqmoms(neq_moms,vd_t(njacks)),jGt_em_a_b_sub_eqmoms(neq_moms,vd_t(njacks));
    
    vector<vd_t> jZq_chiral_eqmoms(neq_moms,vd_t(0.0,njacks)),jSigma1_chiral_eqmoms(neq_moms,vd_t(0.0,njacks));
    vector<vd_t> jZq_em_chiral_eqmoms(neq_moms,vd_t(njacks)),jSigma1_em_chiral_eqmoms(neq_moms,vd_t(njacks));
    vector<vvd_t> jZ_chiral_eqmoms(neq_moms,vvd_t(vd_t(5),njacks)),jZ1_chiral_eqmoms(neq_moms,vvd_t(vd_t(5),njacks));
    vector<vvd_t> jZ_em_chiral_eqmoms(neq_moms,vvd_t(vd_t(5),njacks)),jZ1_em_chiral_eqmoms(neq_moms,vvd_t(vd_t(5),njacks));
    
    vector< vXd_t > jGp_pars_eqmoms(neq_moms,vXd_t(VectorXd(3),njacks)), jGs_pars_eqmoms(neq_moms,vXd_t(VectorXd(3),njacks)), \
    jGp_em_pars_eqmoms(neq_moms,vXd_t(VectorXd(3),njacks)), jGs_em_pars_eqmoms(neq_moms,vXd_t(VectorXd(3),njacks));
    vector< vXd_t > jGv_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)), jGa_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)),\
    jGt_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)), jGv_em_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)),\
    jGa_em_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)), jGt_em_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks));
    vector< vXd_t > jZq_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)), jSigma1_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)),\
    jZq_em_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks)), jSigma1_em_pars_eqmoms(neq_moms,vXd_t(VectorXd(2),njacks));
    
    vector<vd_t> jSigma1_RIp_ainv_eqmoms(neq_moms,vd_t(0.0,njacks)),jSigma1_em_RIp_ainv_eqmoms(neq_moms,vd_t(0.0,njacks));
    vector<vvd_t> jZO_RIp_ainv_eqmoms(neq_moms,vvd_t(vd_t(5),njacks)),jZO_em_RIp_ainv_eqmoms(neq_moms,vvd_t(vd_t(5),njacks));
    
    vector<vd_t> jSigma1_sub_RIp_ainv_eqmoms(neq_moms,vd_t(0.0,njacks)),jSigma1_em_sub_RIp_ainv_eqmoms(neq_moms,vd_t(0.0,njacks));
    vector<vvd_t> jZO_sub_RIp_ainv_eqmoms(neq_moms,vvd_t(vd_t(5),njacks)),jZO_em_sub_RIp_ainv_eqmoms(neq_moms,vvd_t(vd_t(5),njacks));
    vector<vvd_t> jZO_em_sub_strong_RIp_ainv_eqmoms(neq_moms,vvd_t(vd_t(5),njacks));
    
    
    for(int tag=0;tag<neq_moms;tag++)
        for(int imom=0;imom<moms;imom++)
        {
            if(tag_vector[imom]==tag)
            {
#pragma omp parallel for collapse(2)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr=0;mr<nmr;mr++)
                    {
                        jZq_eqmoms[tag][ijack][mr] += jZq_allmoms[imom][ijack][mr] / count_tag_vector[tag];
                        jSigma1_eqmoms[tag][ijack][mr] += jSigma1_allmoms[imom][ijack][mr] / count_tag_vector[tag];
                        jZq_em_eqmoms[tag][ijack][mr] += jZq_em_allmoms[imom][ijack][mr] / count_tag_vector[tag];
                        jSigma1_em_eqmoms[tag][ijack][mr] += jSigma1_em_allmoms[imom][ijack][mr] / count_tag_vector[tag];
                    }
#pragma omp parallel for
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    jZq_sub_eqmoms[tag][ijack] += jZq_sub_allmoms[imom][ijack] / count_tag_vector[tag];
                    jSigma1_sub_eqmoms[tag][ijack] += jSigma1_sub_allmoms[imom][ijack] / count_tag_vector[tag];
                    jZq_em_sub_eqmoms[tag][ijack] += jZq_em_sub_allmoms[imom][ijack] / count_tag_vector[tag];
                    jSigma1_em_sub_eqmoms[tag][ijack] += jSigma1_em_sub_allmoms[imom][ijack] / count_tag_vector[tag];
                }
#pragma omp parallel for collapse(4)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mrA=0;mrA<nmr;mrA++)
                        for(int mrB=0;mrB<nmr;mrB++)
                            for(int i=0;i<5;i++)
                            {
                                jZ_eqmoms[tag][ijack][mrA][mrB][i] += jZ_allmoms[imom][ijack][mrA][mrB][i] / count_tag_vector[tag];
                                jZ1_eqmoms[tag][ijack][mrA][mrB][i] += jZ1_allmoms[imom][ijack][mrA][mrB][i] / count_tag_vector[tag];
                                jZ_em_eqmoms[tag][ijack][mrA][mrB][i] += jZ_em_allmoms[imom][ijack][mrA][mrB][i] / count_tag_vector[tag];
                                jZ1_em_eqmoms[tag][ijack][mrA][mrB][i] += jZ1_em_allmoms[imom][ijack][mrA][mrB][i] / count_tag_vector[tag];
                            }
#pragma omp parallel for collapse(2)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int i=0;i<5;i++)
                    {
                        jZ_sub_eqmoms[tag][ijack][i] += jZ_sub_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZ1_sub_eqmoms[tag][ijack][i] += jZ1_sub_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZ_em_sub_eqmoms[tag][ijack][i] += jZ_em_sub_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZ1_em_sub_eqmoms[tag][ijack][i] += jZ1_em_sub_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        
                        jZ_em_sub_strong_eqmoms[tag][ijack][i] += jZ_em_sub_strong_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZ1_em_sub_strong_eqmoms[tag][ijack][i] += jZ1_em_sub_strong_allmoms[imom][ijack][i] / count_tag_vector[tag];

                    }
#pragma omp parallel for
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    jGp_0_chiral_eqmoms[tag][ijack] += jGp_0_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGa_0_chiral_eqmoms[tag][ijack] += jGa_0_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGv_0_chiral_eqmoms[tag][ijack] += jGv_0_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGs_0_chiral_eqmoms[tag][ijack] += jGs_0_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGt_0_chiral_eqmoms[tag][ijack] += jGt_0_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jZq_chiral_eqmoms[tag][ijack] += jZq_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jSigma1_chiral_eqmoms[tag][ijack] += jSigma1_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    
                    jSigma1_RIp_ainv_eqmoms[tag][ijack] += jSigma1_RIp_ainv_allmoms[imom][ijack]/ count_tag_vector[tag];
                    jSigma1_em_RIp_ainv_eqmoms[tag][ijack] += jSigma1_em_RIp_ainv_allmoms[imom][ijack]/ count_tag_vector[tag];
                   
                    jSigma1_sub_RIp_ainv_eqmoms[tag][ijack] += jSigma1_sub_RIp_ainv_allmoms[imom][ijack]/ count_tag_vector[tag];
                    jSigma1_em_sub_RIp_ainv_eqmoms[tag][ijack] += jSigma1_em_sub_RIp_ainv_allmoms[imom][ijack]/ count_tag_vector[tag];

                    jGp_em_a_b_chiral_eqmoms[tag][ijack] += jGp_em_a_b_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGa_em_a_b_chiral_eqmoms[tag][ijack] += jGa_em_a_b_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGv_em_a_b_chiral_eqmoms[tag][ijack] += jGv_em_a_b_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGs_em_a_b_chiral_eqmoms[tag][ijack] += jGs_em_a_b_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jGt_em_a_b_chiral_eqmoms[tag][ijack] += jGt_em_a_b_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jZq_em_chiral_eqmoms[tag][ijack] += jZq_em_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                    jSigma1_em_chiral_eqmoms[tag][ijack] += jSigma1_em_chiral_allmoms[imom][ijack] / count_tag_vector[tag];
                }
#pragma omp parallel for collapse(2)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int ieq=0;ieq<neq;ieq++)
                    {
                        jGp_equivalent_eqmoms[tag][ijack][ieq] += jGp_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGs_equivalent_eqmoms[tag][ijack][ieq] += jGs_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGp_subpole_eqmoms[tag][ijack][ieq] += jGp_subpole_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGs_subpole_eqmoms[tag][ijack][ieq] += jGs_subpole_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        
                        jGp_em_equivalent_eqmoms[tag][ijack][ieq] += jGp_em_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGs_em_equivalent_eqmoms[tag][ijack][ieq] += jGs_em_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGp_em_subpole_eqmoms[tag][ijack][ieq] += jGp_em_subpole_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGs_em_subpole_eqmoms[tag][ijack][ieq] += jGs_em_subpole_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        
                        jGv_equivalent_eqmoms[tag][ijack][ieq] += jGv_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGa_equivalent_eqmoms[tag][ijack][ieq] += jGa_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGt_equivalent_eqmoms[tag][ijack][ieq] += jGt_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        
                        jGv_em_equivalent_eqmoms[tag][ijack][ieq] += jGv_em_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGa_em_equivalent_eqmoms[tag][ijack][ieq] += jGa_em_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jGt_em_equivalent_eqmoms[tag][ijack][ieq] += jGt_em_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                    }
#pragma omp parallel for collapse(2)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int ieq=0;ieq<neq2;ieq++)
                    {
                        jZq_equivalent_eqmoms[tag][ijack][ieq] += jZq_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jSigma1_equivalent_eqmoms[tag][ijack][ieq] += jSigma1_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        
                        jZq_em_equivalent_eqmoms[tag][ijack][ieq] += jZq_em_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                        jSigma1_em_equivalent_eqmoms[tag][ijack][ieq] += jSigma1_em_equivalent_allmoms[imom][ijack][ieq] / count_tag_vector[tag];
                    }
#pragma omp parallel for collapse(2)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int i=0;i<5;i++)
                    {
                        jZ_chiral_eqmoms[tag][ijack][i] += jZ_chiral_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZ1_chiral_eqmoms[tag][ijack][i] += jZ1_chiral_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZ_em_chiral_eqmoms[tag][ijack][i] += jZ_em_chiral_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZ1_em_chiral_eqmoms[tag][ijack][i] += jZ1_em_chiral_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        
                        jZO_RIp_ainv_eqmoms[tag][ijack][i] += jZO_RIp_ainv_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZO_em_RIp_ainv_eqmoms[tag][ijack][i] += jZO_em_RIp_ainv_allmoms[imom][ijack][i] / count_tag_vector[tag];
                       
                        jZO_sub_RIp_ainv_eqmoms[tag][ijack][i] += jZO_sub_RIp_ainv_allmoms[imom][ijack][i] / count_tag_vector[tag];
                        jZO_em_sub_RIp_ainv_eqmoms[tag][ijack][i] += jZO_em_sub_RIp_ainv_allmoms[imom][ijack][i] / count_tag_vector[tag];

                        jZO_em_sub_strong_RIp_ainv_eqmoms[tag][ijack][i] += jZO_em_sub_strong_RIp_ainv_allmoms[imom][ijack][i] / count_tag_vector[tag];
                    }
                
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int k=0;k<3;k++)
                    {
                        jGp_pars_eqmoms[tag][ijack](k) += jGp_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGp_em_pars_eqmoms[tag][ijack](k) += jGp_em_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGs_pars_eqmoms[tag][ijack](k) += jGs_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGs_em_pars_eqmoms[tag][ijack](k) += jGs_em_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                    }
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int k=0;k<2;k++)
                    {
                        jGv_pars_eqmoms[tag][ijack](k) += jGv_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGv_em_pars_eqmoms[tag][ijack](k) += jGv_em_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGa_pars_eqmoms[tag][ijack](k) += jGa_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGa_em_pars_eqmoms[tag][ijack](k) += jGa_em_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGt_pars_eqmoms[tag][ijack](k) += jGt_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jGt_em_pars_eqmoms[tag][ijack](k) += jGt_em_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jZq_pars_eqmoms[tag][ijack](k) += jZq_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jZq_em_pars_eqmoms[tag][ijack](k) += jZq_em_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jSigma1_pars_eqmoms[tag][ijack](k) += jSigma1_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                        jSigma1_em_pars_eqmoms[tag][ijack](k) += jSigma1_em_pars_allmoms[imom][ijack](k) / count_tag_vector[tag];
                    }
            }
        }
    
    t1=high_resolution_clock::now();
    t_span = duration_cast<duration<double>>(t1-t0);
    cout<<"***** Computed averages over equivalent momenta in "<<t_span.count()<<" s ******"<<endl<<endl;
    
#define PRINT(NAME)				\
print_vec(NAME##_##allmoms,"allmoms/"#NAME);	\
print_vec(NAME##_##eqmoms,"eqmoms/"#NAME)
    
    PRINT(p2_vector);
    PRINT(jZq);
    PRINT(jSigma1);
    PRINT(jZq_em);
    PRINT(jSigma1_em);
    PRINT(jZ);
    PRINT(jZ1);
    PRINT(jZ_em);
    PRINT(jZ1_em);
    PRINT(jZq_sub);
    PRINT(jSigma1_sub);
    PRINT(jZq_em_sub);
    PRINT(jSigma1_em_sub);
    PRINT(jZ_sub);
    PRINT(jZ1_sub);
    PRINT(jZ_em_sub);
    PRINT(jZ1_em_sub);
    PRINT(jZ_em_sub_strong);
    PRINT(jZ1_em_sub_strong);
    PRINT(jGp_equivalent);
    PRINT(jGs_equivalent);
    PRINT(jGp_subpole);
    PRINT(jGs_subpole);
    PRINT(jGv_equivalent);
    PRINT(jGa_equivalent);
    PRINT(jGt_equivalent);
    PRINT(jGp_em_equivalent);
    PRINT(jGs_em_equivalent);
    PRINT(jGp_em_subpole);
    PRINT(jGs_em_subpole);
    PRINT(jGv_em_equivalent);
    PRINT(jGa_em_equivalent);
    PRINT(jGt_em_equivalent);
    PRINT(jZq_equivalent);
    PRINT(jSigma1_equivalent);
    PRINT(jZq_em_equivalent);
    PRINT(jSigma1_em_equivalent);
    PRINT(jGp_0_chiral);
    PRINT(jGa_0_chiral);
    PRINT(jGv_0_chiral);
    PRINT(jGs_0_chiral);
    PRINT(jGt_0_chiral);
    PRINT(jGp_em_a_b_chiral);
    PRINT(jGa_em_a_b_chiral);
    PRINT(jGv_em_a_b_chiral);
    PRINT(jGs_em_a_b_chiral);
    PRINT(jGt_em_a_b_chiral);
    PRINT(jZq_chiral);
    PRINT(jSigma1_chiral);
    PRINT(jZq_em_chiral);
    PRINT(jSigma1_em_chiral);
    PRINT(jZ_chiral);
    PRINT(jZ1_chiral);
    PRINT(jZ_em_chiral);
    PRINT(jZ1_em_chiral);
    PRINT(jGp_pars);
    PRINT(jGp_em_pars);
    PRINT(jGs_pars);
    PRINT(jGs_em_pars);
    PRINT(jGv_pars);
    PRINT(jGv_em_pars);
    PRINT(jGa_pars);
    PRINT(jGa_em_pars);
    PRINT(jGt_pars);
    PRINT(jGt_em_pars);
    PRINT(jZq_pars);
    PRINT(jZq_em_pars);
    PRINT(jSigma1_pars);
    PRINT(jSigma1_em_pars);
    PRINT(jSigma1_RIp_ainv);
    PRINT(jSigma1_em_RIp_ainv);
    PRINT(jZO_RIp_ainv);
    PRINT(jZO_em_RIp_ainv);
    PRINT(jSigma1_sub_RIp_ainv);
    PRINT(jSigma1_em_sub_RIp_ainv);
    PRINT(jZO_sub_RIp_ainv);
    PRINT(jZO_em_sub_RIp_ainv);
    PRINT(jZO_em_sub_strong_RIp_ainv);

#undef PRINT
    
    print_vec(m_eff_equivalent,"allmoms/m_eff_equivalent");
    print_vec(m_eff_equivalent,"eqmoms/m_eff_equivalent");
    print_vec(m_eff_equivalent_Zq,"allmoms/m_eff_equivalent_Zq");
    print_vec(m_eff_equivalent_Zq,"eqmoms/m_eff_equivalent_Zq");
    
    
    
    cout<<endl<<endl;
    cout<<"---------------------------------------------"<<endl;
    high_resolution_clock::time_point t_END=high_resolution_clock::now();
    t_span = duration_cast<duration<double>>(t_END-t_START);
    cout<<"***** TOTAL TIME:  "<<t_span.count()<<" s ******"<<endl<<endl;
    
    cout<<endl;
    cout<<"ZP chiral ALLMOMS jackknife per jackknife - mom 0"<<endl;
    for(int ijack=0;ijack<njacks;ijack++)
        cout<<"ijack: "<<ijack<<"  Z1: "<<jZ1_chiral_allmoms[0][ijack][2]<<"  "<<jSigma1_chiral_allmoms[0][ijack]<<"  "<<jGp_0_chiral_allmoms[0][ijack]<<endl;
    cout<<endl;
    cout<<"SIGMA1 SUB and CHIRAL"<<endl;
    for(int ijack=0;ijack<njacks;ijack++)
        cout<<jSigma1_chiral_allmoms[0][ijack]<<"  "<<jSigma1_sub_allmoms[0][ijack]<<endl;
    
    
    
    
    vvd_t Gp_0_chiral_allmoms = average_Zq_chiral(jGp_0_chiral_allmoms);  //Zq[ave/err][imom]
    vvd_t Gs_0_chiral_allmoms = average_Zq_chiral(jGs_0_chiral_allmoms);
    vvd_t Gv_0_chiral_allmoms = average_Zq_chiral(jGv_0_chiral_allmoms);
    vvd_t Ga_0_chiral_allmoms = average_Zq_chiral(jGa_0_chiral_allmoms);
    vvd_t Gt_0_chiral_allmoms = average_Zq_chiral(jGt_0_chiral_allmoms);
    
    vvd_t Gp_0_sub_allmoms = average_Zq_chiral(jGp_0_sub_allmoms);  //Zq[ave/err][imom]
    vvd_t Gs_0_sub_allmoms = average_Zq_chiral(jGs_0_sub_allmoms);
    vvd_t Gv_0_sub_allmoms = average_Zq_chiral(jGv_0_sub_allmoms);
    vvd_t Ga_0_sub_allmoms = average_Zq_chiral(jGa_0_sub_allmoms);
    vvd_t Gt_0_sub_allmoms = average_Zq_chiral(jGt_0_sub_allmoms);
    
//    cout<<endl;
//    cout<<"ALLMOMS:  Gp_0_chiral"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Gp_0_chiral_allmoms[0][imom]<<"  "<<Gp_0_chiral_allmoms[1][imom]<<endl;
//    cout<<"ALLMOMS:  Gp_0_sub"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Gp_0_sub_allmoms[0][imom]<<"  "<<Gp_0_sub_allmoms[1][imom]<<endl;
//    
//    cout<<endl;
//    cout<<"ALLMOMS:  Gv_0_chiral"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Gv_0_chiral_allmoms[0][imom]<<"  "<<Gv_0_chiral_allmoms[1][imom]<<endl;
//    cout<<"ALLMOMS:  Gv_0_sub"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Gv_0_sub_allmoms[0][imom]<<"  "<<Gv_0_sub_allmoms[1][imom]<<endl;
//
//    cout<<endl;
//    cout<<"ALLMOMS:  Ga_0_chiral"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Ga_0_chiral_allmoms[0][imom]<<"  "<<Ga_0_chiral_allmoms[1][imom]<<endl;
//    cout<<"ALLMOMS:  Ga_0_sub"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Ga_0_sub_allmoms[0][imom]<<"  "<<Ga_0_sub_allmoms[1][imom]<<endl;
//    
//    cout<<endl;
//    cout<<"ALLMOMS:  Gt_0_chiral"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Gt_0_chiral_allmoms[0][imom]<<"  "<<Gt_0_chiral_allmoms[1][imom]<<endl;
//    cout<<"ALLMOMS:  Gt_0_sub"<<endl;
//    for(int imom=0; imom<moms; imom++)
//        cout<<p2_vector_allmoms[imom]<<" "<<Gt_0_sub_allmoms[0][imom]<<"  "<<Gt_0_sub_allmoms[1][imom]<<endl;

  
    
    return 0;
    
}
