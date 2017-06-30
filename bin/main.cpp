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

using namespace std;
using namespace Eigen;

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

//valarray of complex double
using vdcompl_t=valarray<dcompl>;
using vvdcompl_t=valarray< vdcompl_t >;
using vvvdcompl_t=valarray< vvdcompl_t >;
using vvvvdcompl_t=valarray< vvvdcompl_t >;

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
string path_to_conf(int i_conf,const string &name)
{
  char path[1024];
  sprintf(path,"out/%04d/fft_%s",i_conf,name.c_str());
  return path;
}

//returns the linearized spin color index
size_t isc(size_t is,size_t ic)
{return ic+3*is;}

//read a propagator file
prop_t read_prop(ifstream &input, const string &path)
{
  prop_t out(prop_t::Zero());

  //for(size_t imom=0;imom<mom_list.size();imom++)
  for(int id_so=0;id_so<4;id_so++)
    for(int ic_so=0;ic_so<3;ic_so++)
      for(int id_si=0;id_si<4;id_si++)
	for(int ic_si=0;ic_si<3;ic_si++)
	  {
	    double temp[2];
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
vvvprop_t make_vertex(const vprop_t &prop1, const vprop_t &prop2,const vprop_t &gamma)
{

  int nmr=prop1.size(); 
  vvvprop_t vert(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr);
  
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      for(int mu=0;mu<16;mu++)
	{      
	  vert[mr_fw][mr_bw][mu]=prop1[mr_fw]*gamma[mu]*gamma[5]*prop2[mr_bw].adjoint()*gamma[5];  /*it has to be "jackknifed"*/
	}
  return vert;
}



//create the path-string to the contraction
string path_to_contr(int i_conf,const int mr1, const string &T1, const int mr2, const string &T2)
{

  int r1 = mr1%nr;
  int m1 = (mr1-r1)/nr;
  int r2 = mr2%nr;
  int m2 = (mr2-r2)/nr;
  
  char path[1024];
  sprintf(path,"out/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());

  // cout<<path<<endl;
  
  return path;
}

//jackknife Propagator
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
jprop_t jackknife_prop(jprop_t &jS, int nconf, int clust_size, size_t nhits)
{
  int nmr = jS[0].size();
  valarray<prop_t> jSum(prop_t::Zero(),nmr);

  //sum of jS
  for(size_t j=0;j<jS.size();j++) jSum+= jS[j];
  //jackknife fluctuation
  for(size_t j=0;j<jS.size();j++)
    {
      jS[j]=jSum-jS[j];
      for(auto &it : jS[j])
	it/=(nconf-clust_size)/nhits;
    }

  return jS;
}

//jackknife Vertex
jvert_t jackknife_vertex(jvert_t &jVert, int nconf, int clust_size, size_t nhits)
{
  int nmr = jVert[0].size();
  vert_t jSum(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr);
  
  //sum of the jVert
  for(size_t j=0;j<jVert.size();j++) jSum+= jVert[j];
  //jackknife fluctuation
  for(size_t j=0;j<jVert.size();j++)
    {
      jVert[j]=jSum-jVert[j];
      for(auto &it : jVert[j])
	for(auto &jt : it)
	  for(auto &kt : jt)
	    kt/=(nconf-clust_size)*nhits;
    }
  
  return jVert;
}

//invert the propagator
jprop_t invert_jprop( const jprop_t &jprop){
  
  int njacks=jprop.size();
  int nmr=jprop[0].size();

  jprop_t jprop_inv(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
  
  for(int ijack=0;ijack<njacks;ijack++)
    for(int mr=0;mr<nmr;mr++)
      jprop_inv[ijack][mr]=jprop[ijack][mr].inverse();
  
  return jprop_inv;
}

//amputate external legs

jvert_t amputate( const jprop_t  &jprop1_inv, const jvert_t &jV, const jprop_t  &jprop2_inv, vprop_t GAMMA){

  int njacks = jV.size();
  int nmr = jV[0].size();
  jvert_t jLambda(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
  
  for(int ijack=0;ijack<njacks;ijack++)
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
      for(int mr_bw=0;mr_bw<nmr;mr_bw++)
	for(int igam=0;igam<16;igam++)
	  jLambda[ijack][mr_fw][mr_bw][igam]=jprop1_inv[ijack][mr_fw]*jV[ijack][mr_fw][mr_bw][igam]*GAMMA[5]*jprop2_inv[ijack][mr_bw].adjoint()*GAMMA[5];
  
  return jLambda;
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
  double p2=0.0;
  
  vvdcompl_t jZq(vdcompl_t(nmr),njacks);
  vvd_t jZq_real(vd_t(nmr),njacks);
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
    p2+=p_tilde[coord]*p_tilde[coord];
  
  for(int ijack=0;ijack<njacks;ijack++)
    for(int mr=0;mr<nmr;mr++)
      {
	jZq[ijack][mr]=-I*((p_slash*jS_inv[ijack][mr]).trace())/p2/12./V;
	jZq_real[ijack][mr]=jZq[ijack][mr].real();
      }
  
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

  vvdcompl_t jSigma1(vdcompl_t(nmr),njacks);
  vvd_t jSigma1_real(vd_t(nmr),njacks);
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
	jSigma1[ijack][mr]=-I*A[ijack][mr].trace()/12./V;
	jSigma1_real[ijack][mr]=jSigma1[ijack][mr].real();
      }
  
  return jSigma1_real;
}


//project the amputated green function
jproj_t project(vprop_t GAMMA, const jvert_t &jLambda)
{
  int njacks=jLambda.size();
  int nmr=jLambda[0].size();
  
  //L_proj has 5 components: S(0), V(1), P(2), A(3), T(4)
  jvert_t L_proj(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),5),nmr),nmr),njacks);
  vvvvdcompl_t jG(vvvdcompl_t(vvdcompl_t(vdcompl_t(0.0,5),nmr),nmr),njacks);
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
	    {
	      jG[ijack][mr_fw][mr_bw][j]=L_proj[ijack][mr_fw][mr_bw][j].trace()/12.;
	      jG_real[ijack][mr_fw][mr_bw][j]=jG[ijack][mr_fw][mr_bw][j].real();
	    }
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

//compute fit parameters for a generic function f(x)=A+B*x+C*y(x)+D*z(x)+... 
vvd_t fit_par(const vvd_t &coord, const vd_t &error, const vvd_t &y, const int range_min, const int range_max)
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
      for(int j=0; j<n_par; j++)
	for(int k=0; k<n_par; k++)
	  if(isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);

      for(int ijack=0; ijack<njacks; ijack++)
	for(int k=0; k<n_par; k++)
	  if(isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]); 
    }

  for(int ijack=0; ijack<njacks; ijack++)
    jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);

  vvd_t par_array(vd_t(0.0,2),n_par);

  vd_t par_ave(0.0,n_par), par2_ave(0.0,n_par), par_err(0.0,n_par);

  for(int k=0; k<n_par; k++)
    {
      for(int ijack=0;ijack<njacks;ijack++)
	{
	  par_ave[k]+=jpars[ijack](k)/njacks;
	  par2_ave[k]+=jpars[ijack](k)*jpars[ijack](k)/njacks;
	}
      par_err[k]=sqrt((double)(njacks-1))*sqrt(par2_ave[k]-par_ave[k]*par_ave[k]);
      
      par_array[k][0] = par_ave[k];
      par_array[k][1] = par_err[k];
    }

  return par_array;
  
}

valarray<VectorXd> fit_par_jackknife(const vvd_t &coord, const vd_t &error, const vvd_t &y, const int range_min, const int range_max)
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
      for(int j=0; j<n_par; j++)
	for(int k=0; k<n_par; k++)
	  if(isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
      
      for(int ijack=0; ijack<njacks; ijack++)
	for(int k=0; k<n_par; k++)
	  if(isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]); 
    }
  
  for(int ijack=0; ijack<njacks; ijack++)
    jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);

  return jpars;
  
}


//compute linear fit parameters for the Zs
vvvd_t compute_fit_parameters_filtered(vd_t x, vd_t p4, vvvd_t y, vvd_t sigma, int tag, int njacks, double x_min, double x_max)
{
  
  vd_t S(0.0,6),Sx(0.0,6),Sxx(0.0,6);
  vvd_t Sy(vd_t(0.0,njacks),6), Sxy(vd_t(0.0,njacks),6);
  vvvd_t fit_parameter(vvd_t(vd_t(0.0,2),6),njacks); 
  
  for(int iZ=0;iZ<6;iZ++)
    {
      for(int t=0;t<=tag;t++)
	if(p4[t]/(x[t]*x[t])<0.28 and x[t]>x_min and x[t]<x_max)  //democratic filter
	  {
	    S[iZ]+=1/(sigma[t][iZ]*sigma[t][iZ]);
	    Sx[iZ]+= x[t]/(sigma[t][iZ]*sigma[t][iZ]);
	    Sxx[iZ]+=  x[t]*x[t]/(sigma[t][iZ]*sigma[t][iZ]);
	    
	    for(int ijack=0;ijack<njacks;ijack++)
	      {	     
		Sy[iZ][ijack]+= y[ijack][t][iZ]/(sigma[t][iZ]*sigma[t][iZ]);
		Sxy[iZ][ijack]+= x[t]*y[ijack][t][iZ]/(sigma[t][iZ]*sigma[t][iZ]);
	      }
	  }
      for(int ijack=0;ijack<njacks;ijack++)  // y = m*x + q
	{
	  fit_parameter[ijack][iZ][0]=(S[iZ]*Sxy[iZ][ijack]-Sx[iZ]*Sy[iZ][ijack])/(S[iZ]*Sxx[iZ]-Sx[iZ]*Sx[iZ]); //m
	  fit_parameter[ijack][iZ][1]=(Sxx[iZ]*Sy[iZ][ijack]-Sx[iZ]*Sxy[iZ][ijack])/(S[iZ]*Sxx[iZ]-Sx[iZ]*Sx[iZ]); //q
	}
    }
  
  return fit_parameter;
  
}


void print_file(const char* name_file, vd_t p2, vvd_t Z, vvd_t Z_err, int tag)
{
  ofstream outfile (name_file);
  outfile.precision(8);
  outfile<<fixed;
  if (outfile.is_open())
    {	  
      outfile<<"##p2_tilde \t Zq \t Zq_err \t Zs \t Zs_err \t Za \t Za_err \t Zp \t Zp_err \t Zv \t Zv_err \t Zt \t Zt_err "<<endl;
      for(int t=0;t<=tag;t++)
	{
	  outfile<<p2[t]<<"\t"<<Z[t][0]<<"\t"<<Z_err[t][0]<<"\t"<<Z[t][1]<<"\t"<<Z_err[t][1]<<"\t"<<Z[t][2]<<"\t"<<Z_err[t][2] \
		 <<"\t"<<Z[t][3]<<"\t"<<Z_err[t][3]<<"\t"<<Z[t][4]<<"\t"<<Z_err[t][4]<<"\t"<<Z[t][5]<<"\t"<<Z_err[t][5]<<endl;
	}
      outfile.close();
    }
  else cout << "Unable to open the output file "<<name_file<<endl;
}

void print_file_filtered(const char* name_file, vd_t p2, vd_t p4, vvd_t Z, vvd_t Z_err, int tag)
{
  ofstream outfile (name_file);
  outfile.precision(8);
  outfile<<fixed;
  if (outfile.is_open())
    {	  
      outfile<<"##p2_tilde \t Zq \t Zq_err \t Zs \t Zs_err \t Za \t Za_err \t Zp \t Zp_err \t Zv \t Zv_err \t Zt \t Zt_err "<<endl;
      for(int t=0;t<=tag;t++)
	{
	  if(p4[t]/(p2[t]*p2[t])<0.28)
	    outfile<<p2[t]<<"\t"<<Z[t][0]<<"\t"<<Z_err[t][0]<<"\t"<<Z[t][1]<<"\t"<<Z_err[t][1]<<"\t"<<Z[t][2]<<"\t"<<Z_err[t][2] \
		   <<"\t"<<Z[t][3]<<"\t"<<Z_err[t][3]<<"\t"<<Z[t][4]<<"\t"<<Z_err[t][4]<<"\t"<<Z[t][5]<<"\t"<<Z_err[t][5]<<endl;
	}
      outfile.close();
    }
  else cout << "Unable to open the output file "<<name_file<<endl;
}



vvd_t get_contraction(const int mr1, const string &T1, const int mr2, const string &T2, const string &ID, const string &reim, const string &parity, const int T, const int nconfs, const int njacks , const int* conf_id)
{
   
  vd_t data_V0P5_real(0.0,T);
  vd_t data_V0P5_imag(0.0,T);
  vd_t data_P5P5_real(0.0,T);
  vd_t data_P5P5_imag(0.0,T);
  
  vvd_t jP5P5_real(vd_t(0.0,T),njacks);
  vvd_t jP5P5_imag(vd_t(0.0,T),njacks);
  vvd_t jV0P5_real(vd_t(0.0,T),njacks);
  vvd_t jV0P5_imag(vd_t(0.0,T),njacks);

  int clust_size=nconfs/njacks;

  /////////
   
  for(int iconf=0;iconf<nconfs;iconf++)
    {
      int ijack=iconf/clust_size;
       
      ifstream infile;

      infile.open(path_to_contr(conf_id[iconf],mr1,T1,mr2,T2));
      infile.ignore(256,'5');
       
      for(int t=0; t<T; t++)
	{
	  infile>>data_V0P5_real[t];
	  infile>>data_V0P5_imag[t];	  
	}
       
      infile.ignore(256,'5');
      infile.ignore(256,'5');
       
      for(int t=0; t<T; t++)
	{
	  infile>>data_P5P5_real[t];
	  infile>>data_P5P5_imag[t];	  
	}

      for(int t=0; t<T; t++) jV0P5_real[ijack][t]+=data_V0P5_real[t];
      for(int t=0; t<T; t++) jV0P5_imag[ijack][t]+=data_V0P5_imag[t];
      for(int t=0; t<T; t++) jP5P5_real[ijack][t]+=data_P5P5_real[t];
      for(int t=0; t<T; t++) jP5P5_imag[ijack][t]+=data_P5P5_imag[t];
      
      infile.close(); 
    }
  
  jV0P5_real=jackknife_double(jV0P5_real,T,nconfs,clust_size);
  jV0P5_imag=jackknife_double(jV0P5_imag,T,nconfs,clust_size);
  jP5P5_real=jackknife_double(jP5P5_real,T,nconfs,clust_size);
  jP5P5_imag=jackknife_double(jP5P5_imag,T,nconfs,clust_size);

  vvd_t jvec(vd_t(0.0,T),njacks);

  if(ID=="P5P5" and reim=="RE") jvec=jP5P5_real;
  if(ID=="P5P5" and reim=="IM") jvec=jP5P5_imag;
  if(ID=="V0P5" and reim=="RE") jvec=jV0P5_real;
  if(ID=="V0P5" and reim=="IM") jvec=jV0P5_imag;

  double par;
  
  if(parity=="EVEN") par=1.0;
  if(parity=="ODD") par=-1.0;
  
  vvd_t jvec_sym(vd_t(0.0,T),njacks);
  vvd_t jvec_par(vd_t(0.0,T/2+1),njacks);
  
  for(int ijack=0;ijack<njacks;ijack++)
    {
      for(int t=0;t<T;t++)
	jvec_sym[ijack][(T-t)%T]=jvec[ijack][t];
      for(int t=0;t<T/2+1;t++)
	jvec_par[ijack][t]=(jvec[ijack][t]+par*jvec_sym[ijack][t])/2.0;
    }

  // if(ID=="P5P5" and reim=="RE" and parity=="EVEN"){
  //   cout<<"**********DEBUG*************"<<endl;
  //   for(int ijack=0;ijack<njacks;ijack++)
  //     for(int t=0;t<T;t++)
  // 	cout<<jvec[ijack][t]<<endl;
  //   cout<<"**********DEBUG*************"<<endl;}

  return jvec_par;

}

//function to use in Newton's method for M_eff
double f_mass (int t, int T, double x0, double y)
{
  double f = cosh(x0*(t-T/2))/cosh(x0*(t+1-T/2)) - y;  // y=c(t)/c(t+1), where c(t) is the correlator at the time t

  return f;
}

//derivative to use in Newton's method for M_eff
double f_prime_mass (int t, int T, double x0)
{
  int k = t-T/2;

  double fp = ( k*sinh(x0*k) - (1+k)*cosh(x0*k)*tanh(x0*(1+k)) )/cosh(x0*(t+1-T/2));

  return fp;
}

//Newton's Method for M_eff (in a fixed jackknife)
double solve_Newton (vvd_t C, int ijack, int t, const int T) 
{
  double k = C[ijack][t]/C[ijack][t+1];

  // cout<<"**********DEBUG*************"<<endl;
  // cout<<"jack: "<<ijack<<"  t: "<<t<<"  c(t)/c(t+1): "<<k<<endl;
  // cout<<"**********DEBUG*************"<<endl;
  
  if(k<1.0)
    {return nan("");}
  else{
  
  double eps=1e-14;
  int max_iteration=500; 
  int count_iteration=0;
  int g=0;

  double x0=1.09; //seed
  double x1;
    
  double y, yp, x;
    
  x1=x0;
  do
    {
      x=x1;
	
      y=f_mass(t,T,x,k);
      yp=f_prime_mass(t,T,x);
	
      x1 = x - y/yp;
	
      count_iteration++;
      g++;
	
      //  cout<<count_iteration<<endl;
	
    } while ( abs(x1-x) >= x1*eps and count_iteration!=max_iteration );


  // cout<<x0<<" ";
  
  // cout<<"********DEBUG*****************************"<<endl; 
  // if(count_iteration==max_iteration)
  //   cerr<<t<<" Newton's method did not converge for the jackknife n. "<<ijack<<" in "<<max_iteration<<" iterations. The value is "<<x1<<" k "<<k<<endl;
  // else cout<<t<<" Jackknife n. "<<ijack<<" has converged with success to the value "<<x1<<" in "<<g<<" iterations"<<" k "<<k<<endl;
  // cout<<"********DEBUG*****************************"<<endl; 
  
  return x1;
  }
}

//compute delta m_cr
vvvd_t compute_deltam_cr(const int T, const int nconfs, const int njacks,const int* conf_id)
{
  int nmr=8;
  
  //define jackknife V0P5 correlators
  vvvvd_t jV0P5_LL(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  vvvvd_t jV0P5_0M(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  vvvvd_t jV0P5_M0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  vvvvd_t jV0P5_0T(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  vvvvd_t jV0P5_T0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  vvvvd_t jV0P5_0P(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  vvvvd_t jV0P5_P0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  //define deltam_cr
  vvvvd_t num_deltam_cr_corr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nmr),nmr);
  vvvvd_t den_deltam_cr_corr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nmr),nmr);
  vvvvd_t deltam_cr_corr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nmr),nmr);

  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      {
	//load corrections
	jV0P5_LL[mr_fw][mr_bw]=get_contraction(mr_fw,"F",mr_bw,"F","V0P5","IM","ODD",T,nconfs,njacks,conf_id);
	jV0P5_0M[mr_fw][mr_bw]=get_contraction(mr_fw,"0",mr_bw,"FF","V0P5","IM","ODD",T,nconfs,njacks,conf_id);
	jV0P5_M0[mr_fw][mr_bw]=get_contraction(mr_fw,"FF",mr_bw,"0","V0P5","IM","ODD",T,nconfs,njacks,conf_id);
	jV0P5_0T[mr_fw][mr_bw]=get_contraction(mr_fw,"0",mr_bw,"T","V0P5","IM","ODD",T,nconfs,njacks,conf_id);
	jV0P5_T0[mr_fw][mr_bw]=get_contraction(mr_fw,"T",mr_bw,"0","V0P5","IM","ODD",T,nconfs,njacks,conf_id);
	//load the derivative wrt counterterm
	jV0P5_0P[mr_fw][mr_bw]=get_contraction(mr_fw,"0",mr_bw,"P","V0P5","RE","ODD",T,nconfs,njacks,conf_id);
	jV0P5_P0[mr_fw][mr_bw]=get_contraction(mr_fw,"P",mr_bw,"0","V0P5","RE","ODD",T,nconfs,njacks,conf_id);
      }

 
  
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      for(int ijack=0;ijack<njacks;ijack++)
	for(int t=0;t<T/2+1;t++)
	  {
	    num_deltam_cr_corr[mr_fw][mr_bw][ijack][t]=jV0P5_LL[mr_fw][mr_bw][ijack][t]+jV0P5_0M[mr_fw][mr_bw][ijack][t]+jV0P5_M0[mr_fw][mr_bw][ijack][t]+jV0P5_0T[mr_fw][mr_bw][ijack][t]+jV0P5_T0[mr_fw][mr_bw][ijack][t];
	    den_deltam_cr_corr[mr_fw][mr_bw][ijack][t]=-jV0P5_P0[mr_fw][mr_bw][ijack][t]+jV0P5_0P[mr_fw][mr_bw][ijack][t];
	    deltam_cr_corr[mr_fw][mr_bw][ijack][t]=-num_deltam_cr_corr[mr_fw][mr_bw][ijack][t]/den_deltam_cr_corr[mr_fw][mr_bw][ijack][t];
	  }

  vvvd_t mean_value(vvd_t(vd_t(0.0,T/2+1),nmr),nmr), sqr_mean_value(vvd_t(vd_t(0.0,T/2+1),nmr),nmr), error(vvd_t(vd_t(0.0,T/2+1),nmr),nmr);
  
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      for(int t=0;t<T/2+1;t++)
	{
	  for(int ijack=0;ijack<njacks;ijack++)
	    {
	      mean_value[mr_fw][mr_bw][t]+=deltam_cr_corr[mr_fw][mr_bw][ijack][t]/njacks;
	      sqr_mean_value[mr_fw][mr_bw][t]+=deltam_cr_corr[mr_fw][mr_bw][ijack][t]*deltam_cr_corr[mr_fw][mr_bw][ijack][t]/njacks;
	    }
	  error[mr_fw][mr_bw][t]=sqrt((double)(njacks-1))*sqrt(sqr_mean_value[mr_fw][mr_bw][t]-mean_value[mr_fw][mr_bw][t]*mean_value[mr_fw][mr_bw][t]);
	}

  //t-range for the fit
  int t_min=12;
  int t_max=23;
  
  vvd_t coord(vd_t(0.0,T/2+1),1);
  for(int j=0; j<T/2+1; j++)
    {
      coord[0][j] = 1.0;  //fit a costante
    }

  vvvvd_t deltam_cr_fit_parameters(vvvd_t(vvd_t(vd_t(0.0,2),coord.size()),nmr),nmr);
  
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      deltam_cr_fit_parameters[mr_fw][mr_bw]=fit_par(coord,error[mr_fw][mr_bw],deltam_cr_corr[mr_fw][mr_bw],t_min,t_max); 
  
  vvvd_t deltam_cr(vvd_t(vd_t(0.0,2),nmr),nmr);
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      {
      deltam_cr[mr_fw][mr_bw][0]=deltam_cr_fit_parameters[mr_fw][mr_bw][0][0]; 
      deltam_cr[mr_fw][mr_bw][1]=deltam_cr_fit_parameters[mr_fw][mr_bw][0][1];
      }
  
  return deltam_cr;
}

//compute effective mass
vvvd_t compute_eff_mass(const int T, const int nconfs, const int njacks, const int *conf_id)
{

  vvvvd_t jP5P5_00(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      jP5P5_00[mr_fw][mr_bw]=get_contraction(mr_fw,"0",mr_bw,"0","P5P5","RE","EVEN",T,nconfs,njacks,conf_id);
  
  // cout<<"**********DEBUG*************"<<endl;
  // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
  //   for(int mr_bw=0;mr_bw<nmr;mr_bw++)
  //     for(int ijack=0;ijack<njacks;ijack++)
  // 	for(int t=0;t<T/2-1;t++)
  // 	  cout<<mr_fw<<" "<<mr_bw<<" ijack "<<ijack<<" t "<<t<<"\t"<< jP5P5_00[mr_fw][mr_bw][ijack][t]/jP5P5_00[mr_fw][mr_bw][ijack][t+1]<<endl;
  // cout<<"**********DEBUG*************"<<endl;

  vvvvd_t M_eff(vvvd_t(vvd_t(vd_t(T/2),njacks),nmr),nmr);

  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      {
	// cout<<"**********DEBUG*************"<<endl;
	// int r1 = mr_fw%nr;
	// int m1 = (mr_fw-r1)/nr;
	// int r2 = mr_bw%nr;
	// int m2 = (mr_bw-r2)/nr;
	// cout<<"r1 "<<r1<<" m1 "<<m1<<" r2 "<<r2<<" m2 "<<m2<<endl;
	// cout<<"--------------------------------------------"<<endl;
	// cout<<"**********DEBUG*************"<<endl;
	
	for(int ijack=0; ijack<njacks;ijack++)
	  for(int t=0;t<T/2;t++) 
	    M_eff[mr_fw][mr_bw][ijack][t] = solve_Newton (jP5P5_00[mr_fw][mr_bw],ijack,t,T);
	
      }
  
  // cout<<"**********DEBUG*************"<<endl;
  // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
  //   for(int mr_bw=0;mr_bw<nmr;mr_bw++)
  //     for(int ijack=0;ijack<njacks;ijack++)
  // 	for(int t=0;t<T/2;t++)
  // 	  cout<<mr_fw<<" "<<mr_bw<<" ijack "<<ijack<<" t "<<t<<"\t"<<M_eff[mr_fw][mr_bw][ijack][t]<<endl;
  // cout<<"**********DEBUG*************"<<endl;

    // cout<<"**********DEBUG*************"<<endl;
  // for(double i=0;i<10;i+=0.1){ cout<<i+1<<"\t"<<f_mass(22,T,i,jP5P5_00[0][0][0][22]/jP5P5_00[0][0][0][23])<<endl;}
  // cout<<"**********DEBUG*************"<<endl;
  
  
  vvvd_t mass_ave(vvd_t(vd_t(0.0,T/2),nmr),nmr), sqr_mass_ave(vvd_t(vd_t(0.0,T/2),nmr),nmr), mass_err(vvd_t(vd_t(0.0,T/2),nmr),nmr);
  //   vd_t mass_ave(0.0,T/2), sqr_mass_ave(0.0,T/2), mass_err(0.0,T/2);
  
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      for(int t=0;t<T/2;t++)
	{
	  for(int ijack=0;ijack<njacks;ijack++)
	    {
	      mass_ave[mr_fw][mr_bw][t]+=M_eff[mr_fw][mr_bw][ijack][t]/njacks;
	      sqr_mass_ave[mr_fw][mr_bw][t]+=M_eff[mr_fw][mr_bw][ijack][t]*M_eff[mr_fw][mr_bw][ijack][t]/njacks;
	    }
	  mass_err[mr_fw][mr_bw][t]=sqrt((double)(njacks-1))*sqrt(sqr_mass_ave[mr_fw][mr_bw][t]-mass_ave[mr_fw][mr_bw][t]*mass_ave[mr_fw][mr_bw][t]);      
	}

  
  //t-range for the fit
  int t_min=12;
  int t_max=23;
  
  vvd_t coord(vd_t(0.0,T/2),1);
  for(int j=0; j<T/2; j++)
    {
      coord[0][j] = 1.0;  //fit a costante
    }

  vvvvd_t eff_mass_fit_parameters(vvvd_t(vvd_t(vd_t(0.0,2),coord.size()),nmr),nmr);
  
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      {
	eff_mass_fit_parameters[mr_fw][mr_bw]=fit_par(coord,mass_err[mr_fw][mr_bw],M_eff[mr_fw][mr_bw],t_min,t_max);
      }
  
  vvvd_t eff_mass(vvd_t(vd_t(0.0,2),nmr),nmr);
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      {
	eff_mass[mr_fw][mr_bw][0]=eff_mass_fit_parameters[mr_fw][mr_bw][0][0]; 
	eff_mass[mr_fw][mr_bw][1]=eff_mass_fit_parameters[mr_fw][mr_bw][0][1];
      }

  //  cout<<"********DEBUG*****************************"<<endl; 
  // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
  //   for(int mr_bw=0;mr_bw<nmr;mr_bw++)
  //     {
  // 	int r1 = mr_fw%nr;
  // 	int m1 = (mr_fw-r1)/nr;
  // 	int r2 = mr_bw%nr;
  // 	int m2 = (mr_bw-r2)/nr;
  
  // 	cout<<"r1 "<<r1<<" m1 "<<m1<<" r2 "<<r2<<" m2 "<<m2<<"  eff_mass: "<<eff_mass[mr_fw][mr_bw][0]<<" +- "<<eff_mass[mr_fw][mr_bw][1]<<endl;
  //     }
  // cout<<"********DEBUG*****************************"<<endl; 
  
  return eff_mass;
}


/***********************************************************/
/*************************** main **************************/
/***********************************************************/

  
int main(int narg,char **arg)
{
  
  if (narg!=11){
    cerr<<"Number of arguments not valid: <mom file> <nconfs> <njacks> <L> <T> <initial conf_id> <step conf_id> <p2fit min> <p2fit max> <action=sym/iwa>"<<endl;
    exit(0);
  }

  system("clear");
  cout<<endl<<endl;
  
  int nconfs=stoi(arg[2]); 
  int njacks=stoi(arg[3]);
  int clust_size=nconfs/njacks;
  int conf_id[nconfs];
  double L=stod(arg[4]),T=stod(arg[5]);
  size_t nhits=1; //!

  nm = 4;  //! to be passed from command line
  nr = 2;

  nmr=nm*nr;
  
  for(int iconf=0;iconf<nconfs;iconf++)
    conf_id[iconf]=stoi(arg[6])+iconf*stoi(arg[7]);
  
  double p2fit_min=stod(arg[8]);  //!
  double p2fit_max=stod(arg[9]);  //!

  // const double use_tad = 1.0;  //!

  cout<<"N confs = "<<nconfs<<endl;
  cout<<"N jacks = "<<njacks<<endl;
  cout<<"Clust size = "<<clust_size<<endl;
  cout<<"L = "<<L<<"\t T = "<<T<<endl;
  cout<<"Fit range = ["<<p2fit_min<<":"<<p2fit_max<<"]"<<endl;
  

  double beta=0.0, plaquette=0.0;
  vector<double> c_v(3), c_a(3), c_s(3), c_p(3), c_t(3);
  
  //beta & plaquette
  if(strcmp(arg[10],"iwa")==0)  //Nf=4 (Iwasaki)
    {
      beta=1.90;
      plaquette=0.574872;

      c_v={0.2881372,-0.2095,-0.516583};
      c_a={0.9637998,-0.2095,-0.516583};
      c_s={2.02123300,-1./4.,0.376167};
      c_p={0.66990790,-1./4.,0.376167};
      c_t={0.3861012,-0.196,-0.814167};
      
      cout<<"Action:  Iwasaki"<<endl;
    }
  else if(strcmp(arg[10],"sym")==0)  //Nf=2 (Symanzik)
    {
      beta=3.90;
      plaquette=0.582591;

      c_a={1.5240798,-1./3.,-125./288.};
      c_v={0.6999177,-1./3.,-125./288.};
      c_s={2.3547298,-1./4.,0.5};        
      c_p={0.70640549,-1./4.,0.5};
      c_t={0.9724758,-13./36.,-161./216.};

      cout<<"Action:  Symanzik"<<endl;
    }
  else
    {
      cerr<<"WARNING: wrong action argument. Please write 'sym' for Symanzik action or 'iwa' for Iwasaki action.";
      exit(0);
    }
  
  //g2_tilde
  double g2=6.0/beta;
  double g2_tilde=g2/plaquette;
  
  cout<<"Beta = "<<beta<<endl;
  cout<<"Plaquette = "<<plaquette<<endl;
  cout<<"g2_tilde = "<<g2_tilde<<endl<<endl;

  //delta m_cr
  vvvd_t deltam_cr_array= compute_deltam_cr(T,nconfs,njacks,conf_id);
  
 
  // cout<<"********DEBUG*****************************"<<endl; 
  // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
  //   for(int mr_bw=0;mr_bw<nmr;mr_bw++)
  //     {
  // 	int r1 = mr_fw%nr;
  // 	int m1 = (mr_fw-r1)/nr;
  // 	int r2 = mr_bw%nr;
  // 	int m2 = (mr_bw-r2)/nr;
	
  // 	cout<<"r1 "<<r1<<" m1 "<<m1<<" r2 "<<r2<<" m2 "<<m2<<"  deltam_cr "<<deltam_cr_array[mr_fw][mr_bw][0]<<"+-"<<deltam_cr_array[mr_fw][mr_bw][1]<<endl;
  //     }
  // cout<<"********DEBUG*****************************"<<endl<<endl; 
    
  
  vvd_t deltam_cr(vd_t(0.0,nmr),nmr);
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      deltam_cr[mr_fw][mr_bw] = deltam_cr_array[mr_fw][mr_bw][0];
  
  //double deltam_cr = 0.230697;
  
  //Effective Mass
  vvvd_t eff_mass_array = compute_eff_mass(T,nconfs,njacks,conf_id); 

  vvd_t eff_mass(vd_t(0.0,nmr),nmr);
  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
      eff_mass[mr_fw][mr_bw] = eff_mass_array[mr_fw][mr_bw][0];
  
  // cout<<"eff_mass: "<<eff_mass_array[0]<<" +- "<<eff_mass_array[1]<<endl;
  
  // ///////////////////////////////////////////////////////////////////////////////////////////////////////
  
  
   read_mom_list(arg[1]);
  
   cout<<"Read: "<<mom_list.size()<<" momenta."<<endl<<endl;
  
  //create gamma matrices
   vprop_t GAMMA=make_gamma();
    
   vector<string> Mass={"M0_","M1_","M2_","M3_"};
   vector<string> R={"R0_","R1_"};
   vector<string> Type={"0","F","FF","T","P"/*,"S"*/};

   vector<double> p2_vector, p4_vector;
   
   // int nm=Mass.size();
   // int nr=R.size();
   int nt=Type.size();

   int combo=nm*nr*nt*nhits*nconfs;

   //DEBUG
   // cout<<combo<<endl;
   //DEBUG

   //array of input files to be read in a given conf
   ifstream input[combo];

  

   int icombo=-1;

   for(int iconf=0;iconf<nconfs;iconf++)
     for(size_t ihit=0;ihit<nhits;ihit++)
       {
	 string hit_suffix = "";
	 if(nhits>1) hit_suffix = "_hit_" + to_string(ihit);
	 
	 for(int t=0;t<nt;t++)
	   for(int m=0;m<nm;m++)
	     for(int r=0;r<nr;r++)
	       {
		 icombo++;		 
		 string path = path_to_conf(conf_id[iconf],"S_"+Mass[m]+R[r]+Type[t]+hit_suffix);
		 
		 input[icombo].open(path,ios::binary);
		 
		 if(!input[icombo].good())
		   {cerr<<"Unable to open file "<<path<<" combo "<<icombo<<endl;
		     exit(1);}
	       }
       }
   
   
   for(size_t imom=0; imom<10; imom++)
     {
       // put to zero jackknife props and verts
       jprop_t jS_0(valarray<prop_t>(prop_t::Zero(),nmr),njacks);		
       jprop_t jS_self_tad(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
       jprop_t jS_p(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
       // jprop_t jS_s(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
       
       jvert_t jVert_0 (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
       jvert_t jVert_11_self_tad (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
       jvert_t jVert_p (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
       //  jvert_t jVert_s (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
       
       jprop_t jS_em(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
       jvert_t jVert_em (vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
       

       icombo=-1;
       
       for(int iconf=0;iconf<nconfs;iconf++)
	 {
	   vvprop_t S(vprop_t(prop_t::Zero(),nmr),nt);  // S[type][mr] e.g.: S[1][0]=S_M0_R0_F, S[2][1]=S_M0_R1_FF
	   
	   for(size_t ihit=0;ihit<nhits;ihit++)
	     {
	       int ijack=iconf/clust_size;
	       
	       string hit_suffix = "";
	       if(nhits>1) hit_suffix = "_hit_" + to_string(ihit);
	       
	       for(int t=0;t<nt;t++)
		 for(int m=0;m<nm;m++)
		   for(int r=0;r<nr;r++)
		     {
		       icombo++;
		       string path = path_to_conf(conf_id[iconf],"S_"+Mass[m]+R[r]+Type[t]+hit_suffix);
		       
		       // input[icombo].open(path,ios::binary);
		       
		       // if(!input[icombo].good())
		       // 	 {cerr<<"Unable to open file "<<path<<endl;
		       // exit(1);}
		       
		       int mr = r + nr*m; // M0R0,M0R1,M1R0,M1R1,M2R0,M2R1,M3R0,M3R1
		       
		       //create all the propagators in a given conf and a given mom
		       S[t][mr] = read_prop(input[icombo],path);
		       
		       if(t==4) S[t][mr]*=dcompl(0.0,-1.0);
		       if(t==5) S[t][mr]*=dcompl(1.0,0.0);  
		     }
	       
	       //create vertex functions with fixed momentum
	       
	       vert_t Vert_0=make_vertex(S[0], S[0], GAMMA);  //Vert_0[mr_fw][mr_bw][gamma]
	       
	       vert_t Vert_11=make_vertex(S[1], S[1], GAMMA);
	       
	       vert_t Vert_02=make_vertex(S[0], S[2], GAMMA);
	       vert_t Vert_20=make_vertex(S[2], S[0], GAMMA);
	       
	       vert_t Vert_0t=make_vertex(S[0], S[3], GAMMA);
	       vert_t Vert_t0=make_vertex(S[3], S[0], GAMMA);
	       
	       vert_t Vert_0p=make_vertex(S[0], S[4], GAMMA);
	       vert_t Vert_p0=make_vertex(S[4], S[0], GAMMA);
	       
	       // vert_t Vert_0s = make_vertex(S[0], S[5], GAMMA);
	       // vert_t Vert_s0 = make_vertex(S[5], S[0], GAMMA);
	       
	       //create pre-jackknife propagator:  jS_0[ijack][mr]
	       jS_0[ijack] += S[0];
	       jS_self_tad[ijack] += S[2] + S[3];
	       jS_p[ijack] += S[4];
	       // jS_s[ijack] += S[5];
	       
	       //create pre-jackknife vertex:  jVert_0[ijack][mr_fw][mr_bw][gamma]
	       jVert_0[ijack] += Vert_0;
	       jVert_11_self_tad[ijack] += Vert_11 + Vert_02 + Vert_20 + Vert_0t + Vert_t0;
	       jVert_p[ijack] += Vert_0p + Vert_p0;
	       // jVert_s[ijack] += Vert_0s + Vert_s0;
	       
	     } //close hits loop
	 } //close confs loop
       
       //jackknife of propagators
       jS_0 = jackknife_prop(jS_0,nconfs,clust_size,nhits);
       jS_self_tad = jackknife_prop(jS_self_tad,nconfs,clust_size,nhits);
       jS_p = jackknife_prop(jS_p,nconfs,clust_size,nhits);
       
       //jackknife of vertices
       jVert_0 = jackknife_vertex(jVert_0,nconfs,clust_size,nhits);
       jVert_11_self_tad = jackknife_vertex(jVert_11_self_tad,nconfs,clust_size,nhits);
       jVert_p = jackknife_vertex(jVert_p,nconfs,clust_size,nhits);
       
       //define em propagator and vertex
       for(int ijack=0;ijack<njacks;ijack++)
	 for(int mr=0;mr<nmr;mr++)
	   {
	     jS_em[ijack][mr] = jS_self_tad[ijack][mr] - deltam_cr[mr][mr]*jS_p[ijack][mr]; // + scalar correction
	     for(int mr2=0;mr2<nmr;mr2++)
	       for(int igam=0;igam<16;igam++)
		 jVert_em[ijack][mr][mr2][igam] = jVert_11_self_tad[ijack][mr][mr2][igam] - deltam_cr[mr][mr2]*jVert_p[ijack][mr][mr2][igam]; // + scalar correction
	   }
       
       //inverse of the propagators
       jprop_t jS_0_inv = invert_jprop(jS_0);         //jS_0_inv[ijack][mr]
       jprop_t jS_em_inv = jS_0_inv*jS_em*jS_0_inv;
       
       //amputate external legs
       jvert_t jLambda_0 = amputate(jS_0_inv, jVert_0, jS_0_inv, GAMMA);     //jLambda_0[ijack][mr_fw][mr_bw][gamma]
       jvert_t jLambda_em = amputate(jS_0_inv, jVert_em, jS_0_inv, GAMMA);
       jvert_t jLambda_a = amputate(jS_em_inv, jVert_0, jS_0_inv, GAMMA);
       jvert_t jLambda_b = amputate(jS_0_inv, jVert_0, jS_em_inv, GAMMA);
       
       //compute Zq according to RI'-MOM, one for each momentum               //<----------------------------------------------------! QUI COMPARE imom (!!!)
       jZ_t jZq = compute_jZq(GAMMA,jS_0_inv,L,T,imom);     //jZq[ijack][mr]
       jZ_t jZq_em = - compute_jZq(GAMMA,jS_em_inv,L,T,imom);
       //compute Zq according to Sigma1-way
       jZ_t jSigma1 = compute_jSigma1(GAMMA,jS_0_inv,L,T,imom);
       jZ_t jSigma1_em = - compute_jSigma1(GAMMA,jS_em_inv,L,T,imom);
       
       //compute the projected green function as a vector (S,V,P,A,T)
       jproj_t jG_0 = project(GAMMA,jLambda_0);    //jG_0[ijack][mr_fw][mr_bw][i] (i=S,V,P,A,T)
       jproj_t jG_em = project(GAMMA,jLambda_em);
       jproj_t jG_a = project(GAMMA,jLambda_a);
       jproj_t jG_b = project(GAMMA,jLambda_b);
       
       //compute Z's according to RI-MOM and to Sigma1-way, one for each momentum
       jZbil_t jZ(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
       jZbil_t jZ1(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
       
       jZbil_t jZ_em(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
       jZbil_t jZ1_em(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
       
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
       
       // cout<<"********DEBUG*************"<<endl;
       // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
       //   for(int mr_bw=0;mr_bw<nmr;mr_bw++)
       //     cout<<0<<"\t"<<jZ[0][mr_fw][mr_bw][1].real()<<endl;
       // cout<<"********DEBUG*************"<<endl;

       
       //create p_tilde vector
       
       vd_t p(0.0,4);
       vd_t p_tilde(0.0,4);
       double p2=0.0;
       double p2_space=0.0;
       double p4=0.0;  //for the democratic filter

       p={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
       p_tilde={sin(p[0]),sin(p[1]),sin(p[2]),sin(p[3])};

       for(int coord=0;coord<4;coord++)
	 p2+=p_tilde[coord]*p_tilde[coord];
       for(int coord=0;coord<3;coord++)
	 p2_space+=p_tilde[coord]*p_tilde[coord];
       for(int coord=0;coord<4;coord++)
	 p4+=p_tilde[coord]*p_tilde[coord]*p_tilde[coord]*p_tilde[coord]; //for the democratic filter
       
       p2_vector.push_back(p2);
       p4_vector.push_back(p4);

       vector<double> c_q(3);
   
       if(strcmp(arg[10],"sym")==0) c_q={1.14716212+2.07733285/(double)Np[imom],-73./360.-157./180./(double)Np[imom],7./240.};   //Symanzik action
       if(strcmp(arg[10],"iwa")==0) c_q={0.6202244+1.8490436/(double)Np[imom],-0.0748167-0.963033/(double)Np[imom],0.0044};      //Iwasaki action

       
       //Subtraction of O(a^2) effects through perturbation theory
       
       jZ_t jZq_sub(vd_t(nmr),njacks), jSigma1_sub(vd_t(nmr),njacks);
       jproj_t jG_0_sub(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks);
       jZbil_t jZ_sub(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks), jZ1_sub(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);

       jZ_t jZq_em_sub(vd_t(nmr),njacks), jSigma1_em_sub(vd_t(nmr),njacks);
       jproj_t jG_em_sub(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks), jG_a_sub(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks), jG_b_sub(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks);
       jZbil_t jZ_em_sub(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks), jZ1_em_sub(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
      
       for(int ijack=0;ijack<njacks;ijack++)
	 for(int mr=0; mr<nmr; mr++)
	   {
	     
	     //subtraction of O(g^2a^2) effects
	     jZq_sub[ijack][mr]=subtract(c_q,jZq[ijack][mr],p2,p4,g2_tilde);
	     jSigma1_sub[ijack][mr]=subtract(c_q,jSigma1[ijack][mr],p2,p4,g2_tilde);
	     
	     for(int mr2=0; mr2<nmr; mr2++)
	       {
		 jG_0_sub[ijack][mr][mr2][0]=subtract(c_s,jG_0[ijack][mr][mr2][0],p2,p4,g2_tilde);
		 jG_0_sub[ijack][mr][mr2][1]=subtract(c_a,jG_0[ijack][mr][mr2][1],p2,p4,g2_tilde);
		 jG_0_sub[ijack][mr][mr2][2]=subtract(c_p,jG_0[ijack][mr][mr2][2],p2,p4,g2_tilde);
		 jG_0_sub[ijack][mr][mr2][3]=subtract(c_v,jG_0[ijack][mr][mr2][3],p2,p4,g2_tilde);
		 jG_0_sub[ijack][mr][mr2][4]=subtract(c_t,jG_0[ijack][mr][mr2][4],p2,p4,g2_tilde);
		 for(int i=0; i<5; i++)
		   {
		     jZ_sub[ijack][mr][mr2][i] = sqrt(jZq_sub[ijack][mr])*sqrt(jZq_sub[ijack][mr2])/jG_0_sub[ijack][mr][mr2][i];
		     jZ1_sub[ijack][mr][mr2][i] = sqrt(jSigma1_sub[ijack][mr])*sqrt(jSigma1_sub[ijack][mr2])/jG_0_sub[ijack][mr][mr2][i];
		   } 
	       }
	     //subtraction of O(e^2a^2) effects
	     jZq_em_sub[ijack][mr]=subtract(c_q,jZq_em[ijack][mr],p2,p4,3/4);
	     jSigma1_em_sub[ijack][mr]=subtract(c_q,jSigma1_em[ijack][mr],p2,p4,3/4);
	     
	     for(int mr2=0; mr2<nmr; mr2++)
	       {
		 jG_em_sub[ijack][mr][mr2][0]=subtract(c_s,jG_em[ijack][mr][mr2][0],p2,p4,3/4);
		 jG_em_sub[ijack][mr][mr2][1]=subtract(c_a,jG_em[ijack][mr][mr2][1],p2,p4,3/4);
		 jG_em_sub[ijack][mr][mr2][2]=subtract(c_p,jG_em[ijack][mr][mr2][2],p2,p4,3/4);
		 jG_em_sub[ijack][mr][mr2][3]=subtract(c_v,jG_em[ijack][mr][mr2][3],p2,p4,3/4);
		 jG_em_sub[ijack][mr][mr2][4]=subtract(c_t,jG_em[ijack][mr][mr2][4],p2,p4,3/4);

		 jG_a_sub[ijack][mr][mr2][0]=subtract(c_s,jG_a[ijack][mr][mr2][0],p2,p4,3/4);
		 jG_a_sub[ijack][mr][mr2][1]=subtract(c_a,jG_a[ijack][mr][mr2][1],p2,p4,3/4);
		 jG_a_sub[ijack][mr][mr2][2]=subtract(c_p,jG_a[ijack][mr][mr2][2],p2,p4,3/4);
		 jG_a_sub[ijack][mr][mr2][3]=subtract(c_v,jG_a[ijack][mr][mr2][3],p2,p4,3/4);
		 jG_a_sub[ijack][mr][mr2][4]=subtract(c_t,jG_a[ijack][mr][mr2][4],p2,p4,3/4);

		 jG_b_sub[ijack][mr][mr2][0]=subtract(c_s,jG_b[ijack][mr][mr2][0],p2,p4,3/4);
		 jG_b_sub[ijack][mr][mr2][1]=subtract(c_a,jG_b[ijack][mr][mr2][1],p2,p4,3/4);
		 jG_b_sub[ijack][mr][mr2][2]=subtract(c_p,jG_b[ijack][mr][mr2][2],p2,p4,3/4);
		 jG_b_sub[ijack][mr][mr2][3]=subtract(c_v,jG_b[ijack][mr][mr2][3],p2,p4,3/4);
		 jG_b_sub[ijack][mr][mr2][4]=subtract(c_t,jG_b[ijack][mr][mr2][4],p2,p4,3/4);
		 
		 for(int i=0; i<5; i++)
		   {
		     jZ_em_sub[ijack][mr][mr2][i] = (-jG_em_sub[ijack][mr][mr2][i]+jG_a_sub[ijack][mr][mr2][i]+jG_b_sub[ijack][mr][mr2][i])/jG_0[ijack][mr][mr2][i] + \
		       0.5*(jZq_em_sub[ijack][mr]/jZq_sub[ijack][mr] + jZq_em_sub[ijack][mr2]/jZq_sub[ijack][mr2]);
		     jZ1_em_sub[ijack][mr][mr2][i] = (-jG_em_sub[ijack][mr][mr2][i]+jG_a_sub[ijack][mr][mr2][i]+jG_b_sub[ijack][mr][mr2][i])/jG_0[ijack][mr][mr2][i] + \
		       0.5*(jSigma1_em_sub[ijack][mr]/jSigma1_sub[ijack][mr] + jSigma1_em_sub[ijack][mr2]/jSigma1_sub[ijack][mr2]);
		   } 
	       }
	   }
      
       
       //Goldstone pole subtraction from jG_p and jG_s
       
       int neq = fact(nm+nr-1)/fact(nr)/fact(nm-1);
       vd_t m_eff_equivalent(neq);
       vvd_t jG_p_equivalent(vd_t(neq),njacks);
       vvd_t jG_s_equivalent(vd_t(neq),njacks);
       
       
       double M_average;
       vd_t  Gp_average(njacks),  Gs_average(njacks);
       int ieq=0;
       for(int mA=0; mA<nm; mA++)
	 for(int mB=mA; mB<nm; mB++)
	   {
	     M_average=0.0;
	     for(int ijack=0;ijack<njacks;ijack++) Gp_average[ijack]=0.0;
	     for(int ijack=0;ijack<njacks;ijack++) Gs_average[ijack]=0.0;
	     
	     for(int r=0; r<1; r++)
	       {
		 M_average += (eff_mass[r+nr*mA][r+nr*mB]+eff_mass[r+nr*mB][r+nr*mA])/(2.0*1);
		 for(int ijack=0;ijack<njacks;ijack++) Gp_average[ijack] += (jG_0[ijack][r+nr*mA][r+nr*mB][2]+jG_0[ijack][r+nr*mB][r+nr*mA][2])/(2.0*1);
		 for(int ijack=0;ijack<njacks;ijack++) Gs_average[ijack] += (jG_0[ijack][r+nr*mA][r+nr*mB][0]+jG_0[ijack][r+nr*mB][r+nr*mA][0])/(2.0*1);
	       }
	     m_eff_equivalent[ieq]=M_average;
	     for(int ijack=0;ijack<njacks;ijack++) jG_p_equivalent[ijack][ieq]=Gp_average[ijack];
	     for(int ijack=0;ijack<njacks;ijack++) jG_s_equivalent[ijack][ieq]=Gs_average[ijack];	  
	     ieq++;  //ieq={00,01,02,03,11,12,13,22,23,33}
	   }

       for(int mA=0; mA<nm; mA++)
       	 for(int mB=0; mB<nm; mB++)
	   for(int ijack=0;ijack<njacks;ijack++)
	     cout<<eff_mass[0+nr*mA][0+nr*mB]*eff_mass[0+nr*mA][0+nr*mB]<<"\t"<<jG_0[ijack][0+nr*mA][0+nr*mB][2]<<endl;
       cout<<endl;
       
       // for(int mA=0; mA<nm; mA++)
       // 	 for(int mB=0; mB<nm; mB++)
       // 	   cout<<eff_mass[1+nr*mA][1+nr*mB]*eff_mass[1+nr*mA][1+nr*mB]<<"\t"<<jG_0[0][1+nr*mA][1+nr*mB][2]<<endl;
       // cout<<endl;
       
       

       vd_t Gp_ave(0.0,neq), sqr_Gp_ave(0.0,neq), Gp_err(0.0,neq);
       vd_t Gs_ave(0.0,neq), sqr_Gs_ave(0.0,neq), Gs_err(0.0,neq);
       
       for(int i=0;i<neq;i++)
	 {
	   for(int ijack=0;ijack<njacks;ijack++)
	     {
	       Gp_ave[i]+=jG_p_equivalent[ijack][i]/njacks;
	       sqr_Gp_ave[i]+=jG_p_equivalent[ijack][i]*jG_p_equivalent[ijack][i]/njacks;

	       Gs_ave[i]+=jG_s_equivalent[ijack][i]/njacks;
	       sqr_Gs_ave[i]+=jG_s_equivalent[ijack][i]*jG_s_equivalent[ijack][i]/njacks;
	     }
	   Gp_err[i]=sqrt((double)(njacks-1))*sqrt(sqr_Gp_ave[i]-Gp_ave[i]*Gp_ave[i]);
	   Gs_err[i]=sqrt((double)(njacks-1))*sqrt(sqr_Gs_ave[i]-Gs_ave[i]*Gs_ave[i]);      
	 }


       // for(int i=0;i<neq;i++) cout<<m_eff_equivalent[i]*m_eff_equivalent[i]<<"  "<<Gp_ave[i]<<"  "<<Gp_err[i]<<endl; //////////
       // cout<<endl;
      
       //range for the fit
       int t_min=0;
       int t_max=neq-1;

       vvd_t coord(vd_t(0.0,neq),3);
       for(int i=0; i<neq; i++)
	 {
	   coord[0][i] = 1.0;  //costante
	   coord[1][i] = m_eff_equivalent[i]*m_eff_equivalent[i];   //M^2 
	   coord[2][i] = 1.0/(m_eff_equivalent[i]*m_eff_equivalent[i]);  //1/M^2	   
	 }

       valarray<VectorXd> jack_Gp_pars=fit_par_jackknife(coord,Gp_err,jG_p_equivalent,t_min,t_max);  //jack_Gp_pars[ijack][par]
       valarray<VectorXd> jack_Gs_pars=fit_par_jackknife(coord,Gs_err,jG_s_equivalent,t_min,t_max);

       vd_t C_p(njacks), C_s(njacks);

       for(int ijack=0;ijack<njacks;ijack++)
	 {
	   C_p[ijack]=jack_Gp_pars[ijack][2];
	   C_s[ijack]=jack_Gs_pars[ijack][2];
	 }

       // for(int ijack=0;ijack<njacks;ijack++) cout<<jack_Gp_pars[ijack][0]<<"  "<<jack_Gp_pars[ijack][1]<<"  "<<C_p[ijack]<<endl;
       // cout<<endl;
  
       vvd_t jG_p_subpole(vd_t(neq),njacks), jG_s_subpole(vd_t(neq),njacks);

       for(int ijack=0;ijack<njacks;ijack++)
	 for(int i=0; i<neq; i++)
	   jG_p_subpole[ijack][i] = jG_p_equivalent[ijack][i] - C_p[ijack]/(m_eff_equivalent[i]*m_eff_equivalent[i]);



  // for(int mA=0; mA<nm; mA++)
  //   for(int mB=mA; mB<nm; mB++)
  //     {
  // 	cout<<"m"<<mA<<" m"<<mB<<"   ";
  // 	for(int r=0;r<nr;r++)
  // 	  {
  // 	    cout<<"m"<<mA<<"r"<<r<<" "<<eff_mass[r+nr*mA][r+nr*mB]<<"  ";
  // 	    cout<<"m"<<mB<<"r"<<r<<" "<<eff_mass[r+nr*mB][r+nr*mA]<<"  ";
  // 	    cout<<"m"<<mA<<"r"<<"MIX"<<" "<<eff_mass[r+nr*mA][(r+1)%2+nr*mB]<<"  ";
  // 	    cout<<"m"<<mB<<"r"<<"MIX"<<" "<<eff_mass[r+nr*mB][(r+1)%2+nr*mA]<<"  ";
  // 	  }
  // 	cout<<endl<<endl;
  //     }
  

  // for(int mA=0; mA<nm; mA++)
  //   for(int mB=mA; mB<nm; mB++)
  //     {
  // 	cout<<"m"<<mA<<" m"<<mB<<"  ";
  // 	for(int r=0;r<nr;r++)
  // 	 cout<<eff_mass[r+nr*mA][(r+1)%2+nr*mB]<<"  "<<eff_mass[r+nr*mB][(r+1)%2+nr*mA]<<"  ";
  // 	cout<<endl<<endl;
  //     }
  
  //jG_0[ijack][r+nr*mA][r+nr*mB][2]+jG_0[ijack][r+nr*mB][r+nr*mA][2]
  
       // for(int i=0; i<neq; i++)
       // 	 cout<<m_eff_equivalent[i]*m_eff_equivalent[i]<<"\t"<< jG_p_equivalent[0][i]<<"\t"<< jG_p_subpole[0][i]<<endl;
       // cout<<endl;
    
   
  //  cout<<"\t"<<jZq_sub[0][0]<<endl<<endl;

  // for(int ijack=0;ijack<njacks;ijack++)
  //   cout<< jG_0[ijack][0][0][2]<<endl;

  // vd_t Gp_subpole(neq);

  // for(int i=0;i<neq;i++)
  //   for(int ijack=0; ijack<njacks; ijack++)
  //     Gp_subpole[i]+=jG_p_subpole[ijack][i]/njacks;
    
  // for(int i=0;i<neq;i++)  cout<<m_eff_equivalent[i]*m_eff_equivalent[i]<<"\t"<< Gp_subpole[i]<<endl;

  
       // for(int ijack=0;ijack<njacks;ijack++)
       // 	 for(int j=0; j<neq; j++)
	   
       // // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
       // // 	 for(int mr_bw=0;mr_bw<nmr;mr_bw++)

       // 	   jG_p_fit_parameters[ijack][i]=fit_par(coord,error[mr_fw][mr_bw],jG_p_equivalent[mr_fw][mr_bw],t_min,t_max); 

       // for(int j=0; j<nmr*nmr; j++)
       // 	 cout<<j<<"  "<<j%nmr<<"  "<<(j-(j%nmr))/nmr<<endl;

       // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
       // 	 for(int mr_bw=0;mr_bw<nmr;mr_bw++)
       // 	   if(mr_bw%nr==mr_fw%nr)
       // 	     cout<<eff_mass[mr_fw][mr_bw]*eff_mass[mr_fw][mr_bw]<<"\t"<<jG_0_sub[0][mr_fw][mr_bw][2]<<"\t"<<jG_0_sub[1][mr_fw][mr_bw][2]<<endl;

       // cout<<eff_mass[0+4*0][0+4*0]<<"   "<<eff_mass[1+4*0][1+4*0]<<"     m0r0_m0r0  m0r1_m0r1"<<endl;
       // cout<<eff_mass[0+4*1][0+4*0]<<"   "<<eff_mass[0+4*0][0+4*1]<<"     m1r0_m0r0  m0r0_m1r0"<<endl;
       // cout<<eff_mass[0+4*1][0+4*0]<<"   "<<eff_mass[1+4*0][1+4*1]<<"     m1r0_m0r0  m0r1_m1r1"<<endl;
       // cout<<eff_mass[0+4*1][0+4*0]<<"   "<<eff_mass[1+4*1][1+4*0]<<"     m1r0_m0r0  m1r1_m0r1"<<endl;
       
       

     }//moms loop
    
  
  // //Create new extended vector
  
  // cout<<"Creating the extended vector..."<<endl;
  
  // vvvd_t new_list(vvd_t(vd_t(0.0,19),mom_list.size()),njacks);
  // vvvd_t new_list_em(vvd_t(vd_t(0.0,19),mom_list.size()),njacks);
  
  // //****************************************************************************************//
  // //---0---1---2---3---4---5---6---7---8---9---10---11---12---13---14---15---16---17---18---//
  // //---nT--nx--ny--nz--p2--tag-Zq--ZS--ZA--ZP--ZV---ZT---p4---Sig1-ZS1--ZA1--ZP1--ZV1--ZT1--//
  // //****************************************************************************************//  

  // for(int ijack=0;ijack<njacks;ijack++)
  //   for(size_t imom=0;imom<mom_list.size();imom++)
  //     {
  // 	for(int i=0;i<4;i++)
  // 	  new_list[ijack][imom][i]=mom_list[imom][i];
  // 	new_list[ijack][imom][4]=p2[imom];
  // 	new_list[ijack][imom][6]=jZq[ijack][imom].real();
  // 	for(int i=0;i<5;i++) new_list[ijack][imom][7+i]=jZ[ijack][imom][0+i].real();
  // 	new_list[ijack][imom][12]=p4[imom];
  // 	new_list[ijack][imom][13]=jSigma1[ijack][imom].real();
  // 	for(int i=0;i<5;i++) new_list[ijack][imom][14+i]=jZ1[ijack][imom][0+i].real();	
  //     }

  // for(int ijack=0;ijack<njacks;ijack++)
  //   for(size_t imom=0;imom<mom_list.size();imom++)
  //     {
  // 	for(int i=0;i<4;i++)
  // 	  new_list_em[ijack][imom][i]=mom_list[imom][i];
  // 	new_list_em[ijack][imom][4]=p2[imom];
  // 	new_list_em[ijack][imom][6]=jZq_em[ijack][imom].real();
  // 	for(int i=0;i<5;i++) new_list_em[ijack][imom][7+i]=jZ_em[ijack][imom][0+i].real();
  // 	new_list_em[ijack][imom][12]=p4[imom];
  // 	new_list_em[ijack][imom][13]=jSigma1_em[ijack][imom].real();
  // 	for(int i=0;i<5;i++) new_list_em[ijack][imom][14+i]=jZ1_em[ijack][imom][0+i].real();	
  //     }
 
 
  // //Assign the tag for fixed ijack
  // int tag=0;
  // double eps=1.0e-15;  //Precision: is it correct?
  
  // //LO
  // for(size_t imom=0;imom<mom_list.size();imom++)
  //   {
  //     size_t count=0;
  //     for(size_t i=0;i<imom;i++)
  // 	{
  // 	  if((abs(new_list[0][i][4]-new_list[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_list[i][0]-new_list[imom][0])<eps*/ && \
  // 	      abs(abs(new_list[0][i][1])*abs(new_list[0][i][2])*abs(new_list[0][i][3])-(abs(new_list[0][imom][1])*abs(new_list[0][imom][2])*abs(new_list[0][imom][3])))<eps ) || \
  // 	     (abs(new_list[0][i][4]-new_list[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_list[i][0]+new_list[imom][0]+1.)<eps*/ && \
  // 	      abs(abs(new_list[0][i][1])*abs(new_list[0][i][2])*abs(new_list[0][i][3])-(abs(new_list[0][imom][1])*abs(new_list[0][imom][2])*abs(new_list[0][imom][3])))<eps ))
  // 	    {
  // 	      new_list[0][imom][5]=new_list[0][i][5];
  // 	    }else count++;
	  
  // 	  if(count==imom)
  // 	    {
  // 	      tag++;
  // 	      new_list[0][imom][5]=tag;
  // 	    }
  // 	}
  //   }
  
  // for(size_t imom=0;imom<mom_list.size();imom++)
  //   for(int ijack=1;ijack<njacks;ijack++)
  //     new_list[ijack][imom][5]=new_list[0][imom][5];
  
  // //em corrections
  // int tag_em=0;
  // for(size_t imom=0;imom<mom_list.size();imom++)
  //   {
  //     size_t count=0;
  //     for(size_t i=0;i<imom;i++)
  // 	{
  // 	  if((abs(new_list_em[0][i][4]-new_list_em[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_list_em[i][0]-new_list_em[imom][0])<eps*/ && \
  // 	      abs(abs(new_list_em[0][i][1])*abs(new_list_em[0][i][2])*abs(new_list_em[0][i][3])-(abs(new_list_em[0][imom][1])*abs(new_list_em[0][imom][2])*abs(new_list_em[0][imom][3])))<eps ) || \
  // 	     (abs(new_list_em[0][i][4]-new_list_em[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_list_em[i][0]+new_list_em[imom][0]+1.)<eps*/ && \
  // 	      abs(abs(new_list_em[0][i][1])*abs(new_list_em[0][i][2])*abs(new_list_em[0][i][3])-(abs(new_list_em[0][imom][1])*abs(new_list_em[0][imom][2])*abs(new_list_em[0][imom][3])))<eps ))
  // 	    {
  // 	      new_list_em[0][imom][5]=new_list_em[0][i][5];
  // 	    }else count++;
	  
  // 	  if(count==imom)
  // 	    {
  // 	      tag_em++;
  // 	      new_list_em[0][imom][5]=tag_em;
  // 	    }
  // 	}
  //   }
  
  // for(size_t imom=0;imom<mom_list.size();imom++)
  //   for(int ijack=1;ijack<njacks;ijack++)
  //     new_list_em[ijack][imom][5]=new_list_em[0][imom][5];
  
  // cout<<"Number of equivalent momenta: "<<tag+1<<endl;



   
  
  // //Average of Z's corresponding to equivalent momenta (same tag) and print on file
  // vd_t p2_eq(vd_t(0.0,tag+1));
  // vd_t p4_eq(vd_t(0.0,tag+1)); //for the democratic filter
  
  // valarray< valarray< valarray< vector<double> > > > jZ_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks); //Zq RI'-MOM
  // vvvd_t jZ_average(vvd_t(vd_t(0.0,6),tag+1),njacks);
  
  // valarray< valarray< valarray< vector<double> > > > jZ1_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks);  // sigma1
  // vvvd_t jZ1_average(vvd_t(vd_t(0.0,6),tag+1),njacks);

  // valarray< valarray< valarray< vector<double> > > > jZ_em_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks); //Zq RI'-MOM
  // vvvd_t jZ_em_average(vvd_t(vd_t(0.0,6),tag+1),njacks);
  
  // valarray< valarray< valarray< vector<double> > > > jZ1_em_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks);  // sigma1
  // vvvd_t jZ1_em_average(vvd_t(vd_t(0.0,6),tag+1),njacks);
  
  // cout<<"Averaging the Z's corresponding to equivalent momenta and printing on the output file..."<<endl;

  // for(int ijack=0;ijack<njacks;ijack++)
  //   for(int t=0;t<=tag;t++)
  //     {
  // 	int count_equivalent=0;
  // 	for(size_t imom=0;imom<mom_list.size();imom++)
  // 	  { 
  // 	    if(t==new_list[ijack][imom][5])
  // 	      {
  // 		count_equivalent++;
  // 		p2_eq[t]=new_list[0][imom][4];
  // 		p4_eq[t]=new_list[0][imom][12];
		
  // 		jZ_same_tag[ijack][t][0].push_back(new_list[ijack][imom][6]);  //LO
  // 		jZ_same_tag[ijack][t][1].push_back(new_list[ijack][imom][7]);
  // 		jZ_same_tag[ijack][t][2].push_back(new_list[ijack][imom][8]);
  // 		jZ_same_tag[ijack][t][3].push_back(new_list[ijack][imom][9]);
  // 		jZ_same_tag[ijack][t][4].push_back(new_list[ijack][imom][10]);
  // 		jZ_same_tag[ijack][t][5].push_back(new_list[ijack][imom][11]);

  // 		jZ1_same_tag[ijack][t][0].push_back(new_list[ijack][imom][13]);  
  // 		jZ1_same_tag[ijack][t][1].push_back(new_list[ijack][imom][14]);
  // 		jZ1_same_tag[ijack][t][2].push_back(new_list[ijack][imom][15]);
  // 		jZ1_same_tag[ijack][t][3].push_back(new_list[ijack][imom][16]);
  // 		jZ1_same_tag[ijack][t][4].push_back(new_list[ijack][imom][17]);
  // 		jZ1_same_tag[ijack][t][5].push_back(new_list[ijack][imom][18]);

  // 		jZ_em_same_tag[ijack][t][0].push_back(new_list_em[ijack][imom][6]);  //em
  // 		jZ_em_same_tag[ijack][t][1].push_back(new_list_em[ijack][imom][7]);
  // 		jZ_em_same_tag[ijack][t][2].push_back(new_list_em[ijack][imom][8]);
  // 		jZ_em_same_tag[ijack][t][3].push_back(new_list_em[ijack][imom][9]);
  // 		jZ_em_same_tag[ijack][t][4].push_back(new_list_em[ijack][imom][10]);
  // 		jZ_em_same_tag[ijack][t][5].push_back(new_list_em[ijack][imom][11]);

  // 		jZ1_em_same_tag[ijack][t][0].push_back(new_list_em[ijack][imom][13]); 
  // 		jZ1_em_same_tag[ijack][t][1].push_back(new_list_em[ijack][imom][14]);
  // 		jZ1_em_same_tag[ijack][t][2].push_back(new_list_em[ijack][imom][15]);
  // 		jZ1_em_same_tag[ijack][t][3].push_back(new_list_em[ijack][imom][16]);
  // 		jZ1_em_same_tag[ijack][t][4].push_back(new_list_em[ijack][imom][17]);
  // 		jZ1_em_same_tag[ijack][t][5].push_back(new_list_em[ijack][imom][18]);	

		
  // 	      }
  // 	  }
	
  // 	for(int i=0;i<count_equivalent;i++)  //average over the equivalent Z's in each jackknife
  // 	  { 
  // 	    jZ_average[ijack][t][0]+=jZ_same_tag[ijack][t][0][i]/count_equivalent; //LO
  // 	    jZ_average[ijack][t][1]+=jZ_same_tag[ijack][t][1][i]/count_equivalent;
  // 	    jZ_average[ijack][t][2]+=jZ_same_tag[ijack][t][2][i]/count_equivalent;
  // 	    jZ_average[ijack][t][3]+=jZ_same_tag[ijack][t][3][i]/count_equivalent;
  // 	    jZ_average[ijack][t][4]+=jZ_same_tag[ijack][t][4][i]/count_equivalent;
  // 	    jZ_average[ijack][t][5]+=jZ_same_tag[ijack][t][5][i]/count_equivalent;

  // 	    jZ1_average[ijack][t][0]+=jZ1_same_tag[ijack][t][0][i]/count_equivalent; 
  // 	    jZ1_average[ijack][t][1]+=jZ1_same_tag[ijack][t][1][i]/count_equivalent;
  // 	    jZ1_average[ijack][t][2]+=jZ1_same_tag[ijack][t][2][i]/count_equivalent;
  // 	    jZ1_average[ijack][t][3]+=jZ1_same_tag[ijack][t][3][i]/count_equivalent;
  // 	    jZ1_average[ijack][t][4]+=jZ1_same_tag[ijack][t][4][i]/count_equivalent;
  // 	    jZ1_average[ijack][t][5]+=jZ1_same_tag[ijack][t][5][i]/count_equivalent;

	    
  // 	    jZ_em_average[ijack][t][0]+=jZ_em_same_tag[ijack][t][0][i]/count_equivalent;  //em
  // 	    jZ_em_average[ijack][t][1]+=jZ_em_same_tag[ijack][t][1][i]/count_equivalent;
  // 	    jZ_em_average[ijack][t][2]+=jZ_em_same_tag[ijack][t][2][i]/count_equivalent;
  // 	    jZ_em_average[ijack][t][3]+=jZ_em_same_tag[ijack][t][3][i]/count_equivalent;
  // 	    jZ_em_average[ijack][t][4]+=jZ_em_same_tag[ijack][t][4][i]/count_equivalent;
  // 	    jZ_em_average[ijack][t][5]+=jZ_em_same_tag[ijack][t][5][i]/count_equivalent;

  // 	    jZ1_em_average[ijack][t][0]+=jZ1_em_same_tag[ijack][t][0][i]/count_equivalent; 
  // 	    jZ1_em_average[ijack][t][1]+=jZ1_em_same_tag[ijack][t][1][i]/count_equivalent;
  // 	    jZ1_em_average[ijack][t][2]+=jZ1_em_same_tag[ijack][t][2][i]/count_equivalent;
  // 	    jZ1_em_average[ijack][t][3]+=jZ1_em_same_tag[ijack][t][3][i]/count_equivalent;
  // 	    jZ1_em_average[ijack][t][4]+=jZ1_em_same_tag[ijack][t][4][i]/count_equivalent;
  // 	    jZ1_em_average[ijack][t][5]+=jZ1_em_same_tag[ijack][t][5][i]/count_equivalent;	    
  // 	  }
  //     }

  // vvd_t Z_mean_value(vd_t(0.0,6),tag+1);    //LO
  // vvd_t Z2_mean_value(vd_t(0.0,6),tag+1);
  // vvd_t Z_error(vd_t(0.0,6),tag+1);

  // vvd_t Z1_mean_value(vd_t(0.0,6),tag+1);  
  // vvd_t Z21_mean_value(vd_t(0.0,6),tag+1);
  // vvd_t Z1_error(vd_t(0.0,6),tag+1);

  // vvd_t Z_em_mean_value(vd_t(0.0,6),tag+1);    //em
  // vvd_t Z2_em_mean_value(vd_t(0.0,6),tag+1);
  // vvd_t Z_em_error(vd_t(0.0,6),tag+1);

  // vvd_t Z1_em_mean_value(vd_t(0.0,6),tag+1); 
  // vvd_t Z21_em_mean_value(vd_t(0.0,6),tag+1);
  // vvd_t Z1_em_error(vd_t(0.0,6),tag+1);
  
  // for(int t=0;t<=tag;t++)
  //   for(int i=0;i<6;i++)
  //     {
  // 	for(int ijack=0;ijack<njacks;ijack++)
  // 	  {
  // 	    Z_mean_value[t][i]+=jZ_average[ijack][t][i]/njacks;
  // 	    Z2_mean_value[t][i]+=jZ_average[ijack][t][i]*jZ_average[ijack][t][i]/njacks;
  // 	  }
  // 	Z_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z2_mean_value[t][i]-Z_mean_value[t][i]*Z_mean_value[t][i]);
  //     }

  // for(int t=0;t<=tag;t++)  //PROVA: sigma1
  //   for(int i=0;i<6;i++)
  //     {
  // 	for(int ijack=0;ijack<njacks;ijack++)
  // 	  {
  // 	    Z1_mean_value[t][i]+=jZ1_average[ijack][t][i]/njacks;
  // 	    Z21_mean_value[t][i]+=jZ1_average[ijack][t][i]*jZ1_average[ijack][t][i]/njacks;
  // 	  }
  // 	Z1_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z21_mean_value[t][i]-Z1_mean_value[t][i]*Z1_mean_value[t][i]);
  //     }

  // for(int t=0;t<=tag;t++)
  //   for(int i=0;i<6;i++)
  //     {
  // 	for(int ijack=0;ijack<njacks;ijack++)
  // 	  {
  // 	    Z_em_mean_value[t][i]+=jZ_em_average[ijack][t][i]/njacks;
  // 	    Z2_em_mean_value[t][i]+=jZ_em_average[ijack][t][i]*jZ_em_average[ijack][t][i]/njacks;
  // 	  }
  // 	Z_em_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z2_em_mean_value[t][i]-Z_em_mean_value[t][i]*Z_em_mean_value[t][i]);
  //     }

  // for(int t=0;t<=tag;t++)  //PROVA: sigma1
  //   for(int i=0;i<6;i++)
  //     {
  // 	for(int ijack=0;ijack<njacks;ijack++)
  // 	  {
  // 	    Z1_em_mean_value[t][i]+=jZ1_em_average[ijack][t][i]/njacks;
  // 	    Z21_em_mean_value[t][i]+=jZ1_em_average[ijack][t][i]*jZ1_em_average[ijack][t][i]/njacks;
  // 	  }
  // 	Z1_em_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z21_em_mean_value[t][i]-Z1_em_mean_value[t][i]*Z1_em_mean_value[t][i]);
  //     }

  // //print on file
  
  // print_file("Z.txt",p2_eq,Z_mean_value,Z_error,tag);
  // print_file_filtered("Z_filtered.txt",p2_eq,p4_eq,Z_mean_value,Z_error,tag);
  // print_file("Z_sigma1.txt",p2_eq,Z1_mean_value,Z1_error,tag);
  // print_file_filtered("Z_filtered_sigma1.txt",p2_eq,p4_eq,Z1_mean_value,Z1_error,tag);

  // print_file("Z_em.txt",p2_eq,Z_em_mean_value,Z_em_error,tag);
  // print_file_filtered("Z_em_filtered.txt",p2_eq,p4_eq,Z_em_mean_value,Z_em_error,tag);
  // print_file("Z_em_sigma1.txt",p2_eq,Z1_em_mean_value,Z1_em_error,tag);
  // print_file_filtered("Z_em_filtered_sigma1.txt",p2_eq,p4_eq,Z1_em_mean_value,Z1_em_error,tag);
  

  // cout<<"Averaging the corrected Z's corresponding to equivalent momenta and printing on the output file..."<<endl;

  // valarray< valarray< valarray< vector<double> > > > jZ_corr_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks);
  // vvvd_t jZ_corr_average(vvd_t(vd_t(0.0,6),tag+1),njacks);

  // for(int ijack=0;ijack<njacks;ijack++)
  //   for(int t=0;t<=tag;t++)
  //     {
  // 	int count_equivalent=0;
  // 	for(size_t imom=0;imom<mom_list.size();imom++)
  // 	  { 
  // 	    if(t==new_list[ijack][imom][5])
  // 	      {
  // 		count_equivalent++;
  // 		p2_eq[t]=new_list[0][imom][4];
  // 		p4_eq[t]=new_list[0][imom][12];
		
  // 		jZ_corr_same_tag[ijack][t][0].push_back(new_list_corr[ijack][imom][6]);
  // 		jZ_corr_same_tag[ijack][t][1].push_back(new_list_corr[ijack][imom][7]);
  // 		jZ_corr_same_tag[ijack][t][2].push_back(new_list_corr[ijack][imom][8]);
  // 		jZ_corr_same_tag[ijack][t][3].push_back(new_list_corr[ijack][imom][9]);
  // 		jZ_corr_same_tag[ijack][t][4].push_back(new_list_corr[ijack][imom][10]);
  // 		jZ_corr_same_tag[ijack][t][5].push_back(new_list_corr[ijack][imom][11]);	
  // 	      }
  // 	  }
	
  // 	for(int i=0;i<count_equivalent;i++)  //average over the equivalent Z's in each jackknife
  // 	  { 
  // 	    jZ_corr_average[ijack][t][0]+=jZ_corr_same_tag[ijack][t][0][i]/count_equivalent;
  // 	    jZ_corr_average[ijack][t][1]+=jZ_corr_same_tag[ijack][t][1][i]/count_equivalent;
  // 	    jZ_corr_average[ijack][t][2]+=jZ_corr_same_tag[ijack][t][2][i]/count_equivalent;
  // 	    jZ_corr_average[ijack][t][3]+=jZ_corr_same_tag[ijack][t][3][i]/count_equivalent;
  // 	    jZ_corr_average[ijack][t][4]+=jZ_corr_same_tag[ijack][t][4][i]/count_equivalent;
  // 	    jZ_corr_average[ijack][t][5]+=jZ_corr_same_tag[ijack][t][5][i]/count_equivalent;
  // 	  }
  //     }

  // vvd_t Z_corr_mean_value(vd_t(0.0,6),tag+1);
  // vvd_t Z2_corr_mean_value(vd_t(0.0,6),tag+1);
  // vvd_t Z_corr_error(vd_t(0.0,6),tag+1);
  
  // for(int t=0;t<=tag;t++)
  //   for(int i=0;i<6;i++)
  //     {
  // 	for(int ijack=0;ijack<njacks;ijack++)
  // 	  {
  // 	    Z_corr_mean_value[t][i]+=jZ_corr_average[ijack][t][i]/njacks;
  // 	    Z2_corr_mean_value[t][i]+=jZ_corr_average[ijack][t][i]*jZ_corr_average[ijack][t][i]/njacks;
  // 	  }
  // 	Z_corr_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z2_corr_mean_value[t][i]-Z_corr_mean_value[t][i]*Z_corr_mean_value[t][i]);
  //     }

  // //print on file
  // print_file("Z_corrected.txt",p2_eq,Z_corr_mean_value,Z_corr_error,tag);
  // print_file_filtered("Z_corrected_filtered.txt",p2_eq,p4_eq,Z_corr_mean_value,Z_corr_error,tag);
  

  // //Fit parameters before and after the correction
  
    
  // vvvd_t jfit_parameters=compute_fit_parameters_filtered(p2_eq,p4_eq,jZ_average,Z_error,tag,njacks,p2fit_min,p2fit_max);
  // vvvd_t jfit_parameters_corr=compute_fit_parameters_filtered(p2_eq,p4_eq,jZ_corr_average,Z_corr_error,tag,njacks,p2fit_min,p2fit_max);

  // vd_t A(0.0,6), B(0.0,6), A_error(0.0,6), B_error(0.0,6), A2(0.0,6), B2(0.0,6);
  // vd_t A_corr(0.0,6), B_corr(0.0,6), A_corr_error(0.0,6), B_corr_error(0.0,6), A2_corr(0.0,6), B2_corr(0.0,6);
  
  // for(int iZ=0;iZ<6;iZ++)
  //   {
  //     for(int ijack=0;ijack<njacks;ijack++)
  // 	{
  // 	  A[iZ]+=jfit_parameters[ijack][iZ][0]/njacks;
  // 	  A2[iZ]+=jfit_parameters[ijack][iZ][0]*jfit_parameters[ijack][iZ][0]/njacks;
  // 	  B[iZ]+=jfit_parameters[ijack][iZ][1]/njacks;
  // 	  B2[iZ]+=jfit_parameters[ijack][iZ][1]*jfit_parameters[ijack][iZ][1]/njacks;
	  
	  
  // 	  A_corr[iZ]+=jfit_parameters_corr[ijack][iZ][0]/njacks;
  // 	  A2_corr[iZ]+=jfit_parameters_corr[ijack][iZ][0]*jfit_parameters_corr[ijack][iZ][0]/njacks;
  // 	  B_corr[iZ]+=jfit_parameters_corr[ijack][iZ][1]/njacks;
  // 	  B2_corr[iZ]+=jfit_parameters_corr[ijack][iZ][1]*jfit_parameters_corr[ijack][iZ][1]/njacks;
  // 	}
  //     A_error[iZ]=sqrt((double)(njacks-1))*sqrt(A2[iZ]-A[iZ]*A[iZ]);
  //     B_error[iZ]=sqrt((double)(njacks-1))*sqrt(B2[iZ]-B[iZ]*B[iZ]);

  //     A_corr_error[iZ]=sqrt((double)(njacks-1))*sqrt(A2_corr[iZ]-A_corr[iZ]*A_corr[iZ]);
  //     B_corr_error[iZ]=sqrt((double)(njacks-1))*sqrt(B2_corr[iZ]-B_corr[iZ]*B_corr[iZ]);
      
  //   }
  
  // cout<<" "<<endl;
  // cout<<"Fit parameters BEFORE the correction for the filtered data: y=a*x+b"<<endl;
  // cout<<"-------------------------------------------------------------------"<<endl;
  // cout<<"     ZQ     "<<endl;
  // cout<<"a = "<<A[0]<<" +/- "<<A_error[0]<<endl;
  // cout<<"b = "<<B[0]<<" +/- "<<B_error[0]<<endl;
  // cout<<"     ZS     "<<endl;
  // cout<<"a = "<<A[1]<<" +/- "<<A_error[1]<<endl;
  // cout<<"b = "<<B[1]<<" +/- "<<B_error[1]<<endl;
  // cout<<"     ZA     "<<endl;
  // cout<<"a = "<<A[2]<<" +/- "<<A_error[2]<<endl;
  // cout<<"b = "<<B[2]<<" +/- "<<B_error[2]<<endl;
  // cout<<"     ZP     "<<endl;
  // cout<<"a = "<<A[3]<<" +/- "<<A_error[3]<<endl;
  // cout<<"b = "<<B[3]<<" +/- "<<B_error[3]<<endl;
  // cout<<"     ZV     "<<endl;
  // cout<<"a = "<<A[4]<<" +/- "<<A_error[4]<<endl;
  // cout<<"b = "<<B[4]<<" +/- "<<B_error[4]<<endl;
  // cout<<"     ZT     "<<endl;
  // cout<<"a = "<<A[5]<<" +/- "<<A_error[5]<<endl;
  // cout<<"b = "<<B[5]<<" +/- "<<B_error[5]<<endl;
    
  // cout<<" "<<endl;
  // cout<<"Fit parameters AFTER the correction for the filtered data: y=a*x+b"<<endl;
  // cout<<"-------------------------------------------------------------------"<<endl;
  // cout<<"     ZQ     "<<endl;
  // cout<<"a = "<<A_corr[0]<<" +/- "<<A_corr_error[0]<<endl;
  // cout<<"b = "<<B_corr[0]<<" +/- "<<B_corr_error[0]<<endl;
  // cout<<"     ZS     "<<endl;
  // cout<<"a = "<<A_corr[1]<<" +/- "<<A_corr_error[1]<<endl;
  // cout<<"b = "<<B_corr[1]<<" +/- "<<B_corr_error[1]<<endl;
  // cout<<"     ZA     "<<endl;
  // cout<<"a = "<<A_corr[2]<<" +/- "<<A_corr_error[2]<<endl;
  // cout<<"b = "<<B_corr[2]<<" +/- "<<B_corr_error[2]<<endl;
  // cout<<"     ZP     "<<endl;
  // cout<<"a = "<<A_corr[3]<<" +/- "<<A_corr_error[3]<<endl;
  // cout<<"b = "<<B_corr[3]<<" +/- "<<B_corr_error[3]<<endl;
  // cout<<"     ZV     "<<endl;
  // cout<<"a = "<<A_corr[4]<<" +/- "<<A_corr_error[4]<<endl;
  // cout<<"b = "<<B_corr[4]<<" +/- "<<B_corr_error[4]<<endl;
  // cout<<"     ZT     "<<endl;
  // cout<<"a = "<<A_corr[5]<<" +/- "<<A_corr_error[5]<<endl;
  // cout<<"b = "<<B_corr[5]<<" +/- "<<B_corr_error[5]<<endl;
    
  
  // cout<<"End of the program."<<endl;
  
  return 0;

      }
