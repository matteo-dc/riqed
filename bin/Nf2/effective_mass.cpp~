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


//create the path-string to the contraction
string path_to_contr(int i_conf,const int mr1, const string &T1, const int mr2, const string &T2)
{

  int r1 = mr1%nr;
  int m1 = (mr1-r1)/nr;
  int r2 = mr2%nr;
  int m2 = (mr2-r2)/nr;
  
  char path[1024];
  sprintf(path,"/marconi_work/INF17_lqcd123_0/sanfo/RIQED/3.90_24_0.0100/out/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());

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

      if(!infile.good())
	{cerr<<"Unable to open file "<<path_to_contr(conf_id[iconf],mr1,T1,mr2,T2)<<endl;
	  exit(1);}

      //DEBUG
      // cout<<"  Reading contraction from "<<path_to_contr(conf_id[iconf],mr1,T1,mr2,T2)<<endl;
      //DEBUG
      
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

  // cout<<"********DEBUG*****************************"<<endl; 
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

int main(int narg,char **arg)
{

  int nconfs=240;
  int njacks=15;
  int clust_size=nconfs/njacks;
  int conf_id[nconfs];
  double L=24,T=48;
  size_t nhits=1; //!

  nm = 4;  //! to be passed from command line
  nr = 2;

  nmr=nm*nr;

  for(int iconf=0;iconf<nconfs;iconf++)
    conf_id[iconf]=100+iconf*1;

  vvvd_t eff_mass_array = compute_eff_mass(T,nconfs,njacks,conf_id);
 
  
  ofstream outfile;
  outfile.open("eff_mass_array", ios::out | ios::binary);

   if (outfile.is_open())
    {	  
      for(int mr_fw=0;mr_fw<nmr;mr_fw++)
	for(int mr_bw=0;mr_bw<nmr;mr_bw++)
	  for(int i=0;i<2;i++)
	    outfile.write((char*) &eff_mass_array[mr_fw][mr_bw][i],sizeof(double));
      
      outfile.close();
    }
   else cout << "Unable to open the output file "<<endl;

   return 0;

}
