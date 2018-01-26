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

#define lambdaQCD 0.250
#define Z3 1.20206
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

//create Dirac Gamma matrices
vprop_t make_gamma()
{
  int NGamma=4;
  
  vprop_t gam(NGamma);
  vector< vector<int> > col(NGamma, vector<int>(4));
  vector< vector<int> > real_part(NGamma, vector<int>(4));
  vector< vector<int> > im_part(NGamma, vector<int>(4));
    
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

  for(int i_gam=0;i_gam<5;i_gam++)
    for(int i_row=0;i_row<4;i_row++)
      for(int ic=0;ic<3;ic++)
	{
	  gam[i_gam](isc(i_row,ic),isc(col[i_gam][i_row],ic))=dcompl(real_part[i_gam][i_row],im_part[i_gam][i_row] );
	}
  
  return gam;
}





/***********************************************************************************/
/*************************************** main **************************************/
/***********************************************************************************/

  
int main(int narg,char **arg)
{
#pragma omp parallel
#pragma omp master
  cout<<"Using "<<omp_get_num_threads()<<" threads"<<endl;

  high_resolution_clock::time_point t_START=high_resolution_clock::now();
  
  high_resolution_clock::time_point t0=high_resolution_clock::now();
  
  if (narg!=12){
    cerr<<"Number of arguments not valid: <mom file> <nconfs> <njacks> <L> <T> <initial conf_id> <step conf_id> <p2fit min> <p2fit max> <action=sym/iwa/free> <path before 'out' directory: /marconi_work/.../ >"<<endl;
    exit(0);
  }

  int nconfs=stoi(arg[2]); 
  int njacks=stoi(arg[3]);
  int clust_size=nconfs/njacks;
  int conf_id[nconfs];
  double L=stod(arg[4]),T=stod(arg[5]);
  size_t nhits=1; //!

  nm = 4;  //! to be passed from command line
  nr = 2;

  for(int iconf=0;iconf<nconfs;iconf++)
    conf_id[iconf]=stoi(arg[6])+iconf*stoi(arg[7]);

   cout<<"N confs = "<<nconfs<<endl;
  cout<<"N jacks = "<<njacks<<endl;
  cout<<"Clust size = "<<clust_size<<endl;
  cout<<"L = "<<L<<"\t T = "<<T<<endl;
  cout<<"Fit range = ["<<p2fit_min<<":"<<p2fit_max<<"]"<<endl;
  

  double beta=0.0, plaquette=0.0;
  vector<double> c_v(3), c_a(3), c_s(3), c_p(3), c_t(3);
  vector<double> c_v_em(3), c_a_em(3), c_s_em(3), c_p_em(3), c_t_em(3);
  
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

      c_a={1.5240798/4.,-1./3./4.,-125./288./4.};  //we divide by 4 (or 6 for T) to account for the sum on Lorentz indices
      c_v={0.6999177/4.,-1./3./4.,-125./288./4.};
      c_s={2.3547298,-1./4.,0.5};        
      c_p={0.70640549,-1./4.,0.5};
      c_t={0.9724758/6.,-13./36./6.,-161./216./6.};

      // c_a_em={0.3997992,1./16.,-13./48.};          // Wilson Action with Landau gauge
      c_a_em={0.3997992/4.,1./16./4.,-1./4./4.};
      // c_v_em={0.2394370,-3./16.,-1./4.};
      c_v_em={0.2394365/4.,-3./16./4.,-1./4./4.};
      c_s_em={0.32682365,1./2.,5./12.};        
      c_p_em={0.00609817,0.,5./12.};
      c_t_em={0.3706701/6.,-1./6./6.,-17./36./6.};

      /*   c_a_em={1000.,1000.,1000.};
      c_v_em={1000.,1000.,1000.};
      c_s_em={1000.,1000.,1000.};        
      c_p_em={1000.,1000.,1000.};
      c_t_em={1000.,1000.,1000.};*/

      cout<<"Action:  Symanzik"<<endl;
    }
   else if(strcmp(arg[10],"free")==0)  //Nf=2 (Symanzik)
    {
      // beta=99999999999999.9;
      beta = 1.0e300;
      plaquette=1.0;

      c_v={0.0,0.0,0.0};
      c_a={0.0,0.0,0.0};
      c_s={0.0,0.0,0.0};
      c_p={0.0,0.0,0.0};
      c_t={0.0,0.0,0.0};
      
      c_v_em={0.0,0.0,0.0};
      c_a_em={0.0,0.0,0.0};
      c_s_em={0.0,0.0,0.0};
      c_p_em={0.0,0.0,0.0};
      c_t_em={0.0,0.0,0.0};

      cout<<"Action:  Free"<<endl;
    }
  else
    {
      cerr<<"WARNING: wrong action argument. Please write 'sym' for Symanzik action or 'iwa' for Iwasaki action. Write 'free' for the free action.";
      exit(0);
    }
