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

//list of gamma for a given momentum
using qline_t=valarray<prop_t>;

//list of jackknife propagators
using jprop_t=valarray< valarray<prop_t> >;

//list of jackknife vertices
using jvert_t=valarray< valarray<qline_t> >;

//valarray of complex double
using vd_t=valarray<double>;

//valarray of valarray of complex double
using vvd_t=valarray< valarray<double> > ;

//valarray of valarray of valarray of complex double
using vvvd_t=valarray< valarray< valarray<double> > >;

//valarray of complex double
using vdcompl_t=valarray<dcompl>;

//valarray of valarray of complex double
using vvdcompl_t=valarray< valarray<dcompl> > ;

//valarray of valarray of valarray of complex double
using vvvdcompl_t=valarray< valarray< valarray<dcompl> > >;

//list of momenta
vector<coords_t> mom_list;

//list of N(p)
vector<int> Np;



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

//returns the linearized spin color index
size_t isc(size_t is,size_t ic)
{return ic+3*is;}

//read a propagator file
vprop_t read_prop(const string &path)
{
  vprop_t out(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  
  ifstream input(path,ios::binary);
  if(!input.good())
    {
      cerr<<"Unable to open file "<<path<<endl;
      exit(1);
    }
  
  for(int id_so=0;id_so<4;id_so++)
    for(int ic_so=0;ic_so<3;ic_so++)
      for(size_t imom=0;imom<mom_list.size();imom++)
	for(int id_si=0;id_si<4;id_si++)
	  for(int ic_si=0;ic_si<3;ic_si++)
	    {
	      double temp[2];
	      input.read((char*)&temp,sizeof(double)*2);
	      if(not input.good())
		{
		  cerr<<"Unable to read from "<<path<<" id_so: "<<id_so<<", ic_so: "<<ic_so<<", imom: "<<imom<<", id_si: "<<id_si<<", ic_si:"<<ic_si<<endl;
		  exit(1);
		}
	      out[imom](isc(id_si,ic_si),isc(id_so,ic_so))=dcompl(temp[0],temp[1]); //store
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
vprop_t make_vertex(const vprop_t &prop1, const vprop_t &prop2, size_t mom,const vprop_t &gamma)
{
  vprop_t vert(16);
  for(int mu=0;mu<16;mu++)
    {      
      vert[mu]=prop1[mom]*gamma[mu]*gamma[5]*prop2[mom].adjoint()*gamma[5];  /*it has to be "jackknifed"*/
    }
  return vert;
}

//create the path-string to the configuration
string path_to_conf(int i_conf,const char *name)
{
  char path[1024];
  sprintf(path,"out/%04d/fft_%s",i_conf,name);
  return path;
}


//jackknife Propagator
jprop_t jackknife_prop(  jprop_t &jS, int nconf, int clust_size )
{
  valarray<prop_t> jSum(prop_t::Zero(),mom_list.size());

  //sum of jS
  for(size_t j=0;j<jS.size();j++) jSum+= jS[j];
  //jackknife fluctuation
  for(size_t j=0;j<jS.size();j++)
    {
      jS[j]=jSum-jS[j];
      for(auto &it : jS[j])
      it/=nconf-clust_size;
    }

  return jS;
}

//jackknife Vertex
jvert_t jackknife_vertex( jvert_t &jVert, int nconf, int clust_size )
{
  valarray<qline_t> jSum(valarray<prop_t>(prop_t::Zero(),16),mom_list.size());
  
  //sum of the jVert
  for(size_t j=0;j<jVert.size();j++) jSum+= jVert[j];
  //jackknife fluctuation
  for(size_t j=0;j<jVert.size();j++)
    {
      jVert[j]=jSum-jVert[j];
      for(auto &it : jVert[j])
	for(auto &jt : it)
	  jt/=nconf-clust_size;
    }
  
  return jVert;
}

//invert the propagator
jprop_t inverse_jprop( const jprop_t &jprop, int njacks){

  jprop_t jprop_inv(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);
  
  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      jprop_inv[ijack][imom]=jprop[ijack][imom].inverse();
  
  return jprop_inv;
}

//amputate external legs

jvert_t amputate( const jprop_t  &jprop1_inv, const jvert_t &jV, const jprop_t  &jprop2_inv, vprop_t GAMMA, int njacks ){
  
  jvert_t jLambda(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks);
  
  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      for(int igam=0;igam<16;igam++)
	jLambda[ijack][imom][igam]=jprop1_inv[ijack][imom]*jV[ijack][imom][igam]*GAMMA[5]*jprop2_inv[ijack][imom].adjoint()*GAMMA[5];
  
  return jLambda;
}




//compute jZq
vvdcompl_t compute_jZq(vprop_t GAMMA, jprop_t jS_inv, double L, double T, int nconfs, int njacks, int cluster_size)
{
  double V=L*L*L*T;
  
  //compute p_slash as a vector of prop-type matrices
  vvd_t p(vd_t(0.0,4),mom_list.size());
  vvd_t p_tilde(vd_t(0.0,4),mom_list.size());
  vprop_t p_slash(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  vd_t p2(vd_t(0.0,mom_list.size()));
  vvdcompl_t jZq(vdcompl_t(mom_list.size()),njacks);
  dcompl I(0,1);

  

  for(size_t imom=0;imom<mom_list.size();imom++)
    {
      int count=0;
      
      p[imom]={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
      p_tilde[imom]={sin(p[imom][0]),sin(p[imom][1]),sin(p[imom][2]),sin(p[imom][3])};
      
      for(int igam=1;igam<5;igam++)
	{
	  //	p_slash[imom]+=GAMMA[igam]*p[imom][igam-1];
	  p_slash[imom]+=GAMMA[igam]*p_tilde[imom][igam-1];

	  if(p_tilde[imom][igam-1]!=0.)
	    count++;
	}

      Np.push_back(count);
      
      /*  Note that: p_slash*p_slash=p2*GAMMA[0]  */
      
      //compute p^2
      for(int coord=0;coord<4;coord++)
	//	p2[imom]+=p[imom][coord]*p[imom][coord];
	p2[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
      
    }
  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      //compute jZq = Quark field RC (RI'-MOM), one for each momentum and jackknife 
      jZq[ijack][imom]=-I*((p_slash[imom]*jS_inv[ijack][imom]).trace())/p2[imom]/12./V;
  
  return jZq;
  
}

//compute jSigma1
vvdcompl_t compute_jSigma1(vprop_t GAMMA, jprop_t jS_inv, double L, double T, int nconfs, int njacks, int cluster_size)
{
  double V=L*L*L*T;
  
  //compute p_slash as a vector of prop-type matrices
  vvd_t p(vd_t(0.0,4),mom_list.size());
  vvd_t p_tilde(vd_t(0.0,4),mom_list.size());
  vprop_t p_slash(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  vd_t p2(vd_t(0.0,mom_list.size()));
  vvdcompl_t jSigma1(valarray<dcompl>(mom_list.size()),njacks);
  dcompl I(0,1);

  valarray<valarray<prop_t>> A(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);

  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      {
	int count=0;
	
	p[imom]={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
	p_tilde[imom]={sin(p[imom][0]),sin(p[imom][1]),sin(p[imom][2]),sin(p[imom][3])};
	
	for(int igam=1;igam<5;igam++)
	  if(p_tilde[imom][igam-1]!=0.)
	    {
	      A[ijack][imom]+=GAMMA[igam]*jS_inv[ijack][imom]/p_tilde[imom][igam-1];
	      count++;
	    }
	A[ijack][imom]/=(double)count;
      }
  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      //compute jZq = Quark field RC (RI'-MOM), one for each momentum and jackknife
      jSigma1[ijack][imom]=-I*A[ijack][imom].trace()/12./V;
  
  return jSigma1;
  
}


//project the amputated green function
vvvdcompl_t project(vprop_t GAMMA, const valarray<valarray<qline_t>> &jLambda, int nconfs, int njacks,  int clust_size)
{  
  //L_proj has 5 components: S(0), V(1), P(2), A(3), T(4)
  valarray< valarray< valarray<prop_t> > > L_proj(valarray< valarray<prop_t> >(valarray<prop_t>(prop_t::Zero(),5),mom_list.size()),njacks);
  vvvdcompl_t jG(vvdcompl_t(vdcompl_t(0.0,5),mom_list.size()),njacks);
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
     for(size_t imom=0;imom<mom_list.size();imom++)
       {
	 L_proj[ijack][imom][0]=jLambda[ijack][imom][0]*P[0];
	 
	 for(int igam=1;igam<5;igam++)
	   L_proj[ijack][imom][1]+=jLambda[ijack][imom][igam]*P[igam];
	 
	 L_proj[ijack][imom][2]=jLambda[ijack][imom][5]*P[5];
	 
	 for(int igam=6;igam<10;igam++)  
	   L_proj[ijack][imom][3]+=jLambda[ijack][imom][igam]*P[igam];
	 
	 
	 for(int igam=10;igam<16;igam++)  
	   L_proj[ijack][imom][4]+=jLambda[ijack][imom][igam]*P[igam];
	 
	 for(int j=0;j<5;j++)
	  jG[ijack][imom][j]=L_proj[ijack][imom][j].trace()/12.;
	  
       }
 
 return jG;
 
}

//subtraction of O(a^2) effects
double subtract(vector<double> c, double f, double p2, double p4, double g2_tilde)
{
  double f_new;

  f_new = f - g2_tilde*(p2*(c[0]+c[1]*log(p2))+c[2]*p4/p2)/(12.*M_PI*M_PI);

  return f_new;  
}

//compute fit parameters
vvvd_t compute_fit_parameters(vd_t x, vd_t p4, vvvd_t y, vvd_t sigma, int tag, int njacks, double x_min, double x_max)
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

/***********************************************************/
/*************************** main **************************/
/***********************************************************/

  
int main(int narg,char **arg)
{
  
  if (narg!=11){
    cerr<<"Number of arguments not valid: <mom file> <nconfs> <njacks> <L> <T> <initial conf_id> <step conf_id> <p2fit min> <p2fit max> <action=sym/iwa>"<<endl;
    exit(0);
  }
  
  int nconfs=stoi(arg[2]);
  int njacks=stoi(arg[3]);
  int clust_size=nconfs/njacks;
  int conf_id[nconfs];
  double L=stod(arg[4]),T=stod(arg[5]);
  
  for(int iconf=0;iconf<nconfs;iconf++)
    conf_id[iconf]=stoi(arg[6])+iconf*stoi(arg[7]);
  
  double p2fit_min=stod(arg[8]);
  double p2fit_max=stod(arg[9]);

  double beta=0.0, plaquette=0.0;
  vector<double> c_v(3), c_a(3), c_s(3), c_p(3), c_t(3);
  
  //beta & plaquette
  if(strcmp(arg[10],"iwa")==0)  //Nf=4 (Iwasaki)
    {
      beta=1.9;
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

   //delta m_cr
   double deltam_cr = 0.23;


   ///////////////////////////////////////////////////////////////////////////////////////////////////////


  cout<<"Reading the list of momenta..."<<endl;

  read_mom_list(arg[1]);

  cout<<"Creating Dirac gamma matrices..."<<endl;
  
  //create gamma matrices
  vprop_t GAMMA=make_gamma();

  cout<<"Reading propagators from the files, creating the vertices and preparing the jackknife..."<<endl;

   // put to zero jackknife vertex
  jprop_t jS_0(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks), \
    jS_self_tad(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks), \
    jS_p(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);
  
  jvert_t jVert_0(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks),	\
    jVert_11_self_tad(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks), \
    jVert_p(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks);
  
  jprop_t jS_em(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);
  jvert_t jVert_em(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks);
  
  for(int iconf=0;iconf<nconfs;iconf++)
    {
      int ijack=iconf/clust_size;
      
      //read props
      vprop_t S_0 = read_prop(path_to_conf(conf_id[iconf],"SPECT0")); //QCD
      
      vprop_t S_1ph = read_prop(path_to_conf(conf_id[iconf],"SPECT0_F")); //QCD + 1 photon insertion
      vprop_t S_2ph = read_prop(path_to_conf(conf_id[iconf],"SPECT0_FF")); //QCD + 2 photons insertion
      vprop_t S_t = read_prop(path_to_conf(conf_id[iconf],"SPECT0_T")); //QCD + tadpole insertion
      vprop_t S_p = read_prop(path_to_conf(conf_id[iconf],"SPECT0_P")); //QCD + pseudoscalar insertion
      //vprop_t S_s = read_prop(path_to_conf(conf_id[iconf],"SPECT0_S")); //QCD + scalar insertion
      
      for(auto &p : S_p) p*=complex<double>(0.0,-1.0);
      
      for(size_t imom=0;imom<mom_list.size();imom++)
	{
	  //create vertex functions with the i_mom momentum
	  qline_t Vert_0=make_vertex(S_0, S_0, imom, GAMMA);
	  
	  qline_t Vert_11=make_vertex(S_1ph, S_1ph, imom, GAMMA);
	  
	  qline_t Vert_02=make_vertex(S_0, S_2ph, imom, GAMMA);
	  qline_t Vert_20=make_vertex(S_2ph, S_0, imom, GAMMA);
	  
	  qline_t Vert_0t=make_vertex(S_0, S_t, imom, GAMMA);
	  qline_t Vert_t0=make_vertex(S_t, S_0, imom, GAMMA);
	  
	  qline_t Vert_0p=make_vertex(S_0, S_p, imom, GAMMA);
	  qline_t Vert_p0=make_vertex(S_p, S_0, imom, GAMMA);
	  
	  // qline_t Vert_0s = make_vertex(S_0, S_s, imom, GAMMA);
	  // qline_t Vert_s0 = make_vertex(S_s, S_0, imom, GAMMA);
	  
	  
	  //create pre-jackknife propagator
	  jS_0[ijack][imom] += S_0[imom];
	  jS_self_tad[ijack][imom] += S_2ph[imom] + S_t[imom];
	  jS_p[ijack][imom] += S_p[imom];
	  
	  //create pre-jackknife vertex
	  jVert_0[ijack][imom] += Vert_0;
	  jVert_11_self_tad[ijack][imom] += Vert_11 + Vert_02 + Vert_20 + Vert_0t + Vert_t0;
	  jVert_p[ijack][imom] += Vert_0p + Vert_p0;
	  
	}
    }
  
  cout<<"Applying the jackknife resampling to propagators and vertices..."<<endl;
  
  cout<<"   Jackknife of propagators (1/2)"<<endl;
  //compute fluctuations of the propagator
  jS_0 = jackknife_prop(jS_0,nconfs,clust_size);
  jS_self_tad = jackknife_prop(jS_self_tad,nconfs,clust_size);
  jS_p = jackknife_prop(jS_p,nconfs,clust_size);
  
  cout<<"   Jackknife of vertices (2/2)"<<endl;
  //compute fluctuations of the vertex
  jVert_0 = jackknife_vertex(jVert_0,nconfs,clust_size);
  jVert_11_self_tad = jackknife_vertex(jVert_11_self_tad,nconfs,clust_size);
  jVert_p = jackknife_vertex(jVert_p,nconfs,clust_size);

  //define em propagator and vertex
  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      {
	jS_em[ijack][imom] = jS_self_tad[ijack][imom] - deltam_cr*jS_p[ijack][imom];
	for(int igam=0;igam<16;igam++)
	  jVert_em[ijack][imom][igam] = jVert_11_self_tad[ijack][imom][igam] - deltam_cr*jVert_p[ijack][imom][igam];
      }
    
  

  cout<<"Inverting the propagators..."<<endl;
   
  //inverse of the propagators
  jprop_t jS_0_inv = inverse_jprop(jS_0, njacks);
  jprop_t jS_em_inv = jS_0_inv*jS_em*jS_0_inv;
   
  cout<<"Amputating the external legs..."<<endl;
  
  //amputate external legs
  jvert_t jLambda_0 = amputate(jS_0_inv, jVert_0, jS_0_inv, GAMMA, njacks);
  jvert_t jLambda_em = amputate(jS_0_inv, jVert_em, jS_0_inv, GAMMA, njacks);
  jvert_t jLambda_a = amputate(jS_em_inv, jVert_0, jS_0_inv, GAMMA, njacks);
  jvert_t jLambda_b = amputate(jS_0_inv, jVert_0, jS_em_inv, GAMMA, njacks);

  	  
  cout<<"Computing Zq..."<<endl;
  
  //compute Zq according to RI'-MOM, one for each momentum
  vvdcompl_t jZq = compute_jZq(GAMMA,jS_0_inv,L,T,nconfs,njacks,clust_size);
  vvdcompl_t jZq_em = - compute_jZq(GAMMA,jS_em_inv,L,T,nconfs,njacks,clust_size);
  //compute Zq according to Sigma1-way
  vvdcompl_t jSigma1 = compute_jSigma1(GAMMA,jS_0_inv,L,T,nconfs,njacks,clust_size);
  vvdcompl_t jSigma1_em = - compute_jSigma1(GAMMA,jS_em_inv,L,T,nconfs,njacks,clust_size);

  cout<<"Projecting the Green functions..."<<endl;
  
  //compute the projected green function as a vector (S,V,P,A,T)
  vvvdcompl_t jG_0 = project(GAMMA,jLambda_0,nconfs,njacks,clust_size);
  vvvdcompl_t jG_em = project(GAMMA,jLambda_em,nconfs,njacks,clust_size);
  vvvdcompl_t jG_a = project(GAMMA,jLambda_a,nconfs,njacks,clust_size);
  vvvdcompl_t jG_b = project(GAMMA,jLambda_b,nconfs,njacks,clust_size);
 

  cout<<"Computing the Z's..."<<endl;
  
  //compute Z's according to RI-MOM and to Sigma1-way, one for each momentum
  vvvdcompl_t jZ(vvdcompl_t(vdcompl_t(0.0,5),mom_list.size()),njacks);
  vvvdcompl_t jZ1(vvdcompl_t(vdcompl_t(0.0,5),mom_list.size()),njacks);
  
  vvvdcompl_t jZ_em(vvdcompl_t(vdcompl_t(0.0,5),mom_list.size()),njacks);
  vvvdcompl_t jZ1_em(vvdcompl_t(vdcompl_t(0.0,5),mom_list.size()),njacks);

  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      for(int k=0;k<5;k++)
	{
	  jZ[ijack][imom][k] = jZq[ijack][imom]/jG_0[ijack][imom][k];
	  jZ1[ijack][imom][k] = jSigma1[ijack][imom]/jG_0[ijack][imom][k];

	  jZ_em[ijack][imom][k] = (-jG_em[ijack][imom][k]+jG_a[ijack][imom][k]+jG_b[ijack][imom][k])/jG_0[ijack][imom][k] + jZq_em[ijack][imom]/jZq[ijack][imom];
	  jZ1_em[ijack][imom][k] = (-jG_em[ijack][imom][k]+jG_a[ijack][imom][k]+jG_b[ijack][imom][k])/jG_0[ijack][imom][k] + jSigma1_em[ijack][imom]/jSigma1[ijack][imom];

	}
	  
  //create p_tilde vector  
  vvd_t p(vd_t(0.0,4),mom_list.size());
  vvd_t p_tilde(vd_t(0.0,4),mom_list.size());
  vd_t p2(vd_t(0.0,mom_list.size()));
  vd_t p2_space(vd_t(0.0,mom_list.size()));
  vd_t p4(vd_t(0.0,mom_list.size()));  //for the democratic filter
    
  for(size_t imom=0;imom<mom_list.size();imom++)
	{
	  p[imom]={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
	  p_tilde[imom]={sin(p[imom][0]),sin(p[imom][1]),sin(p[imom][2]),sin(p[imom][3])};

	  for(int coord=0;coord<4;coord++)
	    p2[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
	  for(int coord=0;coord<3;coord++)
	    p2_space[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
	  for(int coord=0;coord<4;coord++)
	    p4[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord]*p_tilde[imom][coord]*p_tilde[imom][coord]; //for the democratic filter
	}
  
  //Create new extended vector
  
  cout<<"Creating the extended vector..."<<endl;
  
  vvvd_t new_list(vvd_t(vd_t(0.0,19),mom_list.size()),njacks);

  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      {
	for(int i=0;i<4;i++)
	  new_list[ijack][imom][i]=mom_list[imom][i];
	new_list[ijack][imom][4]=p2[imom];
	new_list[ijack][imom][6]=jZq[ijack][imom].real();
	for(int i=0;i<5;i++)
	  new_list[ijack][imom][7+i]=jZ[ijack][imom][0+i].real();

	new_list[ijack][imom][12]=p4[imom]; //for the democratic filter

	new_list[ijack][imom][13]=jSigma1[ijack][imom].real();  //PROVA: sigma1
	for(int i=0;i<5;i++)
	  new_list[ijack][imom][14+i]=jZ1[ijack][imom][0+i].real(); //PROVA: sigma1
	
	
      }

  
 
  //Assign the tag for fixed ijack
  int tag=0;
  double eps=1.0e-15;  //Precision: is it correct?
  for(size_t imom=0;imom<mom_list.size();imom++)
    {
      size_t count=0;
      for(size_t i=0;i<imom;i++)
	{
	  if((abs(new_list[0][i][4]-new_list[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_list[i][0]-new_list[imom][0])<eps*/ && \
	      abs(abs(new_list[0][i][1])*abs(new_list[0][i][2])*abs(new_list[0][i][3])-(abs(new_list[0][imom][1])*abs(new_list[0][imom][2])*abs(new_list[0][imom][3])))<eps ) || \
	     (abs(new_list[0][i][4]-new_list[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_list[i][0]+new_list[imom][0]+1.)<eps*/ && \
	      abs(abs(new_list[0][i][1])*abs(new_list[0][i][2])*abs(new_list[0][i][3])-(abs(new_list[0][imom][1])*abs(new_list[0][imom][2])*abs(new_list[0][imom][3])))<eps  )  )
	    {
	      new_list[0][imom][5]=new_list[0][i][5];
	    }else{
	    count++;
	  }
	  
	  if(count==imom)
	    {
	      tag++;
	      new_list[0][imom][5]=tag;
	    }
	}
    }
  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int ijack=1;ijack<njacks;ijack++)
      new_list[ijack][imom][5]=new_list[0][imom][5];

  cout<<"Number of equivalent momenta: "<<tag+1<<endl;

  

  //Subtraction of discretization effects O(a^2)
  vvvd_t new_list_corr(vvd_t(vd_t(0.0,13),mom_list.size()),njacks);
  vvvd_t jG_new(vvd_t(vd_t(0.0,5),mom_list.size()),njacks);
  vector< vector<double> > c_q (mom_list.size(),vector<double>(3));
  
  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      {
	for(int i=0;i<6;i++)
	  new_list_corr[ijack][imom][i]=new_list[ijack][imom][i];
	new_list_corr[ijack][imom][12]=new_list[ijack][imom][12];


	
	if(strcmp(arg[10],"sym")==0) c_q[imom]={1.14716212+2.07733285/(double)Np[imom],-73./360.-157./180./(double)Np[imom],7./240.};   //Symanzik action
	if(strcmp(arg[10],"iwa")==0) c_q[imom]={0.6202244+1.8490436/(double)Np[imom],-0.0748167-0.963033/(double)Np[imom],0.0044};      //Iwasaki action
	
	/*
	  new_list_corr[ijack][imom][6]=subtract(c_q,new_list[ijack][imom][6],new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);   //Zq (RI'-MOM)
	  
	  jG_new[ijack][imom][0]=subtract(c_s,jG[ijack][imom][0].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_s
	  jG_new[ijack][imom][1]=subtract(c_a,jG[ijack][imom][1].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_a
	  jG_new[ijack][imom][2]=subtract(c_p,jG[ijack][imom][2].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_p
	  jG_new[ijack][imom][3]=subtract(c_v,jG[ijack][imom][3].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_v
	  jG_new[ijack][imom][4]=subtract(c_t,jG[ijack][imom][4].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_t
	  
	  new_list_corr[ijack][imom][7]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][0]; //Zs
	  new_list_corr[ijack][imom][8]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][1]; //Za
	  new_list_corr[ijack][imom][9]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][2]; //Zp
	  new_list_corr[ijack][imom][10]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][3]; //Zv
	  new_list_corr[ijack][imom][11]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][4]; //Zt
	*/
	
	new_list_corr[ijack][imom][6]=subtract(c_q[imom],new_list[ijack][imom][13],new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde); //Zq    (Sigma1)
	
	jG_new[ijack][imom][0]=subtract(c_s,jG_0[ijack][imom][0].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_s
	jG_new[ijack][imom][1]=subtract(c_a,jG_0[ijack][imom][1].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_a
	jG_new[ijack][imom][2]=subtract(c_p,jG_0[ijack][imom][2].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_p
	jG_new[ijack][imom][3]=subtract(c_v,jG_0[ijack][imom][3].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_v
	jG_new[ijack][imom][4]=subtract(c_t,jG_0[ijack][imom][4].real(),new_list[ijack][imom][4],new_list[ijack][imom][12],g2_tilde);//G_t
	
	new_list_corr[ijack][imom][7]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][0]; //Zs
	new_list_corr[ijack][imom][8]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][1]; //Za
	new_list_corr[ijack][imom][9]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][2]; //Zp
	new_list_corr[ijack][imom][10]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][3]; //Zv
	new_list_corr[ijack][imom][11]=new_list_corr[ijack][imom][6]/jG_new[ijack][imom][4]; //Zt	
	
	
      }
  
  //Average of Z's corresponding to equivalent momenta (same tag) and print on file
  vd_t p2_eq(vd_t(0.0,tag+1));
  vd_t p4_eq(vd_t(0.0,tag+1)); //for the democratic filter
  
  valarray< valarray< valarray< vector<double> > > > jZ_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks); //Zq RI'-MOM
  vvvd_t jZ_average(vvd_t(vd_t(0.0,6),tag+1),njacks);
  
  valarray< valarray< valarray< vector<double> > > > jZ1_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks);  //PROVA: sigma1
  vvvd_t jZ1_average(vvd_t(vd_t(0.0,6),tag+1),njacks);  //PROVA
  
  cout<<"Averaging the Z's corresponding to equivalent momenta and printing on the output file..."<<endl;

  for(int ijack=0;ijack<njacks;ijack++)
    for(int t=0;t<=tag;t++)
      {
	int count_equivalent=0;
	for(size_t imom=0;imom<mom_list.size();imom++)
	  { 
	    if(t==new_list[ijack][imom][5])
	      {
		count_equivalent++;
		p2_eq[t]=new_list[0][imom][4];
		p4_eq[t]=new_list[0][imom][12];
		
		jZ_same_tag[ijack][t][0].push_back(new_list[ijack][imom][6]);
		jZ_same_tag[ijack][t][1].push_back(new_list[ijack][imom][7]);
		jZ_same_tag[ijack][t][2].push_back(new_list[ijack][imom][8]);
		jZ_same_tag[ijack][t][3].push_back(new_list[ijack][imom][9]);
		jZ_same_tag[ijack][t][4].push_back(new_list[ijack][imom][10]);
		jZ_same_tag[ijack][t][5].push_back(new_list[ijack][imom][11]);

		jZ1_same_tag[ijack][t][0].push_back(new_list[ijack][imom][13]);  //PROVA: sigma1
		jZ1_same_tag[ijack][t][1].push_back(new_list[ijack][imom][14]);
		jZ1_same_tag[ijack][t][2].push_back(new_list[ijack][imom][15]);
		jZ1_same_tag[ijack][t][3].push_back(new_list[ijack][imom][16]);
		jZ1_same_tag[ijack][t][4].push_back(new_list[ijack][imom][17]);
		jZ1_same_tag[ijack][t][5].push_back(new_list[ijack][imom][18]);			
	      }
	  }
	
	for(int i=0;i<count_equivalent;i++)  //average over the equivalent Z's in each jackknife
	  { 
	    jZ_average[ijack][t][0]+=jZ_same_tag[ijack][t][0][i]/count_equivalent;
	    jZ_average[ijack][t][1]+=jZ_same_tag[ijack][t][1][i]/count_equivalent;
	    jZ_average[ijack][t][2]+=jZ_same_tag[ijack][t][2][i]/count_equivalent;
	    jZ_average[ijack][t][3]+=jZ_same_tag[ijack][t][3][i]/count_equivalent;
	    jZ_average[ijack][t][4]+=jZ_same_tag[ijack][t][4][i]/count_equivalent;
	    jZ_average[ijack][t][5]+=jZ_same_tag[ijack][t][5][i]/count_equivalent;


	    jZ1_average[ijack][t][0]+=jZ1_same_tag[ijack][t][0][i]/count_equivalent;  //PROVA: sigma1
	    jZ1_average[ijack][t][1]+=jZ1_same_tag[ijack][t][1][i]/count_equivalent;
	    jZ1_average[ijack][t][2]+=jZ1_same_tag[ijack][t][2][i]/count_equivalent;
	    jZ1_average[ijack][t][3]+=jZ1_same_tag[ijack][t][3][i]/count_equivalent;
	    jZ1_average[ijack][t][4]+=jZ1_same_tag[ijack][t][4][i]/count_equivalent;
	    jZ1_average[ijack][t][5]+=jZ1_same_tag[ijack][t][5][i]/count_equivalent;	    
	  }
      }

  vvd_t Z_mean_value(vd_t(0.0,6),tag+1);
  vvd_t Z2_mean_value(vd_t(0.0,6),tag+1);
  vvd_t Z_error(vd_t(0.0,6),tag+1);

  vvd_t Z1_mean_value(vd_t(0.0,6),tag+1);  //PROVA: sigma1
  vvd_t Z21_mean_value(vd_t(0.0,6),tag+1);
  vvd_t Z1_error(vd_t(0.0,6),tag+1);
  
  for(int t=0;t<=tag;t++)
    for(int i=0;i<6;i++)
      {
	for(int ijack=0;ijack<njacks;ijack++)
	  {
	    Z_mean_value[t][i]+=jZ_average[ijack][t][i]/njacks;
	    Z2_mean_value[t][i]+=jZ_average[ijack][t][i]*jZ_average[ijack][t][i]/njacks;
	  }
	Z_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z2_mean_value[t][i]-Z_mean_value[t][i]*Z_mean_value[t][i]);
      }

  for(int t=0;t<=tag;t++)  //PROVA: sigma1
    for(int i=0;i<6;i++)
      {
	for(int ijack=0;ijack<njacks;ijack++)
	  {
	    Z1_mean_value[t][i]+=jZ1_average[ijack][t][i]/njacks;
	    Z21_mean_value[t][i]+=jZ1_average[ijack][t][i]*jZ1_average[ijack][t][i]/njacks;
	  }
	Z1_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z21_mean_value[t][i]-Z1_mean_value[t][i]*Z1_mean_value[t][i]);
      }

  //print on file
  
  print_file("Z_average.txt",p2_eq,Z_mean_value,Z_error,tag);
  print_file_filtered("Z_average_filtered.txt",p2_eq,p4_eq,Z_mean_value,Z_error,tag);
  print_file("Z_average_sigma1.txt",p2_eq,Z1_mean_value,Z1_error,tag);
  print_file_filtered("Z_average_filtered_sigma1.txt",p2_eq,p4_eq,Z1_mean_value,Z1_error,tag);

  
  

 cout<<"Averaging the corrected Z's corresponding to equivalent momenta and printing on the output file..."<<endl;

  valarray< valarray< valarray< vector<double> > > > jZ_corr_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag+1),njacks);
  vvvd_t jZ_corr_average(vvd_t(vd_t(0.0,6),tag+1),njacks);

  for(int ijack=0;ijack<njacks;ijack++)
    for(int t=0;t<=tag;t++)
      {
	int count_equivalent=0;
	for(size_t imom=0;imom<mom_list.size();imom++)
	  { 
	    if(t==new_list[ijack][imom][5])
	      {
		count_equivalent++;
		p2_eq[t]=new_list[0][imom][4];
		p4_eq[t]=new_list[0][imom][12];
		
		jZ_corr_same_tag[ijack][t][0].push_back(new_list_corr[ijack][imom][6]);
		jZ_corr_same_tag[ijack][t][1].push_back(new_list_corr[ijack][imom][7]);
		jZ_corr_same_tag[ijack][t][2].push_back(new_list_corr[ijack][imom][8]);
		jZ_corr_same_tag[ijack][t][3].push_back(new_list_corr[ijack][imom][9]);
		jZ_corr_same_tag[ijack][t][4].push_back(new_list_corr[ijack][imom][10]);
		jZ_corr_same_tag[ijack][t][5].push_back(new_list_corr[ijack][imom][11]);	
	      }
	  }
	
	for(int i=0;i<count_equivalent;i++)  //average over the equivalent Z's in each jackknife
	  { 
	    jZ_corr_average[ijack][t][0]+=jZ_corr_same_tag[ijack][t][0][i]/count_equivalent;
	    jZ_corr_average[ijack][t][1]+=jZ_corr_same_tag[ijack][t][1][i]/count_equivalent;
	    jZ_corr_average[ijack][t][2]+=jZ_corr_same_tag[ijack][t][2][i]/count_equivalent;
	    jZ_corr_average[ijack][t][3]+=jZ_corr_same_tag[ijack][t][3][i]/count_equivalent;
	    jZ_corr_average[ijack][t][4]+=jZ_corr_same_tag[ijack][t][4][i]/count_equivalent;
	    jZ_corr_average[ijack][t][5]+=jZ_corr_same_tag[ijack][t][5][i]/count_equivalent;
	  }
      }

  vvd_t Z_corr_mean_value(vd_t(0.0,6),tag+1);
  vvd_t Z2_corr_mean_value(vd_t(0.0,6),tag+1);
  vvd_t Z_corr_error(vd_t(0.0,6),tag+1);
  
  for(int t=0;t<=tag;t++)
    for(int i=0;i<6;i++)
      {
	for(int ijack=0;ijack<njacks;ijack++)
	  {
	    Z_corr_mean_value[t][i]+=jZ_corr_average[ijack][t][i]/njacks;
	    Z2_corr_mean_value[t][i]+=jZ_corr_average[ijack][t][i]*jZ_corr_average[ijack][t][i]/njacks;
	  }
	Z_corr_error[t][i]=sqrt((double)(njacks-1))*sqrt(Z2_corr_mean_value[t][i]-Z_corr_mean_value[t][i]*Z_corr_mean_value[t][i]);
      }

  //print on file
  print_file("Z_average_corrected.txt",p2_eq,Z_corr_mean_value,Z_corr_error,tag);
  print_file_filtered("Z_average_corrected_filtered.txt",p2_eq,p4_eq,Z_corr_mean_value,Z_corr_error,tag);
  

  //Fit parameters before and after the correction
  
    
  vvvd_t jfit_parameters=compute_fit_parameters(p2_eq,p4_eq,jZ_average,Z_error,tag,njacks,p2fit_min,p2fit_max);
  vvvd_t jfit_parameters_corr=compute_fit_parameters(p2_eq,p4_eq,jZ_corr_average,Z_corr_error,tag,njacks,p2fit_min,p2fit_max);

  vd_t A(0.0,6), B(0.0,6), A_error(0.0,6), B_error(0.0,6), A2(0.0,6), B2(0.0,6);
  vd_t A_corr(0.0,6), B_corr(0.0,6), A_corr_error(0.0,6), B_corr_error(0.0,6), A2_corr(0.0,6), B2_corr(0.0,6);
  
  for(int iZ=0;iZ<6;iZ++)
    {
      for(int ijack=0;ijack<njacks;ijack++)
	{
	  A[iZ]+=jfit_parameters[ijack][iZ][0]/njacks;
	  A2[iZ]+=jfit_parameters[ijack][iZ][0]*jfit_parameters[ijack][iZ][0]/njacks;
	  B[iZ]+=jfit_parameters[ijack][iZ][1]/njacks;
	  B2[iZ]+=jfit_parameters[ijack][iZ][1]*jfit_parameters[ijack][iZ][1]/njacks;
	  
	  
	  A_corr[iZ]+=jfit_parameters_corr[ijack][iZ][0]/njacks;
	  A2_corr[iZ]+=jfit_parameters_corr[ijack][iZ][0]*jfit_parameters_corr[ijack][iZ][0]/njacks;
	  B_corr[iZ]+=jfit_parameters_corr[ijack][iZ][1]/njacks;
	  B2_corr[iZ]+=jfit_parameters_corr[ijack][iZ][1]*jfit_parameters_corr[ijack][iZ][1]/njacks;
	}
      A_error[iZ]=sqrt((double)(njacks-1))*sqrt(A2[iZ]-A[iZ]*A[iZ]);
      B_error[iZ]=sqrt((double)(njacks-1))*sqrt(B2[iZ]-B[iZ]*B[iZ]);

      A_corr_error[iZ]=sqrt((double)(njacks-1))*sqrt(A2_corr[iZ]-A_corr[iZ]*A_corr[iZ]);
      B_corr_error[iZ]=sqrt((double)(njacks-1))*sqrt(B2_corr[iZ]-B_corr[iZ]*B_corr[iZ]);
      
    }
  
  cout<<" "<<endl;
  cout<<"Fit parameters BEFORE the correction for the filtered data: y=a*x+b"<<endl;
  cout<<"-------------------------------------------------------------------"<<endl;
  cout<<"     ZQ     "<<endl;
  cout<<"a = "<<A[0]<<" +/- "<<A_error[0]<<endl;
  cout<<"b = "<<B[0]<<" +/- "<<B_error[0]<<endl;
  cout<<"     ZS     "<<endl;
  cout<<"a = "<<A[1]<<" +/- "<<A_error[1]<<endl;
  cout<<"b = "<<B[1]<<" +/- "<<B_error[1]<<endl;
  cout<<"     ZA     "<<endl;
  cout<<"a = "<<A[2]<<" +/- "<<A_error[2]<<endl;
  cout<<"b = "<<B[2]<<" +/- "<<B_error[2]<<endl;
  cout<<"     ZP     "<<endl;
  cout<<"a = "<<A[3]<<" +/- "<<A_error[3]<<endl;
  cout<<"b = "<<B[3]<<" +/- "<<B_error[3]<<endl;
  cout<<"     ZV     "<<endl;
  cout<<"a = "<<A[4]<<" +/- "<<A_error[4]<<endl;
  cout<<"b = "<<B[4]<<" +/- "<<B_error[4]<<endl;
  cout<<"     ZT     "<<endl;
  cout<<"a = "<<A[5]<<" +/- "<<A_error[5]<<endl;
  cout<<"b = "<<B[5]<<" +/- "<<B_error[5]<<endl;
    
  cout<<" "<<endl;
  cout<<"Fit parameters AFTER the correction for the filtered data: y=a*x+b"<<endl;
  cout<<"-------------------------------------------------------------------"<<endl;
  cout<<"     ZQ     "<<endl;
  cout<<"a = "<<A_corr[0]<<" +/- "<<A_corr_error[0]<<endl;
  cout<<"b = "<<B_corr[0]<<" +/- "<<B_corr_error[0]<<endl;
  cout<<"     ZS     "<<endl;
  cout<<"a = "<<A_corr[1]<<" +/- "<<A_corr_error[1]<<endl;
  cout<<"b = "<<B_corr[1]<<" +/- "<<B_corr_error[1]<<endl;
  cout<<"     ZA     "<<endl;
  cout<<"a = "<<A_corr[2]<<" +/- "<<A_corr_error[2]<<endl;
  cout<<"b = "<<B_corr[2]<<" +/- "<<B_corr_error[2]<<endl;
  cout<<"     ZP     "<<endl;
  cout<<"a = "<<A_corr[3]<<" +/- "<<A_corr_error[3]<<endl;
  cout<<"b = "<<B_corr[3]<<" +/- "<<B_corr_error[3]<<endl;
  cout<<"     ZV     "<<endl;
  cout<<"a = "<<A_corr[4]<<" +/- "<<A_corr_error[4]<<endl;
  cout<<"b = "<<B_corr[4]<<" +/- "<<B_corr_error[4]<<endl;
  cout<<"     ZT     "<<endl;
  cout<<"a = "<<A_corr[5]<<" +/- "<<A_corr_error[5]<<endl;
  cout<<"b = "<<B_corr[5]<<" +/- "<<B_corr_error[5]<<endl;
    
  
  cout<<"End of the program."<<endl;
  
  return 0;
}


