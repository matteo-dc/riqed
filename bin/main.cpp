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

//list of momenta
vector<coords_t> mom_list;

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
  
  //sigma23-31-12
  for(int i=0;i<3;i++)
    {
      gam[10+i]=0.5*(gam[(2+i)%3]*gam[(3+i)%3]-gam[(3+i)%3]*gam[(2+i)%3]);
    }
  //sigma01-02-03
  for(int i=0;i<3;i++)
    {
      gam[13+i]=0.5*(gam[0]*gam[1+i]-gam[1+i]*gam[0]);
    }
 
  return gam;
}

//calculate the vertex function in a given configuration for the given equal momenta
vprop_t make_vertex(const vprop_t &prop, size_t mom,const vprop_t &gamma)
{
  vprop_t vert(16);
  for(int mu=0;mu<16;mu++)
    {      
      vert[mu]=prop[mom]*gamma[mu]*gamma[5]*prop[mom].adjoint()*gamma[5];  /*it has to be "jackknifed"*/
    }
  return vert;
}

//create the path-string to the configuration

//OLD CONFIGURATIONS
/*
string path_to_conf(int i_conf,const char *name)
{
  char path[1024];
  sprintf(path,"out%d/fft_%s",i_conf,name);
  return path;
}
*/

//NEW CONFIGURATIONS
///*
string path_to_conf(int i_conf,const char *name)
{
  char path[1024];
  sprintf(path,"out/%04d/fft_%s",i_conf,name);
  return path;
}
//*/


//jackknife Propagator
valarray<valarray<prop_t>> jackknife_prop(  valarray<valarray<prop_t>> &jS, int nconf, int clust_size )
{
  valarray<prop_t> jSum(valarray<prop_t>(prop_t::Zero(),mom_list.size()));

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
valarray<valarray<qline_t>> jackknife_vertex( valarray<valarray<qline_t>> &jVert, int nconf, int clust_size )
{
  valarray<qline_t> jSum(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()));
  
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

//jackknife Zq
valarray<valarray<dcompl>> jackknife_Zq( valarray<valarray<dcompl>> &jZq, int nconf, int clust_size )
{
  valarray<dcompl> jSum(valarray<dcompl>(0.0,mom_list.size()));
    
  for(size_t j=0;j<jZq.size();j++) jSum+= jZq[j];

  for(size_t j=0;j<jZq.size();j++)
    {
      jZq[j]=jSum-jZq[j];
      for(auto &it : jZq[j])
	it/=nconf-clust_size;
    }

  return jZq;
  
}

//jackknife G
valarray<valarray<valarray<complex<double>>>> jackknife_G(valarray<valarray<valarray<complex<double>>>> &jG, int nconf, int clust_size)
{
  valarray<valarray<complex<double>>> jSum(valarray<complex<double>>(0.0,5),mom_list.size());

  for(size_t j=0;j<jG.size();j++) jSum+=jG[j];

  for(size_t j=0;j<jG.size();j++)
    {
      jG[j]=jSum-jG[j];
       for(auto &it : jG[j])
	 for(auto &jt : it)
	   jt/=nconf-clust_size;
    }

  return jG;
  
}

//jackknife Z
valarray<valarray<valarray<dcompl>>> jackknife_Z(valarray<valarray<valarray<dcompl>>> &jZ, int nconf, int clust_size)
{
  valarray<valarray<complex<double>>> jSum(valarray<complex<double>>(0.0,5),mom_list.size());

  for(size_t j=0;j<jZ.size();j++) jSum+= jZ[j];

  for(size_t j=0;j<jZ.size();j++)
    {
      jZ[j]=jSum-jZ[j];
       for(auto &it : jZ[j])
	 for(auto &jt : it)
	   jt/=nconf-clust_size;
    }

  return jZ;
}



//average Propagator
valarray<prop_t>  average_prop( const valarray<valarray<prop_t>> &jS )
{
  int njacks = jS.size();
  
  valarray<prop_t> jAverage(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  
  //sum of jS
  for(int j=0;j<njacks;j++) jAverage+= jS[j];
  //divide for njacks each component of jAverage (one for each moment)
  for(auto &it : jAverage)
    it/=njacks;
  
  return jAverage;
  
}

//average Vertex
valarray<qline_t> average_vertex( const valarray<valarray<qline_t>> &jVert)
{
  int njacks = jVert.size();
  
  valarray<qline_t> jAverage(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()));
  
  //sum of jVert
  for(int j=0;j<njacks;j++) jAverage+= jVert[j];
  //divide for njacks each component of jAverage (one for each moment)
  for(auto &it : jAverage)
    for(auto &jt : it)
      jt/=njacks;
  
  return jAverage;
}


//error Propagator
valarray<prop_t>  error_prop(const valarray<valarray<prop_t>> &jS ,valarray<prop_t> meanS )
{
  int njacks = jS.size();
  valarray<prop_t> jsq_mean(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  valarray<prop_t> err;
  
  //sum of (jS)^2
  for(int j=0;j<njacks;j++) jsq_mean+= jS[j]*jS[j];
  //divide for njacks each component of jsq_mean (one for each momentum)
  for(auto &it : jsq_mean)
    it/=njacks;
  
  err=jsq_mean-meanS*meanS;
  for(auto &it : err)
    it=it*(njacks-1);
  for(auto &ir : err)
    for(int i=0;i<12;i++)
      for(int j=0;i<12;i++)
	ir(i,j)=sqrt(ir(i,j));
  
  return err;
}

//error Vertex
valarray<qline_t> error_vertex(const valarray<valarray<qline_t>> &jVert, valarray<qline_t> meanVert )
{
  int njacks = jVert.size();
  valarray<qline_t> jsq_mean(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()));
  valarray<qline_t> err;

  //sum of (jVert)^2
  for(int j=0;j<njacks;j++) jsq_mean+= jVert[j]*jVert[j];
  //divide for njacks each component of jsq_mean (one for each momentum)
  for(auto &it : jsq_mean)
    for(auto &jt : it)
      jt/=njacks;
 
  err=jsq_mean-meanVert*meanVert;
  for(auto &it : err)
    for(auto &jt : it)
      jt=jt*(njacks-1);

  for(auto &ir : err)
    for(auto &jr : ir)
      for(int i=0;i<12;i++)
	for(int j=0;i<12;i++)
	  jr(i,j)=sqrt(jr(i,j));

  return err;
}

//compute jZq
valarray<valarray<dcompl>> compute_jZq(vprop_t GAMMA, valarray<valarray<prop_t>> jS_inv, double L, double T, int nconfs, int njacks, int cluster_size)
{
  //double Lx=24,Ly=24,Lz=24,Lt=48;
  double V=L*L*L*T;
  
  //compute p_slash as a vector of prop-type matrices
  valarray<valarray<double>> p(valarray<double>(0.0,4),mom_list.size());
  valarray<valarray<double>> p_tilde(valarray<double>(0.0,4),mom_list.size());
  valarray<prop_t> p_slash(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  valarray<double> p2(valarray<double>(0.0,mom_list.size()));
  valarray<valarray<dcompl>> jZq(valarray<dcompl>(mom_list.size()),njacks);
  complex<double> I(0,1);

  

  for(size_t imom=0;imom<mom_list.size();imom++)
    {
      p[imom]={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
      p_tilde[imom]={sin(p[imom][0]),sin(p[imom][1]),sin(p[imom][2]),sin(p[imom][3])};
      
      for(int igam=1;igam<5;igam++)
	//	p_slash[imom]+=GAMMA[igam]*p[imom][igam-1];
	p_slash[imom]+=GAMMA[igam]*p_tilde[imom][igam-1];
  
  /*  Note that: p_slash*p_slash=p2*GAMMA[0]  */
  
  //compute p^2
      for(int coord=0;coord<4;coord++)
	//	p2[imom]+=p[imom][coord]*p[imom][coord];
	p2[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
      
    }
      for(int ijack=0;ijack<njacks;ijack++)
	for(size_t imom=0;imom<mom_list.size();imom++)
	  { 
	    //compute jZq = Quark field RC (RI'-MOM), one for each momentum and jackknife 
	    jZq[ijack][imom]+=-I*((p_slash[imom]*jS_inv[ijack][imom]).trace())/p2[imom]/12./V;
	  }
  
  return jZq;
  
}

//project the amputated green function
valarray<valarray<valarray<complex<double>>>> project_jLambda(vprop_t GAMMA, const valarray<valarray<qline_t>> &jLambda, int nconfs, int njacks,  int clust_size)
{

  // valarray<valarray<valarray<complex<double>>>> jG=project_jLambda(GAMMA,jLambda,nconfs,njacks,clust_size);
  
  //L_proj has 5 components: S(0), V(1), P(2), A(3), T(4)
  valarray<valarray<valarray<prop_t>>> L_proj(valarray<valarray<prop_t>>(valarray<prop_t>(prop_t::Zero(),5),mom_list.size()),njacks);
  valarray<valarray<valarray<complex<double>>>> jG(valarray<valarray<complex<double>>>(valarray<dcompl>(0.0,5),mom_list.size()),njacks);
  vprop_t P(prop_t::Zero(),16);
  
  //create projectors such that tr(GAMMA*P)=Identity
  P[0]=GAMMA[0]; //scalar
  for(int igam=1;igam<5;igam++)  //vector
    P[igam]=GAMMA[igam].adjoint()/4.; 
  P[5]=GAMMA[5];  //pseudoscalar
  for(int igam=6;igam<10;igam++)  //axial
    P[igam]=GAMMA[igam].adjoint()/4.;
  for(int igam=10;igam<16;igam++)  //tensor
    P[igam]=GAMMA[igam].adjoint();
  
 for(int ijack=0;ijack<njacks;ijack++)
   {
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
   }
 
 return jG;
 
}

prop_t propagator_test(vprop_t GAMMA, coords_t n)
{
  double mu=0.4;
  double k=0.125;
  double r=-1.;
  double V=4.*4.*4.*8.;
  double p[4]={2*M_PI*n[1]/4.,2*M_PI*n[2]/4.,2*M_PI*n[3]/4.,2*M_PI*(n[0]+0.5)/8.};
  
  double p_tilde[4]={sin(p[0]),sin(p[1]),sin(p[2]),sin(p[3])};
  double p_hat[4]={2*sin(p[0]/2),2*sin(p[1]/2),2*sin(p[2]/2),2*sin(p[3]/2)};
  complex<double> I(0,1);

  double p_tilde_2=0.;
  double p_hat_2=0.;

  double M=1/(2.*k)-4.;

  prop_t p_tilde_slash(prop_t::Zero());

  for(int ip=0;ip<4;ip++)
    {
      p_tilde_slash+=GAMMA[ip+1]*p_tilde[ip];
      p_tilde_2+=p_tilde[ip]*p_tilde[ip];
      p_hat_2+=p_hat[ip]*p_hat[ip];
    }

  double Mp=M+0.5*p_hat_2;

  prop_t S=(1/V)*(-I*p_tilde_slash + mu*GAMMA[0] + I*Mp*GAMMA[5]*r)/(Mp*Mp + mu*mu + p_tilde_2);

  return S;
}

//return the extended vector mom_list+p2
valarray<double> create_extended_vector(const valarray<double> &p2, size_t imom, const valarray<dcompl> &Zq, const valarray<valarray<dcompl>> &Z)
{
  valarray<double> new_array(0.0,12);
  
  for(int i=0;i<4;i++)
    new_array[i]=mom_list[imom][i];
  new_array[4]=p2[imom];
  new_array[6]=Zq[imom].real();
  new_array[7]=Z[imom][0].real();
  new_array[8]=Z[imom][1].real();
  new_array[9]=Z[imom][2].real();
  new_array[10]=Z[imom][3].real();
  new_array[11]=Z[imom][4].real();
  
  return new_array;
}


  
int main(int narg,char **arg)
{
  
  if (narg!=6){
    cout<<"Number of arguments not valid: <mom file> <nconfs> <njacks> <L> <T>"<<endl;
    exit(0);
  }
  
  //int nconfs=2; //OLD CONFS
  int nconfs=stoi(arg[2]); //NEW CONFS
  int njacks=stoi(arg[3]);
  int clust_size=nconfs/njacks;
  int conf_id[nconfs];
  double L=stod(arg[4]),T=stod(arg[5]);
  
  for(int iconf=0;iconf<nconfs;iconf++)
    conf_id[iconf]=700+iconf*10;
  

  cout<<"Reading the list of momenta..."<<endl; /******/


  read_mom_list(arg[1]);  //NEW CONFS FREE!!
  // read_mom_list("mom_list.txt");  //NEW CONFS!!
  // read_mom_list("mom_list_OLD.txt"); //OLD CONFS!!

  cout<<"Creating Dirac gamma matrices..."<<endl; /******/
  
  //create gamma matrices
  vprop_t GAMMA=make_gamma();
  
  // put to zero jackknife vertex
  valarray<valarray<prop_t>> jS(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);
  valarray<valarray<qline_t>> jVert(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks);

  cout<<"Reading propagators from the files, creating the vertices and preparing the jackknife..."<<endl; /*****/
  
  for(int iconf=0;iconf<nconfs;iconf++)
    {
      int ijack=iconf/clust_size;
      
      //create a propagator in a given configuration
      
      //vprop_t S=read_prop(path_to_conf(iconf+1,"SPECT0")); //OLD CONFS!!!
      vprop_t S=read_prop(path_to_conf(conf_id[iconf],"SPECT0"));  //NEW CONFS!!!

      
      for(size_t imom=0;imom<mom_list.size();imom++)
	{
	  //create vertex functions with the i_mom momentum
	  qline_t Vert=make_vertex(S,imom,GAMMA);
	  
	  //create pre-jackknife propagator
	  jS[ijack][imom]+=S[imom];
	  //create pre-jackknife vertex
	  jVert[ijack][imom]+=Vert;
	  
	}
    }

  cout<<"Applying the jackknife resampling to propagators and vertices..."<<endl;
  
  valarray<valarray<prop_t>> jS_inv(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);
  valarray<valarray<qline_t>>jLambda(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks);
  
  cout<<"   Jackknife of propagators (1/6)"<<endl;
  //compute fluctuations of the propagator
  jS=jackknife_prop(jS,nconfs,clust_size);
  cout<<"   Jackknife of vertices (2/6)"<<endl;
  //compute fluctuations of the vertex
  jVert=jackknife_vertex(jVert,nconfs,clust_size);
  /* 
  cout<<"   Average of propagators (3/6)"<<endl; 
  //compute the average of the propagator
  valarray<prop_t> S_mean=average_prop(jS);
  cout<<"   Average of vertices (4/6)"<<endl;
  //compute the average of the vertex
  valarray<qline_t> Vertex_mean=average_vertex(jVert);
      
  cout<<"   Error on propagators (5/6)"<<endl;
  //compute the error on the propagator
  valarray<prop_t> S_error=error_prop(jS,S_mean);
  cout<<"   Error on vertices (6/6)"<<endl;
  //compute the error on the vertex
  valarray<qline_t> Vertex_error=error_vertex(jVert,Vertex_mean);
  */
  cout<<"Inverting the propagators..."<<endl;
  
  //inverse of the propagator

   for(int ijack=0;ijack<njacks;ijack++)
    {
      for(size_t imom=0;imom<mom_list.size();imom++)
	jS_inv[ijack][imom]+=jS[ijack][imom].inverse();
    }

   jS_inv=jackknife_prop(jS_inv,nconfs,clust_size);
  
  
  cout<<"Amputating the external legs..."<<endl;
  
  //amputate external legs
  for(int ijack=0;ijack<njacks;ijack++)
    { 
      for(size_t imom=0;imom<mom_list.size();imom++)
	for(int igam=0;igam<16;igam++)
	  {
	    jLambda[ijack][imom][igam]+=jS_inv[ijack][imom]*jVert[ijack][imom][igam]*GAMMA[5]*jS_inv[ijack][imom].adjoint()*GAMMA[5];
	  }
    }

  jLambda=jackknife_vertex(jLambda,nconfs,clust_size);
  
  cout<<"Computing Zq..."<<endl;
  
  //compute Zq according to RI'-MOM, one for each momentum
  valarray<valarray<dcompl>> jZq=compute_jZq(GAMMA,jS_inv,L,T,nconfs,njacks,clust_size);

  jZq=jackknife_Zq(jZq,nconfs,clust_size);
  

  cout<<"Projecting the Green functions..."<<endl;
  
  //compute the projected green function as a vector (S,V,P,A,T)
  valarray<valarray<valarray<complex<double>>>> jG=project_jLambda(GAMMA,jLambda,nconfs,njacks,clust_size);

  cout<<"---------"<<endl;
  
  jG=jackknife_G(jG,nconfs,clust_size);

  cout<<"Computing the Z's..."<<endl;
  
  //compute Z's according to RI-MOM, one for each momentum
  valarray<valarray<valarray<dcompl>>> jZ(valarray<valarray<dcompl>>(valarray<dcompl>(0.0,5),mom_list.size()),njacks);

  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      for(int k=0;k<5;k++)
	jZ[ijack][imom][k]+=jZq[ijack][imom]/jG[ijack][imom][k];
  
  jZ=jackknife_Z(jZ,nconfs,clust_size);

  //create p_tilde vector  
  valarray<valarray<double>> p(valarray<double>(0.0,4),mom_list.size());
  valarray<valarray<double>> p_tilde(valarray<double>(0.0,4),mom_list.size());
  valarray<double> p2(valarray<double>(0.0,mom_list.size()));
  valarray<double> p2_space(valarray<double>(0.0,mom_list.size()));

  for(size_t imom=0;imom<mom_list.size();imom++)
	{
	  p[imom]={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
	  p_tilde[imom]={sin(p[imom][0]),sin(p[imom][1]),sin(p[imom][2]),sin(p[imom][3])};

	  for(int coord=0;coord<4;coord++)
	    p2[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
	  for(int coord=0;coord<3;coord++)
	    p2_space[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
	}
  
  //Create new extended vector
  
  cout<<"Creating Z average vector..."<<endl;
  
  valarray<valarray<valarray<double>>> new_mom_list(valarray<valarray<double>>(valarray<double>(0.0,12),mom_list.size()),njacks);

  for(int ijack=0;ijack<njacks;ijack++)
    for(size_t imom=0;imom<mom_list.size();imom++)
      {
	for(int i=0;i<4;i++)
	  new_mom_list[ijack][imom][i]=mom_list[imom][i];
	new_mom_list[ijack][imom][4]=p2[imom];
	new_mom_list[ijack][imom][6]=jZq[ijack][imom].real();
	for(int i=0;i<5;i++)
	  new_mom_list[ijack][imom][7+i]=jZ[ijack][imom][0+i].real();
	
      }
 
  //Assign the tag for fixed ijack
  int tag=0;
  double eps=1.0e-15;  //Precision: is it correct?
  for(size_t imom=0;imom<mom_list.size();imom++)
    {
      size_t count=0;
      for(size_t i=0;i<imom;i++)
	{
	  if((abs(new_mom_list[0][i][4]-new_mom_list[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_mom_list[i][0]-new_mom_list[imom][0])<eps*/ && \
	      abs(abs(new_mom_list[0][i][1])*abs(new_mom_list[0][i][2])*abs(new_mom_list[0][i][3])-(abs(new_mom_list[0][imom][1])*abs(new_mom_list[0][imom][2])*abs(new_mom_list[0][imom][3])))<eps ) || \
	     (abs(new_mom_list[0][i][4]-new_mom_list[0][imom][4])<eps/* && abs(p2_space[i]-p2_space[imom])<eps && abs(new_mom_list[i][0]+new_mom_list[imom][0]+1.)<eps*/ && \
	      abs(abs(new_mom_list[0][i][1])*abs(new_mom_list[0][i][2])*abs(new_mom_list[0][i][3])-(abs(new_mom_list[0][imom][1])*abs(new_mom_list[0][imom][2])*abs(new_mom_list[0][imom][3])))<eps  )  )
	    {
	      new_mom_list[0][imom][5]=new_mom_list[0][i][5];
	    }else{
	    count++;
	  }
	  
	  if(count==imom)
	    {
	      tag++;
	      new_mom_list[0][imom][5]=tag;
	    }
	}
    }
  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int ijack=1;ijack<njacks;ijack++)
      new_mom_list[ijack][imom][5]=new_mom_list[0][imom][5];

  cout<<"Number of equivalent momenta: "<<tag+1<<endl;
  
  //Average of Z's corresponding to equivalent momenta (same tag) and print on file
  valarray<double> p2_eq(valarray<double>(0.0,tag));
  /*
  valarray<valarray<valarray<double>>> jZ_tag(valarray<valarray<valarray<double>>>(valarray<valarray<double>>(valarray<double>(0.0,6),tag)),njacks);
  
  valarray<valarray<double>> Z_average(valarray<valarray<double>>(valarray<double>(0.0,6),tag));
  valarray<valarray<double>> Z2_average(valarray<valarray<double>>(valarray<double>(0.0,6),tag));
  valarray<valarray<double>> Z_error(valarray<valarray<double>>(valarray<double>(0.0,6),tag));
  */
  valarray<valarray<valarray<vector<double>>>> jZ_same_tag(valarray<valarray<vector<double>>>(valarray<vector<double>>(6),tag),njacks);
  valarray<valarray<valarray<double>>> jZ_average(valarray<valarray<double>>(valarray<double>(0.0,6),tag),njacks);
 
  cout<<"Averaging the Z's corresponding to equivalent momenta and printing on the output file..."<<endl;

  for(int ijack=0;ijack<njacks;ijack++)
    for(int t=0;t<tag;t++)
      {
	int count_equivalent=0;
	for(size_t imom=0;imom<mom_list.size();imom++)
	  { 
	    if(t==new_mom_list[ijack][imom][5])
	      {
		count_equivalent++;
		p2_eq[t]=new_mom_list[0][imom][4];
		
		jZ_same_tag[ijack][t][0].push_back(new_mom_list[ijack][imom][6]);
		jZ_same_tag[ijack][t][1].push_back(new_mom_list[ijack][imom][7]);
		jZ_same_tag[ijack][t][2].push_back(new_mom_list[ijack][imom][8]);
		jZ_same_tag[ijack][t][3].push_back(new_mom_list[ijack][imom][9]);
		jZ_same_tag[ijack][t][4].push_back(new_mom_list[ijack][imom][10]);
		jZ_same_tag[ijack][t][5].push_back(new_mom_list[ijack][imom][11]);
		
	      }
	  }
	for(int i=0;i<count_equivalent;i++)
	  {
	    /* jZ[t][0][i]=(Z_sum[t][0]-jZ[t][0][i])/((double)(count_equivalent-1));
	    jZ[t][1][i]=(Z_sum[t][1]-jZ[t][1][i])/((double)(count_equivalent-1));
	    jZ[t][2][i]=(Z_sum[t][2]-jZ[t][2][i])/((double)(count_equivalent-1));
	    jZ[t][3][i]=(Z_sum[t][3]-jZ[t][3][i])/((double)(count_equivalent-1));
	    jZ[t][4][i]=(Z_sum[t][4]-jZ[t][4][i])/((double)(count_equivalent-1));
	    jZ[t][5][i]=(Z_sum[t][5]-jZ[t][5][i])/((double)(count_equivalent-1));*/
	    
	    jZ_average[ijack][t][0]+=jZ_same_tag[ijack][t][0][i]/count_equivalent;
	    jZ_average[ijack][t][1]+=jZ_same_tag[ijack][t][1][i]/count_equivalent;
	    jZ_average[ijack][t][2]+=jZ_same_tag[ijack][t][2][i]/count_equivalent;
	    jZ_average[ijack][t][3]+=jZ_same_tag[ijack][t][3][i]/count_equivalent;
	    jZ_average[ijack][t][4]+=jZ_same_tag[ijack][t][4][i]/count_equivalent;
	    jZ_average[ijack][t][5]+=jZ_same_tag[ijack][t][5][i]/count_equivalent;
	    /*  
	    Z2_average[t][0]+=jZ[t][0][i]*jZ[t][0][i]/count_equivalent;
	    Z2_average[t][1]+=jZ[t][1][i]*jZ[t][1][i]/count_equivalent;
	    Z2_average[t][2]+=jZ[t][2][i]*jZ[t][2][i]/count_equivalent;
	    Z2_average[t][3]+=jZ[t][3][i]*jZ[t][3][i]/count_equivalent;
	    Z2_average[t][4]+=jZ[t][4][i]*jZ[t][4][i]/count_equivalent;
	    Z2_average[t][5]+=jZ[t][5][i]*jZ[t][5][i]/count_equivalent;
	    */
	  }
	/*
	for(int iz=0;iz<6;iz++)
	  {
	    Z_error[t][iz]=sqrt((double)(count_equivalent-1.))*sqrt(Z2_average[t][iz]-Z_average[t][iz]*Z_average[t][iz]);
	  }
	*/
    }


  valarray<valarray<double>> Z_mean_value(valarray<double>(0.0,6),tag);
  valarray<valarray<double>> Z2_mean_value(valarray<double>(0.0,6),tag);
  valarray<valarray<double>> Z_error(valarray<double>(0.0,6),tag);
  
  for(int t=0;t<tag;t++)
    for(int i=0;i<6;i++)
      {
	for(int ijack=0;ijack<njacks;ijack++)
	  {
	    Z_mean_value[t][i]+=jZ_average[ijack][t][i]/njacks;
	    Z2_mean_value[t][i]+=jZ_average[ijack][t][i]*jZ_average[ijack][t][i]/njacks;
	  }
	Z_error[t][i]=sqrt(njacks-1)*sqrt(Z2_mean_value[t][i]-Z_mean_value[t][i]*Z_mean_value[t][i]);
      }
  
  //output file
  ofstream outfile ("Z_average.txt");
  if (outfile.is_open())
    {	  
      outfile<<"##p2_tilde\t Zq\t Zq_err\t Zs\t Zs_err\t Zv(a)\t Zv(a)_err\t Zp\t Zp_err\t Za(v)\t Za(v)_err\t Zt\t Zt_err "<<endl;
      for(int t=0;t<tag;t++)
	{
	  outfile<<p2_eq[t]<<"\t"<<Z_mean_value[t][0]<<"\t"<<Z_error[t][0]<<"\t"<<Z_mean_value[t][1]<<"\t"<<Z_error[t][1]<<"\t"<<Z_mean_value[t][2]<<"\t"<<Z_error[t][2] \
		 <<"\t"<<Z_mean_value[t][3]<<"\t"<<Z_error[t][3]<<"\t"<<Z_mean_value[t][4]<<"\t"<<Z_error[t][4]<<"\t"<<Z_mean_value[t][5]<<"\t"<<Z_error[t][5]<<endl;
	}
      outfile.close();
    }
  else cout << "Unable to open the output file"<<endl;
  
  
  cout<<"End of the program."<<endl;
  
  return 0;
}


