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
  real_part[4]={1,1,1,1};
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
string path_to_conf(int i_conf,const char *name)
{
  char path[1024];
  sprintf(path,"out%d/fft_%s",i_conf,name);
  return path;
}

//jackknife Propagator
valarray<valarray<prop_t>> jackknife_prop(  valarray<valarray<prop_t>> jS, int nconf, int clust_size )
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
valarray<valarray<qline_t>> jackknife_vertex( valarray<valarray<qline_t>> jVert, int nconf, int clust_size )
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


//average Propagator
valarray<prop_t>  average_prop(  valarray<valarray<prop_t>> jS )
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
  valarray<qline_t> average_vertex( valarray<valarray<qline_t>> jVert)
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
valarray<prop_t>  error_prop(  valarray<valarray<prop_t>> jS ,valarray<prop_t> meanS )
{
  int njacks = jS.size();
  valarray<prop_t> jsq_mean(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  valarray<prop_t> err;
  
  //sum of (jS)^2
  for(int j=0;j<njacks;j++) jsq_mean+= jS[j]*jS[j];
  //divide for njacks each component of jsq_mean (one for each moment)
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
valarray<qline_t> error_vertex( valarray<valarray<qline_t>> jVert, valarray<qline_t> meanVert )
{
 int njacks = jVert.size();
 valarray<qline_t> jsq_mean(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()));
 valarray<qline_t> err;

 //sum of (jVert)^2
 for(int j=0;j<njacks;j++) jsq_mean+= jVert[j]*jVert[j];
 //divide for njacks each component of jsq_mean (one for each moment)
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

//compute Zq
valarray<dcompl> compute_Zq(vprop_t GAMMA, valarray<prop_t> S_inv)
{
  //compute p_slash as a vector of prop-type matrices
  valarray<prop_t> p_slash(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int igam=1;igam<5;igam++)
      p_slash[imom]+=GAMMA[igam]*mom_list[imom][igam-1];
  
  //compute p^2
  valarray<double> p2(valarray<double>(0.0,mom_list.size()));
  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int coord=0;coord<4;coord++)
      p2[imom]+=mom_list[imom][coord]*mom_list[imom][coord];
  
  //compute Zq = Quark field RC (RI'-MOM), one for each momentum
  valarray<dcompl> Zq(mom_list.size());
  complex<double> I(0,1);
  
  for(size_t imom=0;imom<mom_list.size();imom++)
    Zq[imom]=(p_slash[imom]*S_inv[imom]).trace();
  for(auto &it : Zq)
    it=-I*it/12.;
  for(size_t imom=0;imom<mom_list.size();imom++)
    Zq[imom]/=p2[imom];
  
  return Zq;
  
}

//project the amputated green function
  valarray<valarray<complex<double>>> project(vprop_t GAMMA, valarray<qline_t> Lambda)
{

  //L_proj has 5 components: S(0), V(1), P(2), A(3), T(4)
  valarray<qline_t> L_proj(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),5),mom_list.size()));
  valarray<valarray<complex<double>>> G(valarray<dcompl>(0.0,5),mom_list.size());
  vprop_t P(valarray<prop_t>(prop_t::Zero(),16));
  
  //create projectors such that tr(GAMMA*P)=Identity
  P[0]=GAMMA[0]; //scalar
  for(int igam=1;igam<5;igam++)  //vector
    P[igam]=GAMMA[igam].adjoint()/4.; 
  P[5]=GAMMA[5];  //pseudoscalar
  for(int igam=6;igam<10;igam++)  //axial
    P[igam]=GAMMA[igam].adjoint()/4.;
  for(int igam=10;igam<16;igam++)  //tensor
    P[igam]=GAMMA[igam].adjoint();
  
  for(size_t imom=0;imom<mom_list.size();imom++)
    {
      L_proj[imom][0]=Lambda[imom][0]*P[0];
      for(int igam=1;igam<5;igam++)
	L_proj[imom][1]+=Lambda[imom][igam]*P[igam];
      L_proj[imom][2]=Lambda[imom][5]*P[5];
      for(int igam=6;igam<10;igam++)  
	L_proj[imom][3]+=Lambda[imom][igam]*P[igam];
      for(int igam=10;igam<16;igam++)  
	L_proj[imom][4]+=Lambda[imom][igam]*P[igam];
      
      for(int j=0;j<5;j++)
	G[imom][j]=L_proj[imom][j].trace()/12.;
    }
  
  return G;
  
}


  
  int main(int narg,char **arg)
{
  
  int nconfs=2;
  int njacks=nconfs;
  int clust_size=nconfs/njacks;
  
  read_mom_list("mom_list.txt");
  
  //create gamma matrices
  vprop_t GAMMA=make_gamma();
  
  // put to zero jackknife vertex
  valarray<valarray<qline_t>> jVert(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks);
  valarray<valarray<prop_t>> jS(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);
  
  
  for(int iconf=0;iconf<nconfs;iconf++)
    {
      int ijack=iconf/clust_size;
      
      //create a propagator in a given configuration
      vprop_t S=read_prop(path_to_conf(iconf+1,"SPECT0"));

      // cout<<S[255]<<endl;
      
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

  
  //compute fluctuations of the propagator
  jS=jackknife_prop(jS,nconfs,clust_size);
  //compute fluctuations of the vertex
  jVert=jackknife_vertex(jVert,nconfs,clust_size);
  
  //compute the average of the propagator
  valarray<prop_t> S_mean=average_prop(jS);
  //compute the average of the vertex
  valarray<qline_t> Vertex_mean=average_vertex(jVert);

  //compute the error on the propagator
  valarray<prop_t> S_error=error_prop(jS,S_mean);
  //compute the error on the vertex
  valarray<qline_t> Vertex_error=error_vertex(jVert,Vertex_mean);
  

  //inverse of the (average) propagator
  valarray<prop_t> S_inv=S_mean;
  for(auto &it : S_inv)
    it=it.inverse();

  

  //amputate external legs
  valarray<qline_t> Lambda(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()));
  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int igam=0;igam<16;igam++)
      {
	Lambda[imom][igam]=S_inv[imom]*Vertex_mean[imom][igam]*S_inv[imom];
      }

  //compute Zq according to RI'-MOM, one for each momentum
  valarray<dcompl> Zq=compute_Zq(GAMMA,S_inv);

  //compute the projected green function as a vector (S,V,P,A,T)
  valarray<valarray<complex<double>>> G=project(GAMMA,Lambda);

  //compute Z's according to RI-MOM, one for each momentum
  valarray<valarray<dcompl>> Z(valarray<dcompl>(5),mom_list.size());
  
  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int k=0;k<5;k++)
      Z[imom][k]=Zq[imom]/G[imom][k];

  
  return 0;
}
