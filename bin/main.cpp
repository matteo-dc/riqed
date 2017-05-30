#ifdef HAVE_CONFIG_H
 #include <config.hpp>
#endif

#include <complex>
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <valarray>

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
  vprop_t out(mom_list.size());
  
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
  
  //gamma_mu*gamma5   --------invertire posizione!
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
      vert[mu]=prop[mom]*gamma[mu]*prop[mom];  /*it has to be "jackknifed"*/
    }
  return vert;
}

string path_to_conf(int i_conf,const char *name)
{
  char path[1024];
  sprintf(path,"out%d/fft_%s",i_conf,name);
  return path;
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
  //valarray<qline_t> jS(njacks);
  
  for(int iconf=0;iconf<nconfs;iconf++)
    {
       int ijack=iconf/clust_size;
       
      //create a propagator ----in a given configuration
      vprop_t S=read_prop(path_to_conf(iconf+1,"SPECT0"));
      
       for(size_t imom=0;imom<mom_list.size();imom++)
	 {
	   //create vertex functions with the i_mom=0 momentum
	   qline_t Vert=make_vertex(S,imom,GAMMA);
	   
	   jVert[ijack][imom]+=Vert;
	 }
       
      cout<<S.size()<<endl;
    }

  return 0;
}
