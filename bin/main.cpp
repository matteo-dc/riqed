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

//compute Zq
valarray<dcompl> compute_Zq(vprop_t GAMMA, valarray<prop_t> S_inv)
{
  double Lx=24,Ly=24,Lz=24,Lt=48;
  double V=Lx*Ly*Lz*Lt;
  
  //compute p_slash as a vector of prop-type matrices
  valarray<valarray<double>> p(valarray<double>(0.0,4),mom_list.size());
  valarray<valarray<double>> p_tilde(valarray<double>(0.0,4),mom_list.size());
  valarray<prop_t> p_slash(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  valarray<double> p2(valarray<double>(0.0,mom_list.size()));
  valarray<dcompl> Zq(mom_list.size());
  complex<double> I(0,1);

  for(size_t imom=0;imom<mom_list.size();imom++)
    {
      p[imom]={2*M_PI*mom_list[imom][1]/Lx,2*M_PI*mom_list[imom][2]/Ly,2*M_PI*mom_list[imom][3]/Lz,2*M_PI*(mom_list[imom][0]+0.5)/Lt};
      p_tilde[imom]={sin(p[imom][0]),sin(p[imom][1]),sin(p[imom][2]),sin(p[imom][3])};
      
      for(int igam=1;igam<5;igam++)
	//	p_slash[imom]+=GAMMA[igam]*p[imom][igam-1];
	p_slash[imom]+=GAMMA[igam]*p_tilde[imom][igam-1];
  
  /*  Note that: p_slash*p_slash=p2*GAMMA[0]  */
  
  //compute p^2
      for(int coord=0;coord<4;coord++)
	//	p2[imom]+=p[imom][coord]*p[imom][coord];
	p2[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
      
      //compute Zq = Quark field RC (RI'-MOM), one for each momentum  
      Zq[imom]=(p_slash[imom]*S_inv[imom]).trace();
      Zq[imom]/=p2[imom];    
      Zq[imom]=-I*Zq[imom]/12./V;
    }
  
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
  //int nconfs=2; //OLD CONFS
  int nconfs=15; //NEW CONFS
  int njacks=nconfs;
  int clust_size=nconfs/njacks;
  int conf_id[nconfs]={700,710,720,730,740,750,760,770,780,790,800,810,820,830,840};

  cout<<"Reading the list of momenta..."<<endl; /******/


  read_mom_list("mom_list_free.txt");  //NEW CONFS FREE!!
  // read_mom_list("mom_list.txt");  //NEW CONFS!!
  // read_mom_list("mom_list_OLD.txt"); //OLD CONFS!!

  cout<<"Creating Dirac gamma matrices..."<<endl; /******/
  
  //create gamma matrices
  vprop_t GAMMA=make_gamma();
  
  // put to zero jackknife vertex
  valarray<valarray<qline_t>> jVert(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()),njacks);
  valarray<valarray<prop_t>> jS(valarray<prop_t>(prop_t::Zero(),mom_list.size()),njacks);

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
  
  valarray<prop_t> S_inv(valarray<prop_t>(prop_t::Zero(),mom_list.size()));
  valarray<qline_t> Lambda(valarray<qline_t>(valarray<prop_t>(prop_t::Zero(),16),mom_list.size()));
  
  cout<<"   Jackknife of propagators (1/6)"<<endl;
  //compute fluctuations of the propagator
  jS=jackknife_prop(jS,nconfs,clust_size);
  cout<<"   Jackknife of vertices (2/6)"<<endl;
  //compute fluctuations of the vertex
  jVert=jackknife_vertex(jVert,nconfs,clust_size);
  
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
  
  cout<<"Inverting the propagators..."<<endl;
  
  //inverse of the (average) propagator
      
  for(size_t imom=0;imom<mom_list.size();imom++)
    S_inv[imom]=S_mean[imom].inverse();
      
  
  
  cout<<"Amputating the external legs..."<<endl;
  
  //amputate external legs

  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int igam=0;igam<16;igam++)
      {
	Lambda[imom][igam]=S_inv[imom]*Vertex_mean[imom][igam]*GAMMA[5]*S_inv[imom].adjoint()*GAMMA[5];
      }
  
  cout<<"Computing Zq..."<<endl;
  
  //compute Zq according to RI'-MOM, one for each momentum
  valarray<dcompl> Zq=compute_Zq(GAMMA,S_inv);

  cout<<"Projecting the Green functions..."<<endl;
  
  //compute the projected green function as a vector (S,V,P,A,T)
  valarray<valarray<complex<double>>> G=project(GAMMA,Lambda);

  cout<<"Computing the Z's..."<<endl;
  
  //compute Z's according to RI-MOM, one for each momentum
  valarray<valarray<dcompl>> Z(valarray<dcompl>(5),mom_list.size());
  
  for(size_t imom=0;imom<mom_list.size();imom++)
    for(int k=0;k<5;k++)
      Z[imom][k]=Zq[imom]/G[imom][k];
  
  

  //create p_tilde vector  
  valarray<valarray<double>> p(valarray<double>(0.0,4),mom_list.size());
  valarray<valarray<double>> p_tilde(valarray<double>(0.0,4),mom_list.size());
  valarray<double> p2(valarray<double>(0.0,mom_list.size()));
  double L=24.,T=48.;

  for(size_t imom=0;imom<mom_list.size();imom++)
	{
	  p[imom]={2*M_PI*mom_list[imom][1]/L,2*M_PI*mom_list[imom][2]/L,2*M_PI*mom_list[imom][3]/L,2*M_PI*(mom_list[imom][0]+0.5)/T};
	  p_tilde[imom]={sin(p[imom][0]),sin(p[imom][1]),sin(p[imom][2]),sin(p[imom][3])};

	  for(int coord=0;coord<4;coord++)
	    p2[imom]+=p_tilde[imom][coord]*p_tilde[imom][coord];
	}
  
  //Create new extended vector
  cout<<"Creating Z average vector..."<<endl;
  
  valarray<valarray<double>> new_mom_list(valarray<double>(0.0,12),mom_list.size());
  
  for(size_t imom=0;imom<mom_list.size();imom++)
    new_mom_list[imom]=create_extended_vector(p2,imom,Zq,Z);

 
  //Assign the tag
  int tag=0;
  double eps=1.0e-7;  //Precision: is it correct?
  for(size_t imom=0;imom<mom_list.size();imom++)
    {
      size_t count=0;
      for(size_t i=0;i<imom;i++)
	{
	  if(abs(new_mom_list[i][4]-new_mom_list[imom][4])<eps/* && ( new_mom_list[i][1]==-new_mom_list[imom][1] || \
								    new_mom_list[i][2]==-new_mom_list[imom][2] || \
								    new_mom_list[i][3]==-new_mom_list[imom][3] || \
								    new_mom_list[i][0]==-new_mom_list[imom][0] || \
								    abs(new_mom_list[i][1])+abs(new_mom_list[i][2])+abs(new_mom_list[i][3])== \
								    abs(new_mom_list[imom][1])+abs(new_mom_list[imom][2])+abs(new_mom_list[imom][3]) )*/)
	    {
	      new_mom_list[imom][5]=new_mom_list[i][5];
	    }else{
	    count++;
	  }
	  
	  if(count==imom)
	    {
	      tag++;
	      new_mom_list[imom][5]=tag;
	    }
	}
    }

  //Average of Z corresponding to equivalent momenta (same tag) and print on file
  valarray<double> p2_eq(valarray<double>(0.0,tag));
  valarray<valarray<double>> Z_average(valarray<valarray<double>>(valarray<double>(0.0,6),tag));
  
  cout<<"Averaging the Z's corresponding to equivalent momenta and printing on the output file..."<<endl;
  
  for(int t=0;t<tag;t++)
    {
      int count_equivalent=0;
      for(size_t imom=0;imom<mom_list.size();imom++)
	{ 
	  if(t==new_mom_list[imom][5])
	    {
	      count_equivalent++;
	      p2_eq[t]=new_mom_list[imom][4];
	      Z_average[t][0]+=new_mom_list[imom][6]; //Zq
	      Z_average[t][1]+=new_mom_list[imom][7]; //Zs
	      Z_average[t][2]+=new_mom_list[imom][8]; //Zv (a)
	      Z_average[t][3]+=new_mom_list[imom][9]; //Zp
	      Z_average[t][4]+=new_mom_list[imom][10];//Za  (v)
	      Z_average[t][5]+=new_mom_list[imom][11];//Zt
	      
	    }
	}
      for(int i=0;i<6;i++)
	Z_average[t][i]/=(double)count_equivalent;
      
      
      //output file
      ofstream outfile ("Z_average.txt");
      if (outfile.is_open())
	{	  
	  outfile<<"##p2_tilde\t Zq\t Zs\t Zv(a)\t Zp\t Za(v)\t Zt "<<endl;
	  for(int t=0;t<tag;t++)
	    {
	      outfile<<p2_eq[t]<<"\t"<<Z_average[t][0]<<"\t"<<Z_average[t][1]<<"\t"<<Z_average[t][2]<<"\t"<<Z_average[t][3]<<"\t"<<Z_average[t][4]<<"\t"<<Z_average[t][5]<<endl;
	    }
	  outfile.close();
	}
      else cout << "Unable to open the output file"<<endl;
    }
  
  cout<<"End of the program."<<endl;
  
  return 0;
}


