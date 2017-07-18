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

//valarray of Eigen Vectors
using vXd_t=valarray<VectorXd>;

//useful notation
using jZ_t=vvd_t;
using jZbil_t=vvvvd_t;
using jproj_t=vvvvd_t;

//list of momenta
vector<coords_t> mom_list;

int nr,nm,nmr;



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

//read file
void read_internal(double &t,ifstream& infile)
{ 
  infile.read((char*) &t,sizeof(double));
}
//template <class T>
void read_internal(VectorXd &V, ifstream& infile)
{
  for(int i=0; i<V.size();i++) read_internal(V(i),infile);
}
template <class T>
void read_internal(valarray<T> &v, ifstream& infile)
{ 
  for(auto &i : v) read_internal(i,infile);
}
template <class T>
void read_vec( T &vec, const char* path)
{
  ifstream infile(path,ifstream::binary);

  if (infile.is_open())
    {
      for(auto &i : vec)
	  read_internal(i,infile);

      infile.close();

    }
  else cout << "Unable to open the input file "<<path<<endl;
}

//factorial
int fact(int n)
{
    if(n > 1)
        return n * fact(n - 1);
    else
        return 1;
}

vvvd_t average_Zq(vector<jZ_t> jZq)
{
  int moms=jZq.size();
  int njacks=jZq[0].size();
  int nmr=jZq[0][0].size();

  vvd_t Zq_ave(vd_t(nmr),moms), sqr_Zq_ave(vd_t(nmr),moms), Zq_err(vd_t(nmr),moms);
  vvvd_t Zq_ave_err(vvd_t(vd_t(nmr),moms),2); 

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
      Zq_err[imom][mr]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_ave[imom][mr]-Zq_ave[imom][mr]*Zq_ave[imom][mr]);
  
  Zq_ave_err[0]=Zq_ave;
  Zq_ave_err[1]=Zq_err;

  return Zq_ave_err;
}

vvd_t average_Zq_chiral(vector<vd_t> jZq)
{
  int moms=jZq.size();
  int njacks=jZq[0].size();

  vd_t Zq_ave(moms), sqr_Zq_ave(moms), Zq_err(moms);
  vvd_t Zq_ave_err(vd_t(moms),2); 

#pragma omp parallel for
  for(int imom=0;imom<moms;imom++)
    for(int ijack=0;ijack<njacks;ijack++)
      {
	Zq_ave[imom]+=jZq[imom][ijack]/njacks;
	sqr_Zq_ave[imom]+=jZq[imom][ijack]*jZq[imom][ijack]/njacks;
      }
  for(int imom=0;imom<moms;imom++)
    Zq_err[imom]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_ave[imom]-Zq_ave[imom]*Zq_ave[imom]);

  Zq_ave_err[0]=Zq_ave;
  Zq_ave_err[1]=Zq_err;

  return Zq_ave_err;
}

void plot_Zq_sub(vector<jZ_t> jZq, vector<jZ_t> jZq_sub, vector<double> p2_vector, const string &name, const string &all_or_eq_moms)
{
  vvvd_t Zq = average_Zq(jZq);  //Zq[ave/err][imom][nm]
  vvvd_t Zq_sub = average_Zq(jZq_sub);  //Zq[ave/err][imom][nm]
  
  ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
  ofstream datafile2("plot_data_and_script/plot_"+name+"_sub_"+all_or_eq_moms+"_data.txt");

  for(size_t imom=0;imom<p2_vector.size();imom++)
    {
      datafile1<<p2_vector[imom]<<"\t"<<Zq[0][imom][0]<<"\t"<<Zq[1][imom][0]<<endl;  //print only for M0R0
    }
  datafile1.close();
   for(size_t imom=0;imom<p2_vector.size();imom++)
    {
      datafile2<<p2_vector[imom]<<"\t"<<Zq_sub[0][imom][0]<<"\t"<<Zq_sub[1][imom][0]<<endl;  //print only for M0R0
    }
  datafile2.close();
  
  ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");

  scriptfile<<"set autoscale xy"<<endl;
  scriptfile<<"set xlabel '$\\tilde{p}^2$'"<<endl;
  scriptfile<<"set ylabel '$Z_q$'"<<endl;
  scriptfile<<"set yrange [0.7:0.9]"<<endl;
  scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_Q$'"<<endl;
  scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_sub_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'red' title '$Z_Q$ corrected'"<<endl;
  scriptfile<<"set terminal epslatex color"<<endl;
  if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile<<"set output 'allmoms/"<<name<<"_sub.tex'"<<endl;
  else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile<<"set output 'eqmoms/"<<name<<"_sub.tex'"<<endl;
  scriptfile<<"replot"<<endl;
  
  scriptfile.close();

  string command="gnuplot plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt";
  
  system(command.c_str());
  
}

///*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*///

int main(int narg,char **arg)
{

 if (narg!=6){
    cerr<<"Number of arguments not valid: <mom file> <nconfs> <njacks> <L> <T>"<<endl;
    exit(0);
  }

  
 //  int nconfs=stoi(arg[2]); 
  int njacks=stoi(arg[3]);
  // int clust_size=nconfs/njacks;
  // int conf_id[nconfs];
  // double L=stod(arg[4]),T=stod(arg[5]);
  // size_t nhits=1; //!

  nm = 4;  //! to be passed from command line
  nr = 2;
  nmr=nm*nr;

  int neq  = fact(nm+nr-1)/fact(nr)/fact(nm-1);
  int neq2=nm;
  
  read_mom_list(arg[1]);
  int moms=mom_list.size();
  int neq_moms=17; //////////////////

  vector<double> p2_vector_allmoms(moms);

  vector<jZ_t> jZq_allmoms(moms,vvd_t(vd_t(nmr),njacks)), jSigma1_allmoms(moms,vvd_t(vd_t(nmr),njacks)), \
    jZq_em_allmoms(moms,vvd_t(vd_t(nmr),njacks)), jSigma1_em_allmoms(moms,vvd_t(vd_t(nmr),njacks));
  vector<jZbil_t> jZ_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), \
    jZ_em_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_em_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks));
   
  vector<jZ_t> jZq_sub_allmoms(moms,vvd_t(vd_t(nmr),njacks)), jSigma1_sub_allmoms(moms,vvd_t(vd_t(nmr),njacks)),\
    jZq_em_sub_allmoms(moms,vvd_t(vd_t(nmr),njacks)), jSigma1_em_sub_allmoms(moms,vvd_t(vd_t(nmr),njacks));
  vector<jZbil_t> jZ_sub_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_sub_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)),\
    jZ_em_sub_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_em_sub_allmoms(moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks));
   
  vector<vvd_t> jGp_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)),\
    jGp_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks));
  vector<vvd_t> jGp_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), \
    jGp_em_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGs_em_subpole_allmoms(moms,vvd_t(vd_t(neq),njacks));

  vector<vvd_t> jGv_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGa_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)),\
    jGt_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks));
  vector<vvd_t> jGv_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jGa_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)),\
    jGt_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks));

  vector<vvd_t> jZq_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jSigma1_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)),\
    jZq_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)), jSigma1_em_equivalent_allmoms(moms,vvd_t(vd_t(neq),njacks)); 
   
  vector<vd_t> jGp_0_chiral_allmoms(moms,vd_t(njacks)),jGa_0_chiral_allmoms(moms,vd_t(njacks)),jGv_0_chiral_allmoms(moms,vd_t(njacks)),\
    jGs_0_chiral_allmoms(moms,vd_t(njacks)),jGt_0_chiral_allmoms(moms,vd_t(njacks));
  vector<vd_t> jGp_em_a_b_chiral_allmoms(moms,vd_t(njacks)),jGa_em_a_b_chiral_allmoms(moms,vd_t(njacks)),jGv_em_a_b_chiral_allmoms(moms,vd_t(njacks)), \
    jGs_em_a_b_chiral_allmoms(moms,vd_t(njacks)),jGt_em_a_b_chiral_allmoms(moms,vd_t(njacks));
  vector<vd_t> jZq_chiral_allmoms(moms,vd_t(njacks)),jSigma1_chiral_allmoms(moms,vd_t(njacks));
  vector<vd_t> jZq_em_chiral_allmoms(moms,vd_t(njacks)),jSigma1_em_chiral_allmoms(moms,vd_t(njacks));
  vector<vvd_t> jZ_chiral_allmoms(moms,vvd_t(vd_t(5),njacks)),jZ1_chiral_allmoms(moms,vvd_t(vd_t(5),njacks));
  vector<vvd_t> jZ_em_chiral_allmoms(moms,vvd_t(vd_t(5),njacks)),jZ1_em_chiral_allmoms(moms,vvd_t(vd_t(5),njacks));

  vector< vXd_t > jGp_pars_allmoms(moms,vXd_t(VectorXd(3),njacks)), jGs_pars_allmoms(moms,vXd_t(VectorXd(3),njacks)), \
    jGp_em_pars_allmoms(moms,vXd_t(VectorXd(3),njacks)), jGs_em_pars_allmoms(moms,vXd_t(VectorXd(3),njacks));
  vector< vXd_t > jGv_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGa_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)),\
    jGt_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGv_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)),\
    jGa_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jGt_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks));
  vector< vXd_t > jZq_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jSigma1_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)),\
    jZq_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)), jSigma1_em_pars_allmoms(moms,vXd_t(VectorXd(2),njacks)); 

  vd_t m_eff_equivalent(1.0,neq);
  vd_t m_eff_equivalent_Zq(0.0,neq2);

  vector<double> p2_vector_eqmoms(neq_moms);

  vector<jZ_t> jZq_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)), jSigma1_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)),\
    jZq_em_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)), jSigma1_em_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks));
  vector<jZbil_t> jZ_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)),\
    jZ_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks));
   
  vector<jZ_t> jZq_sub_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)), jSigma1_sub_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)),\
    jZq_em_sub_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)), jSigma1_em_sub_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks));
  vector<jZbil_t> jZ_sub_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_sub_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)),\
    jZ_em_sub_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_em_sub_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks));
   
  vector<vvd_t> jGp_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)),\
    jGp_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));
  vector<vvd_t> jGp_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), \
    jGp_em_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGs_em_subpole_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));

  vector<vvd_t> jGv_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGa_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)),\
    jGt_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));
  vector<vvd_t> jGv_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jGa_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)),\
    jGt_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks));

  vector<vvd_t> jZq_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jSigma1_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)),\
    jZq_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)), jSigma1_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq),njacks)); 
   
  vector<vd_t> jGp_0_chiral_eqmoms(neq_moms,vd_t(njacks)),jGa_0_chiral_eqmoms(neq_moms,vd_t(njacks)),jGv_0_chiral_eqmoms(neq_moms,vd_t(njacks)),\
    jGs_0_chiral_eqmoms(neq_moms,vd_t(njacks)),jGt_0_chiral_eqmoms(neq_moms,vd_t(njacks));
  vector<vd_t> jGp_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)),jGa_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)),jGv_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)), \
    jGs_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks)),jGt_em_a_b_chiral_eqmoms(neq_moms,vd_t(njacks));
  vector<vd_t> jZq_chiral_eqmoms(neq_moms,vd_t(njacks)),jSigma1_chiral_eqmoms(neq_moms,vd_t(njacks));
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

#define READ(NAME)				\
  read_vec(NAME##_##allmoms,"allmoms/"#NAME);	\
  read_vec(NAME##_##eqmoms,"eqmoms/"#NAME)
   
  READ(p2_vector);
  READ(jZq);
  READ(jSigma1);
  READ(jZq_em);
  READ(jSigma1_em);
  READ(jZ);
  READ(jZ1);
  READ(jZ_em);
  READ(jZ1_em);
  READ(jZq_sub);
  READ(jSigma1_sub);
  READ(jZq_em_sub);
  READ(jSigma1_em_sub);
  READ(jZ_sub);
  READ(jZ1_sub);
  READ(jZ_em_sub);
  READ(jZ1_em_sub);
  READ(jGp_equivalent);
  READ(jGs_equivalent);
  READ(jGp_subpole);
  READ(jGs_subpole);
  READ(jGv_equivalent);
  READ(jGa_equivalent);
  READ(jGt_equivalent);
  READ(jGp_em_equivalent);
  READ(jGs_em_equivalent);
  READ(jGp_em_subpole);
  READ(jGs_em_subpole);
  READ(jGv_em_equivalent);
  READ(jGa_em_equivalent);
  READ(jGt_em_equivalent);
  READ(jZq_equivalent);
  READ(jSigma1_equivalent);
  READ(jZq_em_equivalent);
  READ(jSigma1_em_equivalent);
  READ(jGp_0_chiral);
  READ(jGa_0_chiral);
  READ(jGv_0_chiral);
  READ(jGs_0_chiral);
  READ(jGt_0_chiral);
  READ(jGp_em_a_b_chiral);
  READ(jGa_em_a_b_chiral);
  READ(jGv_em_a_b_chiral);
  READ(jGs_em_a_b_chiral);
  READ(jGt_em_a_b_chiral);
  READ(jZq_chiral);
  READ(jSigma1_chiral);
  READ(jZq_em_chiral);
  READ(jSigma1_em_chiral);
  READ(jZ_chiral);
  READ(jZ1_chiral);
  READ(jZ_em_chiral);
  READ(jZ1_em_chiral);
  READ(jGp_pars);
  READ(jGp_em_pars);
  READ(jGs_pars);
  READ(jGs_em_pars);
  READ(jGv_pars);
  READ(jGv_em_pars);
  READ(jGa_pars);
  READ(jGa_em_pars);
  READ(jGt_pars);
  READ(jGt_em_pars);
  READ(jZq_pars);
  READ(jZq_em_pars);
  READ(jSigma1_pars);
  READ(jSigma1_em_pars);
   
#undef READ

   read_vec(m_eff_equivalent,"allmoms/m_eff_equivalent");
   read_vec(m_eff_equivalent_Zq,"allmoms/m_eff_equivalent");
   

   //   SECTIONS:
   // plot;
   // plot_with_em;
   // plot_sub;
   // plot_sub_with_em;
   // plot_Goldstone;
   // plot_chiral_extrapolation;
   // plot_Golstone_with_em;
   // plot_chiral_extrapolation_with_em;
   // plot_chiral;
   // plot_chiral_with_em;

   //   SUBSECTIONS:
   // plot_Zq (Zq, Sigma1, ...)  
   // plot_Z

   plot_Zq_sub(jZq_eqmoms,jZq_sub_eqmoms,p2_vector_eqmoms,"Zq","eqmoms");
   plot_Zq_sub(jSigma1_eqmoms,jSigma1_sub_eqmoms,p2_vector_eqmoms,"Sigma1","eqmoms");

   vector<jZ_t> jZq_with_em_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)), jSigma1_with_em_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks));
   vector<jZ_t> jZq_sub_with_em_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks)), jSigma1_sub_with_em_eqmoms(neq_moms,vvd_t(vd_t(nmr),njacks));

#pragma omp parallel for collapse(3)
   for(int imom=0;imom<neq_moms;imom++)
     for(int mr=0;mr<nmr;mr++)
       for(int ijack=0;ijack<njacks;ijack++)
	 {
	   jZq_with_em_eqmoms[imom][ijack][mr]=jZq_eqmoms[imom][ijack][mr]+jZq_em_eqmoms[imom][ijack][mr];
	   jSigma1_with_em_eqmoms[imom][ijack][mr]=jSigma1_eqmoms[imom][ijack][mr]+jSigma1_em_eqmoms[imom][ijack][mr];
	   jZq_sub_with_em_eqmoms[imom][ijack][mr]=jZq_sub_eqmoms[imom][ijack][mr]+jZq_em_sub_eqmoms[imom][ijack][mr];
	   jSigma1_sub_with_em_eqmoms[imom][ijack][mr]=jSigma1_sub_eqmoms[imom][ijack][mr]+jSigma1_em_sub_eqmoms[imom][ijack][mr];
	 }

   
   plot_Zq_sub(jZq_with_em_eqmoms,jZq_sub_with_em_eqmoms,p2_vector_eqmoms,"Zq_with_em","eqmoms");
   plot_Zq_sub(jSigma1_with_em_eqmoms,jSigma1_sub_with_em_eqmoms,p2_vector_eqmoms,"Sigma1_with_em","eqmoms");


   // plot_Zq_sub(jSigma1_eqmoms,jSigma1_sub_eqmoms,p2_vector_eqmoms,"jZq","eqmoms");

   // vvd_t Zq_eq = average_Zq_chiral(jZq_chiral_eqmoms); 
  
   // for(int ieq=0;ieq<neq;ieq++)
   //   {
   //     cout<<m_eff_equivalent[ieq]<<"\t"<<jGp_equivalent_eqmoms[0][0][ieq]<<endl;
   //   }

   
  return 0;
}


 // vvvd_t Zq_allmoms=average_Zq(jZq_allmoms), Zq_eqmoms=average_Zq(jZq_eqmoms), Zq_sub_allmoms=average_Zq(jZq_sub_allmoms), Zq_sub_eqmoms=average_Zq(jZq_sub_eqmoms);
   // vvvd_t Sigma1_allmoms=average_Zq(jSigma1_allmoms), Sigma1_eqmoms=average_Zq(jSigma1_eqmoms),\
   //   Sigma1_sub_allmoms=average_Zq(jSigma1_sub_allmoms), Sigma1_sub_eqmoms=average_Zq(jSigma1_sub_eqmoms);

   // vvvd_t Zq_em_allmoms=average_Zq(jZq_em_allmoms), Zq_em_eqmoms=average_Zq(jZq_em_eqmoms), Zq_em_sub_allmoms=average_Zq(jZq_em_sub_allmoms), Zq_em_sub_eqmoms=average_Zq(jZq_em_sub_eqmoms);
   // vvvd_t Sigma1_em_allmoms=average_Zq(jSigma1_em_allmoms), Sigma1_em_eqmoms=average_Zq(jSigma1_em_eqmoms),\
   //   Sigma1_em_sub_allmoms=average_Zq(jSigma1_em_sub_allmoms), Sigma1_em_sub_eqmoms=average_Zq(jSigma1_em_sub_eqmoms);

   // vvd_t Zq_chiral_allmoms=average_Zq_chiral(jZq_chiral_allmoms), Zq_chiral_eqmoms=average_Zq_chiral(jZq_chiral_eqmoms);

   // vvvvvd_t /*Z_allmoms=average_Z(jZ_allmoms),*/ Z_eqmoms=average_Z(jZ_eqmoms),/*Z1_allmoms=average_Z(jZ1_allmoms),*/ Z1_eqmoms=average_Z(jZ1_eqmoms);
   // vvvvvd_t /*Z_sub_allmoms=average_Z(jZ_sub_allmoms),*/ Z_sub_eqmoms=average_Z(jZ_sub_eqmoms),/*Z1_sub_allmoms=average_Z(jZ1_sub_allmoms),*/ Z1_sub_eqmoms=average_Z(jZ1_sub_eqmoms);
   
 

