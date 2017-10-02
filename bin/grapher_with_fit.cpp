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

valarray<VectorXd> fit_chiral_jackknife(const vvd_t &coord, vd_t &error, const vector<vd_t> &y, const int range_min, const int range_max)
{
    
    int n_par = coord.size();
    int njacks = y[0].size();
    
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
    for(int i=range_min; i<range_max; i++)
    {
      error[i]+=1e-300;
      
        for(int j=0; j<n_par; j++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
        
        for(int ijack=0; ijack<njacks; ijack++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) Sy[ijack](k) += y[i][ijack]*coord[k][i]/(error[i]*error[i]);
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
    
    return jpars;
    
}

valarray< valarray<VectorXd> > fit_chiral_Z_jackknife(const vvd_t &coord, vvd_t &error, const vector<vvd_t> &y, const int range_min, const int range_max)
{
  // cout<<"DEBUG---(a)"<<endl;
  
  int n_par = coord.size();
  int njacks = y[0].size();
  int nbil = y[0][0].size();

  // cout<<"DEBUG---(b)"<<endl;
    
  valarray<MatrixXd> S(MatrixXd(n_par,n_par),nbil);
  valarray< valarray<VectorXd> > Sy(valarray<VectorXd>(VectorXd(n_par),njacks),nbil);
  valarray< valarray<VectorXd> > jpars(valarray<VectorXd>(VectorXd(n_par),njacks),nbil);

  // cout<<"DEBUG---(c)"<<endl;
    
  //initialization
  for(int ibil=0; ibil<nbil;ibil++)
    S[ibil]=MatrixXd::Zero(n_par,n_par);

  // cout<<"DEBUG---(d)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ijack=0; ijack<njacks; ijack++)
      {
	//cout<<"a"<<endl;
	Sy[ibil][ijack]=VectorXd::Zero(n_par);
	//cout<<"b"<<endl;
	jpars[ibil][ijack]=VectorXd::Zero(n_par);
      }

  // cout<<"DEBUG---(d')"<<endl;
    
  //definition
  for(int i=range_min; i<range_max; i++)
    {
      for(int ibil=0; ibil<nbil;ibil++)
	error[i][ibil]+=1e-300;
      
      for(int ibil=0; ibil<nbil;ibil++)
        for(int j=0; j<n_par; j++)
	  for(int k=0; k<n_par; k++)
	    if(std::isnan(error[i][ibil])==0) S[ibil](j,k) += coord[j][i]*coord[k][i]/(error[i][ibil]*error[i][ibil]);

      for(int ibil=0; ibil<nbil;ibil++)
        for(int ijack=0; ijack<njacks; ijack++)
	  for(int k=0; k<n_par; k++)
	    if(std::isnan(error[i][ibil])==0) Sy[ibil][ijack](k) += y[i][ijack][ibil]*coord[k][i]/(error[i][ibil]*error[i][ibil]);
    }

  // cout<<"DEBUG---(e)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ijack=0; ijack<njacks; ijack++)
      jpars[ibil][ijack] = S[ibil].colPivHouseholderQr().solve(Sy[ibil][ijack]);

  // cout<<"DEBUG---(f)"<<endl;
    
  return jpars; //jpars[ibil][ijack][ipar]
    
}


valarray< valarray<VectorXd> > fit_chiral_Z_RIp_jackknife(const vvd_t &coord, const vvd_t &error, const vector<vvd_t> &y, const int range_min, const int range_max, const double &p_min_value)
{
  // cout<<"DEBUG---(a)"<<endl;
  
  int n_par = coord.size();
  int njacks = y[0].size();
  int nbil = y[0][0].size();

  // cout<<"DEBUG---(b)"<<endl;
    
  valarray<MatrixXd> S(MatrixXd(n_par,n_par),nbil);
  valarray< valarray<VectorXd> > Sy(valarray<VectorXd>(VectorXd(n_par),njacks),nbil);
  valarray< valarray<VectorXd> > jpars(valarray<VectorXd>(VectorXd(n_par),njacks),nbil);

  // cout<<"DEBUG---(c)"<<endl;
    
  //initialization
  for(int ibil=0; ibil<nbil;ibil++)
    S[ibil]=MatrixXd::Zero(n_par,n_par);

  // cout<<"DEBUG---(d)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ijack=0; ijack<njacks; ijack++)
      {
	//cout<<"a"<<endl;
	Sy[ibil][ijack]=VectorXd::Zero(n_par);
	//cout<<"b"<<endl;
	jpars[ibil][ijack]=VectorXd::Zero(n_par);
      }

  // cout<<"DEBUG---(d')"<<endl;
    
  //definition
  for(int i=range_min; i<range_max; i++)
    {
      for(int ibil=0; ibil<nbil;ibil++)
	error[i][ibil]+=1e-300;
      
      if(coord[1][i]>p_min_value)
	{
      
	  for(int ibil=0; ibil<nbil;ibil++)
	    for(int j=0; j<n_par; j++)
	      for(int k=0; k<n_par; k++)
		if(std::isnan(error[i][ibil])==0) S[ibil](j,k) += coord[j][i]*coord[k][i]/(error[i][ibil]*error[i][ibil]);

	  for(int ibil=0; ibil<nbil;ibil++)
	    for(int ijack=0; ijack<njacks; ijack++)
	      for(int k=0; k<n_par; k++)
		if(std::isnan(error[i][ibil])==0) Sy[ibil][ijack](k) += y[i][ijack][ibil]*coord[k][i]/(error[i][ibil]*error[i][ibil]);
	}
    }

  // cout<<"DEBUG---(e)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ijack=0; ijack<njacks; ijack++)
      jpars[ibil][ijack] = S[ibil].colPivHouseholderQr().solve(Sy[ibil][ijack]);

  // cout<<"DEBUG---(f)"<<endl;
    
  return jpars; //jpars[ibil][ijack][ipar]
    
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
	  Zq_err[imom][mr]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[imom][mr]-Zq_ave[imom][mr]*Zq_ave[imom][mr]));
    
    Zq_ave_err[0]=Zq_ave;
    Zq_ave_err[1]=Zq_err;
    
    return Zq_ave_err;
}

vvd_t average_Zq_chiral(vector<vd_t> &jZq)
{
    int moms=jZq.size();
    int njacks=jZq[0].size();
    
    vd_t Zq_ave(0.0,moms), sqr_Zq_ave(0.0,moms), Zq_err(0.0,moms);
    vvd_t Zq_ave_err(vd_t(0.0,moms),2);
    
#pragma omp parallel for
    for(int imom=0;imom<moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            Zq_ave[imom]+=jZq[imom][ijack]/njacks;
            sqr_Zq_ave[imom]+=jZq[imom][ijack]*jZq[imom][ijack]/njacks;
        }
#pragma omp parallel for
    for(int imom=0;imom<moms;imom++)
      Zq_err[imom]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_ave[imom]-Zq_ave[imom]*Zq_ave[imom]));
    
    Zq_ave_err[0]=Zq_ave;
    Zq_ave_err[1]=Zq_err;
    
    return Zq_ave_err;
}

vvvd_t average_pars(vector<vXd_t> &jZq_pars)
{
    int moms=jZq_pars.size();
    int njacks=jZq_pars[0].size();
    int pars=jZq_pars[0][0].size();
    
    vvd_t Zq_par_ave(vd_t(0.0,pars),moms), sqr_Zq_par_ave(vd_t(0.0,pars),moms), Zq_par_err(vd_t(0.0,pars),moms);
    vvvd_t Zq_par_ave_err(vvd_t(vd_t(0.0,pars),moms),2);
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int ipar=0;ipar<pars;ipar++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                Zq_par_ave[imom][ipar]+=jZq_pars[imom][ijack](ipar)/njacks;
                sqr_Zq_par_ave[imom][ipar]+=jZq_pars[imom][ijack](ipar)*jZq_pars[imom][ijack](ipar)/njacks;
            }
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int ipar=0;ipar<pars;ipar++)
	  Zq_par_err[imom][ipar]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_par_ave[imom][ipar]-Zq_par_ave[imom][ipar]*Zq_par_ave[imom][ipar]));
    
    Zq_par_ave_err[0]=Zq_par_ave;
    Zq_par_ave_err[1]=Zq_par_err;
    
    return Zq_par_ave_err;
    
}

vvvvvd_t average_Z(vector<jZbil_t> &jZ)
{
    int moms=jZ.size();
    int njacks=jZ[0].size();
    int nmr=jZ[0][0].size();
    int nbil=5;
    
    vvvvd_t Z_ave(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms), sqr_Z_ave(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms), Z_err(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms);
    vvvvvd_t Z_ave_err(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),moms),2);
    
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<moms;imom++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int k=0;k<nbil;k++)
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
                        Z_ave[imom][mr_fw][mr_bw][k]+=jZ[imom][ijack][mr_fw][mr_bw][k]/njacks;
                        sqr_Z_ave[imom][mr_fw][mr_bw][k]+=jZ[imom][ijack][mr_fw][mr_bw][k]*jZ[imom][ijack][mr_fw][mr_bw][k]/njacks;
                    }
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<moms;imom++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int k=0;k<nbil;k++)
		  Z_err[imom][mr_fw][mr_bw][k]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_ave[imom][mr_fw][mr_bw][k]-Z_ave[imom][mr_fw][mr_bw][k]*Z_ave[imom][mr_fw][mr_bw][k]));
    
    Z_ave_err[0]=Z_ave;
    Z_ave_err[1]=Z_err;
    
    return Z_ave_err;
}

vvvd_t average_Z_chiral(vector<vvd_t> &jZ_chiral)
{
    int moms=jZ_chiral.size();
    int njacks=jZ_chiral[0].size();
    int nbil=5;
    
    vvd_t Z_chiral_ave(vd_t(0.0,nbil),moms), sqr_Z_chiral_ave(vd_t(0.0,nbil),moms), Z_chiral_err(vd_t(0.0,nbil),moms);
    vvvd_t Z_chiral_ave_err(vvd_t(vd_t(0.0,nbil),moms),2);
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int k=0;k<nbil;k++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                Z_chiral_ave[imom][k]+=jZ_chiral[imom][ijack][k]/njacks;
                sqr_Z_chiral_ave[imom][k]+=jZ_chiral[imom][ijack][k]*jZ_chiral[imom][ijack][k]/njacks;
            }
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int k=0;k<nbil;k++)
	  Z_chiral_err[imom][k]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_chiral_ave[imom][k]-Z_chiral_ave[imom][k]*Z_chiral_ave[imom][k]));
    
    Z_chiral_ave_err[0]=Z_chiral_ave;
    Z_chiral_ave_err[1]=Z_chiral_err;
    
    return Z_chiral_ave_err;
}

void plot_Zq_sub(vector<jZ_t> &jZq, vector<jZ_t> &jZq_sub, vector<double> &p2_vector, const string &name, const string &all_or_eq_moms)
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
    scriptfile<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;
    scriptfile<<"set ylabel '$Z_q$'"<<endl;
    scriptfile<<"set xrange [0:2.5]"<<endl;
    //  if(name=="Sigma1_em_correction")
    //  scriptfile<<"set yrange [-0.17:-0.11]"<<endl;
    scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q$'"<<endl;
    scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_sub_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lt 1 lc rgb 'red' title '$Z_q^{\\mathrm{corr.}}$'"<<endl;
    scriptfile<<"set terminal epslatex color"<<endl;
    if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile<<"set output 'allmoms/"<<name<<"_sub.tex'"<<endl;
    else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile<<"set output 'eqmoms/"<<name<<"_sub.tex'"<<endl;
    scriptfile<<"replot"<<endl;
    
    scriptfile.close();
    
    string command="gnuplot plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt";
    
    system(command.c_str());
    
}

void plot_Zq_chiral_extrapolation(vector<vvd_t> &jZq_equivalent, vector<vXd_t> &jZq_pars, vd_t &m_eff_equivalent_Zq, const string &name, const string &all_or_eq_moms)
{

    int moms=jZq_equivalent.size();
    int njacks=jZq_equivalent[0].size();
    int neq=jZq_equivalent[0][0].size();
     vector<vvd_t> jZq_equivalent_and_chiral_extr(moms,vvd_t(vd_t(neq+1),njacks));
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZq_equivalent_and_chiral_extr[imom][ijack][0]=jZq_pars[imom][ijack](0);
        }
#pragma omp parallel for collapse(3)
    for(int imom=0;imom<moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int ieq=0;ieq<neq;ieq++)
            {
                jZq_equivalent_and_chiral_extr[imom][ijack][ieq+1]=jZq_equivalent[imom][ijack][ieq];
            }


    ///////////DEBUG////////////
    // for(int ijack=0;ijack<njacks;ijack++)
    //   {
    // 	cout<<"JACK: "<<ijack<<"      <<DEBUG>>"<<endl;
    // 	for(int ieq=0;ieq<neq+1;ieq++)
    // 	  {
    // 	    if(ieq==0) cout<<0<<"\t"<<jZq_equivalent_and_chiral_extr[3][ijack][ieq]<<endl;
    // 	    else cout<<m_eff_equivalent_Zq[ieq]*m_eff_equivalent_Zq[ieq]<<"\t"<<jZq_equivalent_and_chiral_extr[3][ijack][ieq]<<endl;
    // 	  }
    // 	cout<<endl;
    //   }



    

    vvvd_t Zq_equivalent = average_Zq(jZq_equivalent_and_chiral_extr);  //Zq[ave/err][imom][ieq]
    vvvd_t Zq_pars=average_pars(jZq_pars);
    
    ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
    
    for(size_t ieq=0;ieq<m_eff_equivalent_Zq.size()+1;ieq++)
      {
	if(ieq==0)
	  datafile1<<0<<"\t"<<Zq_equivalent[0][3][ieq]<<"\t"<<Zq_equivalent[1][3][ieq]<<endl;  //print only for p2~1
        else
	  datafile1<<m_eff_equivalent_Zq[ieq-1]*m_eff_equivalent_Zq[ieq-1]<<"\t"<<Zq_equivalent[0][3][ieq]<<"\t"<<Zq_equivalent[1][3][ieq]<<endl;  //print only for p2~1
      }
    datafile1.close();
    
    double A=Zq_pars[0][3][0];
    double B=Zq_pars[0][3][1];


    ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");
    
    scriptfile<<"set autoscale xy"<<endl;
    scriptfile<<"set xlabel '$M_{PS}^2$'"<<endl;
    if(name=="Sigma1_chiral_extrapolation") scriptfile<<"set ylabel '$Z_q$'"<<endl;
    if(name=="Sigma1_em_chiral_extrapolation") scriptfile<<"set ylabel '$Z_q^{\\rm \\, em}$'"<<endl;
    scriptfile<<"set xrange [-0.003:0.05]"<<endl;
    if(name=="Sigma1_chiral_extrapolation")scriptfile<<"set yrange [0.74:0.80]"<<endl;
    if(name=="Sigma1_em_chiral_extrapolation")scriptfile<<"set yrange [-0.055:-0.01]"<<endl;

    if(name=="Sigma1_chiral_extrapolation")  scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q$'"<<endl;
    if(name=="Sigma1_chiral_extrapolation")  scriptfile<<"replot '< head -1 plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 7 lt 1 lc rgb 'black' title '$Z_q$ chiral extr.'"<<endl;
    if(name=="Sigma1_em_chiral_extrapolation") scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q^{\\rm \\, em}$'"<<endl;
    if(name=="Sigma1_em_chiral_extrapolation") scriptfile<<"replot '< head -1 plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 7 lt 1 lc rgb 'black' title '$Z_q^{\\rm \\, em}$ chiral extr.'"<<endl;
    scriptfile<<"f(x)="<<A<<"+"<<B<<"*x"<<endl;    
    scriptfile<<"replot f(x) lt 2 lc rgb 'red' title 'linear fit'"<<endl;
    scriptfile<<"set terminal epslatex color"<<endl;
    if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile<<"set output 'allmoms/"<<name<<".tex'"<<endl;
    else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile<<"set output 'eqmoms/"<<name<<".tex'"<<endl;
    scriptfile<<"replot"<<endl;
    
    scriptfile.close();
    
    string command="gnuplot plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt";
    
    system(command.c_str());
    
}

void plot_Zq_chiral(vector<vd_t> &jZq_chiral, vector<double> &p2_vector, const string &name, const string &all_or_eq_moms)
{    
    vvd_t Zq_chiral = average_Zq_chiral(jZq_chiral);  //Zq[ave/err][imom]
    
    ///**************************///
    //linear fit
    /* int p2_min=4;  //a2p2~1
    int p2_max=(int)p2_vector.size();
    
    vvd_t coord_linear(vd_t(0.0,p2_vector.size()),2);
    
    for(int i=0; i<p2_vector.size(); i++)
    {
        coord_linear[0][i] = 1.0;  //costante
        coord_linear[1][i] = p2_vector[i];   //p^2
    }
    
    vXd_t jZq_chiral_par=fit_chiral_jackknife(coord_linear,Zq_chiral[1],jZq_chiral,p2_min,p2_max);  //jZq_chiral_par[ijack][par]

    int njacks=jZq_chiral_par.size();
    int pars=jZq_chiral_par[0].size();
    
    vd_t Zq_par_ave(0.0,pars), sqr_Zq_par_ave(0.0,pars), Zq_par_err(0.0,pars);
    vvd_t Zq_par_ave_err(vd_t(0.0,pars),2);
    
    for(int ipar=0;ipar<pars;ipar++)
      for(int ijack=0;ijack<njacks;ijack++)
	{
	  Zq_par_ave[ipar]+=jZq_chiral_par[ijack](ipar)/njacks;
	  sqr_Zq_par_ave[ipar]+=jZq_chiral_par[ijack](ipar)*jZq_chiral_par[ijack](ipar)/njacks;
	}
  
    for(int ipar=0;ipar<pars;ipar++)
      Zq_par_err[ipar]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_par_ave[ipar]-Zq_par_ave[ipar]*Zq_par_ave[ipar]);
    
    Zq_par_ave_err[0]=Zq_par_ave; //Zq_par_ave_err[ave/err][par]
    Zq_par_ave_err[1]=Zq_par_err;  
    
    double A=Zq_par_ave_err[0][0];
    double A_err=Zq_par_ave_err[1][0];
    double B=Zq_par_ave_err[0][1];
    double B_err=Zq_par_ave_err[1][1];

    cout<<A<<" +/- "<<A_err<<endl<<endl; */
    
    ///*****************************///
                    
    
    ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
    
    for(size_t imom=0;imom<p2_vector.size();imom++)
    {
        datafile1<<p2_vector[imom]<<"\t"<<Zq_chiral[0][imom]<<"\t"<<Zq_chiral[1][imom]<<endl;  //print only for M0R0
    }
    datafile1.close();

    /*  ofstream datafile2("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data_fit.txt");
    datafile2<<"0"<<"\t"<<A<<"\t"<<A_err<<endl;
    datafile2.close();    */
    
    ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");
    
    scriptfile<<"set autoscale xy"<<endl;
    scriptfile<<"set xrange [-0.05:2.5]"<<endl;
    if(name=="Sigma1_chiral") scriptfile<<"set yrange [0.73:0.85]"<<endl;
    if(name=="Sigma1_chiral_em_correction") scriptfile<<"set yrange [-0.06:0]"<<endl;
    scriptfile<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;
    if(name=="Sigma1_chiral") scriptfile<<"set ylabel '$Z_q$'"<<endl;
    if(name=="Sigma1_chiral_em_correction") scriptfile<<"set ylabel '$Z_q^{\\rm \\, em}$'"<<endl;
    if(name=="Sigma1_chiral") scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q$ chiral'"<<endl;
    if(name=="Sigma1_chiral_em_correction") scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q^{\\rm \\, em}$ chiral'"<<endl;

    scriptfile<<"set terminal epslatex color"<<endl;
    if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile<<"set output 'allmoms/"<<name<<".tex'"<<endl;
    else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile<<"set output 'eqmoms/"<<name<<".tex'"<<endl;
    scriptfile<<"replot"<<endl;
    
    scriptfile.close();
    
    string command="gnuplot plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt";
    
    system(command.c_str());
    
}


void plot_Zq_RIp_ainv(vector<vd_t> &jZq_chiral, vector<double> &p2_vector, const string &name, const string &all_or_eq_moms)
{    
    vvd_t Zq_chiral = average_Zq_chiral(jZq_chiral);  //Zq[ave/err][imom]
    
    ///**************************///
    //linear fit
    int p2_min=4;  //a2p2~1
    int p2_max=(int)p2_vector.size();
    
    vvd_t coord_linear(vd_t(0.0,p2_vector.size()),2);
    
    for(int i=0; i<p2_vector.size(); i++)
    {
        coord_linear[0][i] = 1.0;  //costante
        coord_linear[1][i] = p2_vector[i];   //p^2
    }
    
    vXd_t jZq_chiral_par=fit_chiral_jackknife(coord_linear,Zq_chiral[1],jZq_chiral,p2_min,p2_max);  //jZq_chiral_par[ijack][par]

    int njacks=jZq_chiral_par.size();
    int pars=jZq_chiral_par[0].size();
    
    vd_t Zq_par_ave(0.0,pars), sqr_Zq_par_ave(0.0,pars), Zq_par_err(0.0,pars);
    vvd_t Zq_par_ave_err(vd_t(0.0,pars),2);
    
    for(int ipar=0;ipar<pars;ipar++)
      for(int ijack=0;ijack<njacks;ijack++)
	{
	  Zq_par_ave[ipar]+=jZq_chiral_par[ijack](ipar)/njacks;
	  sqr_Zq_par_ave[ipar]+=jZq_chiral_par[ijack](ipar)*jZq_chiral_par[ijack](ipar)/njacks;
	}
  
    for(int ipar=0;ipar<pars;ipar++)
      Zq_par_err[ipar]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Zq_par_ave[ipar]-Zq_par_ave[ipar]*Zq_par_ave[ipar]));
    
    Zq_par_ave_err[0]=Zq_par_ave; //Zq_par_ave_err[ave/err][par]
    Zq_par_ave_err[1]=Zq_par_err;  
    
    double A=Zq_par_ave_err[0][0];
    double A_err=Zq_par_ave_err[1][0];
    double B=Zq_par_ave_err[0][1];
    double B_err=Zq_par_ave_err[1][1];

    cout<<A<<" +/- "<<A_err<<endl<<endl; 
    
    ///*****************************///
                    
    
    ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
    
    for(size_t imom=0;imom<p2_vector.size();imom++)
    {
        datafile1<<p2_vector[imom]<<"\t"<<Zq_chiral[0][imom]<<"\t"<<Zq_chiral[1][imom]<<endl;  //print only for M0R0
    }
    datafile1.close();

    ofstream datafile2("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data_fit.txt");
    datafile2<<"0"<<"\t"<<A<<"\t"<<A_err<<endl;
    datafile2.close();    
    
    ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");
    
    scriptfile<<"set autoscale xy"<<endl;
    scriptfile<<"set xrange [-0.05:2.5]"<<endl;
    if(name=="Sigma1_RIp_ainv") scriptfile<<"set yrange [0.74:0.80]"<<endl;
    if(name=="Sigma1_em_RIp_ainv") scriptfile<<"set yrange [-0.07:0.01]"<<endl;
    scriptfile<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;
    if(name=="Sigma1_RIp_ainv") scriptfile<<"set ylabel '$Z_q$'"<<endl;
    if(name=="Sigma1_em_RIp_ainv")  scriptfile<<"set ylabel '$Z_q^{\\rm \\, em}$'"<<endl;
    // if(name=="Sigma1_RIp_ainv") scriptfile<<"set yrange [0.75:0.81]"<<endl;
    // if(name=="Sigma1_em_RIp_ainv") scriptfile<<"set yrange [-0.06:0]"<<endl;
    if(name=="Sigma1_RIp_ainv") scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q$ '"<<endl;
    if(name=="Sigma1_em_RIp_ainv") scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q^{\\rm \\, em}$ '"<<endl;
    scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data_fit.txt' u 1:2:3 with errorbars pt 7 lt 1 lc rgb 'red' ps 1 title 'extrapolation'"<<endl;
    scriptfile<<"f(x)="<<A<<"+"<<B<<"*x"<<endl;
    scriptfile<<"replot f(x) lw 3 title 'linear fit'"<<endl;
    scriptfile<<"set terminal epslatex color"<<endl;
    if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile<<"set output 'allmoms/"<<name<<".tex'"<<endl;
    else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile<<"set output 'eqmoms/"<<name<<".tex'"<<endl;
    scriptfile<<"replot"<<endl;
    
    scriptfile.close();
    
    string command="gnuplot plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt";
    
    system(command.c_str());
    
}


void plot_Z_sub(vector<jZbil_t> &jZ, vector<jZbil_t> &jZ_sub, vector<double> &p2_vector, const string &name, const string &all_or_eq_moms)
{
    vvvvvd_t Z = average_Z(jZ);  //Z[ave/err][imom][mr][mr2][k]
    vvvvvd_t Z_sub = average_Z(jZ_sub);  //Z[ave/err][imom][mr][mr2][k]

    ///////DEBUG
    if(name=="Z1" && all_or_eq_moms=="eqmoms")
      for(int imom=0;imom<p2_vector.size();imom++)
	{
	  cout<<p2_vector[imom]<<"\t"<<jZ[imom][0][0][0][2]<<endl;	    
	}
    cout<<endl;
    /////////
    
    vector<string> bil={"S","A","P","V","T"};
    
    vector<ofstream> datafile(5), datafile_sub(5);
    
    for(int i=0;i<5;i++)
    {
        datafile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_data.txt");
        datafile_sub[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_sub_"+all_or_eq_moms+"_data.txt");
        
        for(size_t imom=0;imom<p2_vector.size();imom++)
        {
            datafile[i]<<p2_vector[imom]<<"\t"<<Z[0][imom][0][0][i]<<"\t"<<Z[1][imom][0][0][i]<<endl;  //print only for M0R0-M0R0
            datafile_sub[i]<<p2_vector[imom]<<"\t"<<Z_sub[0][imom][0][0][i]<<"\t"<<Z_sub[1][imom][0][0][i]<<endl;
        }
        
        datafile[i].close();
        datafile_sub[i].close();
    }
    
    vector<ofstream> scriptfile(5);
    
    
    for(int i=0;i<5;i++)
    {
        scriptfile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_script.txt");
        scriptfile[i]<<"set autoscale xy"<<endl;
        scriptfile[i]<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;
	scriptfile[i]<<"set xrange [0:2.5]"<<endl;
	if(i==0 && name=="Z1")scriptfile[i]<<"set yrange [*:0.82]"<<endl;
	if(i==2 && name=="Z1")scriptfile[i]<<"set yrange [*:0.55]"<<endl;
	
	if(i==0 && name=="Z1_em_correction")scriptfile[i]<<"set yrange [-0.28:-0.04]"<<endl;
	if(i==1 && name=="Z1_em_correction")scriptfile[i]<<"set yrange [-0.12:-0.07]"<<endl;
	if(i==3 && name=="Z1_em_correction")scriptfile[i]<<"set yrange [-0.17:-0.11]"<<endl;
	
	// scriptfile[i]<<"set yrange [0.7:0.9]"<<endl;
        scriptfile[i]<<"set ylabel '$Z_"<<bil[i]<<"$'"<<endl;
        scriptfile[i]<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_"<<bil[i]<<"$'"<<endl;
        scriptfile[i]<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_sub_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lt 1 lc rgb 'red' title '$Z_"<<bil[i]<<"^{\\mathrm{corr.}}$'"<<endl;
        scriptfile[i]<<"set terminal epslatex color"<<endl;
        if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile[i]<<"set output 'allmoms/"<<name<<"_"<<bil[i]<<"_sub.tex'"<<endl;
        else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile[i]<<"set output 'eqmoms/"<<name<<"_"<<bil[i]<<"_sub.tex'"<<endl;
        scriptfile[i]<<"replot"<<endl;
        scriptfile[i]<<"set term unknown"<<endl;
        
        scriptfile[i].close();
        
        string command="gnuplot plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_script.txt";
        
        system(command.c_str());
    }
    
}

void plot_ZPandS_chiral_extrapolation(const string &bil, vector<vvd_t> &jZ_equivalent, vector<vvd_t> &jG_subpole, vector<vXd_t> &jZ_pars, vd_t &m_eff_equivalent_Z, const string &name, const string &all_or_eq_moms)
{
  int moms=jZ_equivalent.size();
  int njacks=jZ_equivalent[0].size();
  int neq=jZ_equivalent[0][0].size();
  // vector<vvd_t> jZ_equivalent_and_chiral_extr(moms,vvd_t(vd_t(neq+1),njacks));
    
  /*#pragma omp parallel for collapse(2)
  for(int imom=0;imom<moms;imom++)
    for(int ijack=0;ijack<njacks;ijack++)
      {
	jZ_equivalent_and_chiral_extr[imom][ijack][0]=jZ_pars[imom][ijack](0);
      }
#pragma omp parallel for collapse(3)
  for(int imom=0;imom<moms;imom++)
    for(int ijack=0;ijack<njacks;ijack++)
      for(int ieq=0;ieq<neq;ieq++)
	{
	  jZ_equivalent_and_chiral_extr[imom][ijack][ieq+1]=jZ_equivalent[imom][ijack][ieq];
	  }*/
    
  vvvd_t Z_equivalent = average_Zq(jZ_equivalent);  //Z[ave/err][imom][ieq]
  vvvd_t Z_pars=average_pars(jZ_pars);
  vvvd_t G_subpole = average_Zq(jG_subpole);
    
  ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
  ofstream datafile3("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data_fit.txt");
    
  //datafile1<<0<<"\t"<<Z_pars[0][0]<<"\t"<<Z_pars[1][0]<<endl;

  datafile3<<0<<"\t"<<Z_pars[0][3][0]<<"\t"<<Z_pars[1][3][0]<<endl;  //print only for p2~1
  
  for(size_t ieq=0;ieq<m_eff_equivalent_Z.size();ieq++)
    {
      datafile1<<m_eff_equivalent_Z[ieq]*m_eff_equivalent_Z[ieq]<<"\t"<<Z_equivalent[0][3][ieq]<<"\t"<<Z_equivalent[1][3][ieq]<<endl;  //print only for p2~1
    }
  datafile1.close();
  datafile3.close();

  ofstream datafile2("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data_subpole.txt");
    
  //datafile1<<0<<"\t"<<Z_pars[0][0]<<"\t"<<Z_pars[1][0]<<endl;
  for(size_t ieq=0;ieq<m_eff_equivalent_Z.size()/*+1*/;ieq++)
    {
      datafile2<<m_eff_equivalent_Z[ieq]*m_eff_equivalent_Z[ieq]<<"\t"<<G_subpole[0][3][ieq]<<"\t"<<G_subpole[1][3][ieq]<<endl;  //print only for p2~1
    }
  datafile2.close();
    
  double A=Z_pars[0][3][0];
  double B=Z_pars[0][3][1];
  double C=0;
  if(Z_pars[0][3].size()==3)
    C=Z_pars[0][3][2];
    
  ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");
    
  scriptfile<<"set autoscale xy"<<endl;
  scriptfile<<"set xlabel '$M_{PS}^2$'"<<endl;
  if(name=="Gp_chiral_extrapolation") scriptfile<<"set ylabel '$\\Gamma_"<<bil<<"$'"<<endl;
  if(name=="Gs_chiral_extrapolation") scriptfile<<"set ylabel '$\\Gamma_"<<bil<<"$'"<<endl;
  if(name=="Gp_em_chiral_extrapolation") scriptfile<<"set ylabel '$\\delta \\Gamma_"<<bil<<"$'"<<endl;
  if(name=="Gs_em_chiral_extrapolation") scriptfile<<"set ylabel '$\\delta \\Gamma_"<<bil<<"$'"<<endl;
  scriptfile<<"set xrange [-0.003:0.05]"<<endl;
  if(name=="Gp_chiral_extrapolation") scriptfile<<"set yrange [0:5]"<<endl;
  if(name=="Gs_chiral_extrapolation") scriptfile<<"set yrange [0.9:1.6]"<<endl;
  if(name=="Gp_em_chiral_extrapolation") scriptfile<<"set yrange [-1.2:0.6]"<<endl;
  if(name=="Gs_em_chiral_extrapolation") scriptfile<<"set yrange [-0.2:0.2]"<<endl;
  if(name=="Gp_chiral_extrapolation"||name=="Gs_chiral_extrapolation")
    {
      scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$\\Gamma_"<<bil<<"$'"<<endl;
      scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data_subpole.txt' u 1:2:3 with errorbars pt 7 lt 1 lc rgb 'blue' title '$\\Gamma_"<<bil<<"^{sub}$'"<<endl;
      // scriptfile<<"replot '< head -1 plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 5 lt 1 lc rgb 'black' title '$\\Gamma_"<<bil<<"$ chiral extr.'"<<endl;
      scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data_fit.txt' u 1:2:3 with errorbars pt 5 lt 1 lc rgb 'black' title '$\\Gamma_"<<bil<<"$ chiral extr.'"<<endl;
    }
  if(name=="Gp_em_chiral_extrapolation"||name=="Gs_em_chiral_extrapolation")
    {
      scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$\\delta \\Gamma_"<<bil<<"$'"<<endl;
      scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data_subpole.txt' u 1:2:3 with errorbars pt 7 lt 1 lc rgb 'blue' title '$\\delta \\Gamma_"<<bil<<"^{sub}$'"<<endl;
      // scriptfile<<"replot '< head -1 plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 5 lt 1 lc rgb 'black' title '$\\Gamma_"<<bil<<"$ chiral extr.'"<<endl;
      scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data_fit.txt' u 1:2:3 with errorbars pt 5 lt 1 lc rgb 'black' title '$\\delta \\Gamma_"<<bil<<"$ chiral extr.'"<<endl;
    }
  if(Z_pars[0][3].size()==2)
    scriptfile<<"f(x)="<<A<<"+"<<B<<"*x"<<endl;
  if(Z_pars[0][3].size()==3)
    scriptfile<<"f(x)=(x > 0) ? "<<A<<"+"<<B<<"*x"<<"+"<<C<<"/x : 1/0"<<endl;
  scriptfile<<"replot f(x) lt 1 lc rgb 'blue' title 'fit curve'"<<endl;
  scriptfile<<"g(x)="<<A<<"+"<<B<<"*x"<<endl;
  scriptfile<<"replot g(x) lt 2 lc rgb 'red' title 'linear fit'"<<endl;
  scriptfile<<"set terminal epslatex color"<<endl;
  if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile<<"set output 'allmoms/"<<name<<".tex'"<<endl;
  else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile<<"set output 'eqmoms/"<<name<<".tex'"<<endl;
  scriptfile<<"replot"<<endl;
    
  scriptfile.close();
    
  string command="gnuplot plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt";
    
  system(command.c_str());
    
}

void plot_ZVAT_chiral_extrapolation(const string &bil, vector<vvd_t> &jZ_equivalent, vector<vXd_t> &jZ_pars, vd_t &m_eff_equivalent_Z, const string &name, const string &all_or_eq_moms)
{
    int moms=jZ_equivalent.size();
    int njacks=jZ_equivalent[0].size();
    int neq=jZ_equivalent[0][0].size();
    vector<vvd_t> jZ_equivalent_and_chiral_extr(moms,vvd_t(vd_t(neq+1),njacks));
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZ_equivalent_and_chiral_extr[imom][ijack][0]=jZ_pars[imom][ijack](0);
        }
#pragma omp parallel for collapse(3)
    for(int imom=0;imom<moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int ieq=0;ieq<neq;ieq++)
            {
                jZ_equivalent_and_chiral_extr[imom][ijack][ieq+1]=jZ_equivalent[imom][ijack][ieq];
            }
    
    vvvd_t Z_equivalent = average_Zq(jZ_equivalent_and_chiral_extr);  //Z[ave/err][imom][ieq]
    vvvd_t Z_pars=average_pars(jZ_pars);
 
    ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
    
    //datafile1<<0<<"\t"<<Z_pars[0][0]<<"\t"<<Z_pars[1][0]<<endl;
    for(size_t ieq=0;ieq<m_eff_equivalent_Z.size()+1;ieq++)
    {
        if(ieq==0)
            datafile1<<0<<"\t"<<Z_equivalent[0][3][ieq]<<"\t"<<Z_equivalent[1][3][ieq]<<endl;  //print only for p2~1
        else
            datafile1<<m_eff_equivalent_Z[ieq-1]*m_eff_equivalent_Z[ieq-1]<<"\t"<<Z_equivalent[0][3][ieq]<<"\t"<<Z_equivalent[1][3][ieq]<<endl;  //print only for p2~1
    }
    datafile1.close();
    
    double A=Z_pars[0][3][0];
    double B=Z_pars[0][3][1];
  
    ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");
    
    scriptfile<<"set autoscale xy"<<endl;
    scriptfile<<"set xlabel '$M_{PS}^2$'"<<endl;
    if(name=="Gv_chiral_extrapolation") scriptfile<<"set ylabel '$\\Gamma_"<<bil<<"$'"<<endl;
    if(name=="Ga_chiral_extrapolation") scriptfile<<"set ylabel '$\\Gamma_"<<bil<<"$'"<<endl;
    if(name=="Gt_chiral_extrapolation") scriptfile<<"set ylabel '$\\Gamma_"<<bil<<"$'"<<endl;
    if(name=="Gv_em_chiral_extrapolation") scriptfile<<"set ylabel '$\\delta\\Gamma_"<<bil<<"$'"<<endl;
    if(name=="Ga_em_chiral_extrapolation") scriptfile<<"set ylabel '$\\delta\\Gamma_"<<bil<<"$'"<<endl;
    if(name=="Gt_em_chiral_extrapolation") scriptfile<<"set ylabel '$\\delta\\Gamma_"<<bil<<"$'"<<endl;
    scriptfile<<"set xrange [-0.003:0.05]"<<endl;
    if(name=="Gv_chiral_extrapolation") scriptfile<<"set yrange [0.98:1.03]"<<endl;
    if(name=="Ga_chiral_extrapolation") scriptfile<<"set yrange [1.15:1.23]"<<endl;
    if(name=="Gt_chiral_extrapolation") scriptfile<<"set yrange [0.99:1.03]"<<endl;
    if(name=="Gv_em_chiral_extrapolation") scriptfile<<"set yrange [-0.044:0.026]"<<endl;
    if(name=="Ga_em_chiral_extrapolation") scriptfile<<"set yrange [-0.12:-0.06]"<<endl;
    if(name=="Gt_em_chiral_extrapolation") scriptfile<<"set yrange [-0.046:-0.006]"<<endl;
    if(name=="Gv_chiral_extrapolation"||name=="Ga_chiral_extrapolation"||name=="Gt_chiral_extrapolation")
      {
	scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$\\Gamma_"<<bil<<"$'"<<endl;
	scriptfile<<"replot '< head -1 plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 5 lt 1 lc rgb 'black' title '$\\Gamma_"<<bil<<"$ chiral extr.'"<<endl;
      }
     if(name=="Gv_em_chiral_extrapolation"||name=="Ga_em_chiral_extrapolation"||name=="Gt_em_chiral_extrapolation")
      {
	scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$\\delta \\Gamma_"<<bil<<"$'"<<endl;
	scriptfile<<"replot '< head -1 plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 5 lt 1 lc rgb 'black' title '$\\delta \\Gamma_"<<bil<<"$ chiral extr.'"<<endl;
      }
    scriptfile<<"f(x)="<<A<<"+"<<B<<"*x"<<endl;
    scriptfile<<"replot f(x) lt 2 lc rgb 'red' title 'linear fit'"<<endl;
    scriptfile<<"set terminal epslatex color"<<endl;
    if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile<<"set output 'allmoms/"<<name<<".tex'"<<endl;
    else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile<<"set output 'eqmoms/"<<name<<".tex'"<<endl;
    scriptfile<<"replot"<<endl;
    
    scriptfile.close();
    
    string command="gnuplot plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt";
    
    system(command.c_str());
    
}


void plot_Z_chiral(vector<vvd_t> &jZ_chiral, vector<double> &p2_vector, const string &name, const string &all_or_eq_moms)
{
  //  cout<<"DEBUG---(A)"<<endl;
  vvvd_t Z_chiral = average_Z_chiral(jZ_chiral);  //Z_chiral[ave/err][imom][k]

  // cout<<"DEBUG---(B)"<<endl;
    
  ///**************************///
  //linear fit
  int p2_min=5;  //a2p2~1
  int p2_max=(int)p2_vector.size();
    
  vvd_t coord_linear(vd_t(0.0,p2_vector.size()),2);
    
  for(int i=0; i<p2_vector.size(); i++)
    {
      coord_linear[0][i] = 1.0;  //costante
      coord_linear[1][i] = p2_vector[i];   //p^2
      }

  ///************************///

  // cout<<"DEBUG---(C)"<<endl;
    
   valarray<vXd_t> jZ_chiral_par=fit_chiral_Z_jackknife(coord_linear,Z_chiral[1],jZ_chiral,p2_min,p2_max);  //jZ_chiral_par[ibil][ijack][ipar]

  // cout<<"DEBUG---(D)"<<endl;

   int nbil=jZ_chiral_par.size();
  int njacks=jZ_chiral_par[0].size();
  int pars=jZ_chiral_par[0][0].size();
    
  vvd_t Z_par_ave(vd_t(0.0,pars),nbil), sqr_Z_par_ave(vd_t(0.0,pars),nbil), Z_par_err(vd_t(0.0,pars),nbil); //Z
  vvvd_t Z_par_ave_err(vvd_t(vd_t(0.0,pars),nbil),2);  //Zq_par_ave_err[ave/err][ibil][par]

  // cout<<"DEBUG---(E)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ipar=0;ipar<pars;ipar++)
      for(int ijack=0;ijack<njacks;ijack++)
	{
	  Z_par_ave[ibil][ipar]+=jZ_chiral_par[ibil][ijack](ipar)/njacks;
	  sqr_Z_par_ave[ibil][ipar]+=jZ_chiral_par[ibil][ijack](ipar)*jZ_chiral_par[ibil][ijack](ipar)/njacks;
	}

  //  cout<<"DEBUG---(F)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ipar=0;ipar<pars;ipar++)
      Z_par_err[ibil][ipar]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_par_ave[ibil][ipar]-Z_par_ave[ibil][ipar]*Z_par_ave[ibil][ipar]));

  // cout<<"DEBUG---(G)"<<endl;
    
  Z_par_ave_err[0]=Z_par_ave; //Z_par_ave_err[ave/err][ibil][par]
  Z_par_ave_err[1]=Z_par_err;

  // cout<<"DEBUG---(H)"<<endl;

  vd_t A(0.0,nbil),A_err(0.0,nbil),B(0.0,nbil),B_err(0.0,nbil);

  for(int ibil=0; ibil<nbil;ibil++)
    {
      A[ibil]=Z_par_ave_err[0][ibil][0];
      A_err[ibil]=Z_par_ave_err[1][ibil][0];
      B[ibil]=Z_par_ave_err[0][ibil][1];
      B_err[ibil]=Z_par_ave_err[1][ibil][1]; 
      }
    
  ///*****************************///

  //////////////////////
    
  vector<string> bil={"S","A","P","V","T"};
    
  vector<ofstream> datafile(5);
// vector<ofstream> datafile_fit(5);
    
  for(int i=0;i<5;i++)
    {
      //   cout<<endl;
      //   cout<<"Z"<<bil[i]<<" = "<<A[i]<<" +/- "<<A_err[i]<<endl; 
      
      datafile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_data.txt");
      
      for(size_t imom=0;imom<p2_vector.size();imom++)
	datafile[i]<<p2_vector[imom]<<"\t"<<Z_chiral[0][imom][i]<<"\t"<<Z_chiral[1][imom][i]<<endl;
      
      datafile[i].close();
      
      //   datafile_fit[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_data_fit.txt");
      // datafile_fit[i]<<"0"<<"\t"<<A[i]<<"\t"<<A_err[i]<<endl;
      //datafile_fit[i].close();
    }
  cout<<endl;
    
  vector<ofstream> scriptfile(5);
    
    
  for(int i=0;i<5;i++)
    {
      scriptfile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_script.txt");
      scriptfile[i]<<"set autoscale xy"<<endl;
      scriptfile[i]<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;

      //bil={S,A,P,V,T}
      
      scriptfile[i]<<"set xrange [-0.05:2.3]"<<endl;
      if(name=="Z1_chiral"&&i==0)  scriptfile[i]<<"set yrange [0.45:0.85]"<<endl;
      if(name=="Z1_chiral"&&i==1)  scriptfile[i]<<"set yrange [0.74:0.82]"<<endl;
      if(name=="Z1_chiral"&&i==2)  scriptfile[i]<<"set yrange [0.35:0.7]"<<endl;
      if(name=="Z1_chiral"&&i==3)  scriptfile[i]<<"set yrange [0.62:0.69]"<<endl;
      if(name=="Z1_chiral"&&i==4)  scriptfile[i]<<"set yrange [0.65:0.95]"<<endl;
      if(name=="Z1_chiral_em_correction"&&i==0)  scriptfile[i]<<"set yrange [-0.15:0.10]"<<endl;
      if(name=="Z1_chiral_em_correction"&&i==1)  scriptfile[i]<<"set yrange [-0.1:-0.01]"<<endl;
      if(name=="Z1_chiral_em_correction"&&i==2)  scriptfile[i]<<"set yrange [-0.7:0.2]"<<endl;
      if(name=="Z1_chiral_em_correction"&&i==3)  scriptfile[i]<<"set yrange [-0.16:-0.07]"<<endl;
      if(name=="Z1_chiral_em_correction"&&i==4)  scriptfile[i]<<"set yrange [-0.13:0.01]"<<endl;
      
     if(name=="Z1_chiral") scriptfile[i]<<"set ylabel '$Z_"<<bil[i]<<"$'"<<endl;
     if(name=="Z1_chiral_em_correction") scriptfile[i]<<"set ylabel '$\\delta Z_"<<bil[i]<<"$'"<<endl;
     
     if(name=="Z1_chiral") scriptfile[i]<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_"<<bil[i]<<"$ chiral'"<<endl;
     if(name=="Z1_chiral_em_correction")  scriptfile[i]<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$\\delta Z_"<<bil[i]<<"$ chiral'"<<endl;
     
      scriptfile[i]<<"set terminal epslatex color"<<endl;
      if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile[i]<<"set output 'allmoms/"<<name<<"_"<<bil[i]<<".tex'"<<endl;
      else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile[i]<<"set output 'eqmoms/"<<name<<"_"<<bil[i]<<".tex'"<<endl;
      scriptfile[i]<<"replot"<<endl;
      scriptfile[i]<<"set term unknown"<<endl;
        
      scriptfile[i].close();
        
      string command="gnuplot plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_script.txt";
        
      system(command.c_str());
    }
    
  int moms=p2_vector.size();
  // int njacks=jZ_chiral[0].size();
  vector<vd_t> jZP_over_S(moms,vd_t(0.0,njacks));
    
#pragma omp parallel for collapse(2)
  for(int imom=0;imom<moms;imom++)
    for(int ijack=0;ijack<njacks;ijack++)
      jZP_over_S[imom][ijack]=jZ_chiral[imom][ijack][2]/jZ_chiral[imom][ijack][0];
    
  vvd_t ZP_over_S=average_Zq_chiral(jZP_over_S);
    
  ofstream datafile2;
    
    
  datafile2.open("plot_data_and_script/plot_"+name+"_P_over_S_"+all_or_eq_moms+"_data.txt");
    
  for(size_t imom=0;imom<p2_vector.size();imom++)
    datafile2<<p2_vector[imom]<<"\t"<<ZP_over_S[0][imom]<<"\t"<<ZP_over_S[1][imom]<<endl;
    
  datafile2.close();
    
    
  ofstream scriptfile2;
    
  scriptfile2.open("plot_data_and_script/plot_"+name+"_P_over_S_"+all_or_eq_moms+"_script.txt");
  scriptfile2<<"set autoscale xy"<<endl;
  scriptfile2<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;
  // scriptfile2<<"set yrange [0.7:0.9]"<<endl;
  scriptfile2<<"set ylabel '$Z_P/Z_S$'"<<endl;
  scriptfile2<<"plot 'plot_data_and_script/plot_"<<name<<"_P_over_S_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_P/Z_S$ chiral'"<<endl;
  scriptfile2<<"set terminal epslatex color"<<endl;
  if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile2<<"set output 'allmoms/"<<name<<"_P_over_S.tex'"<<endl;
  else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile2<<"set output 'eqmoms/"<<name<<"_P_over_S.tex'"<<endl;
  scriptfile2<<"replot"<<endl;
  scriptfile2<<"set term unknown"<<endl;
    
  scriptfile2.close();
    
  string command2="gnuplot plot_data_and_script/plot_"+name+"_P_over_S_"+all_or_eq_moms+"_script.txt";
    
  system(command2.c_str());
    
}


void plot_ZO_RIp_ainv(vector<vvd_t> &jZ_chiral, vector<double> &p2_vector, const string &name, const string &all_or_eq_moms, const double &p_min_value)
{
  //  cout<<"DEBUG---(A)"<<endl;
  vvvd_t Z_chiral = average_Z_chiral(jZ_chiral);  //Z_chiral[ave/err][imom][k]

  // cout<<"DEBUG---(B)"<<endl;
    
  ///**************************///
  //linear fit
  // int p2_min=5;  //a2p2~1
  int p2_min=0;
  int p2_max=(int)p2_vector.size();
    
  vvd_t coord_linear(vd_t(0.0,p2_vector.size()),2);
    
  for(int i=0; i<p2_vector.size(); i++)
    {
      coord_linear[0][i] = 1.0;  //costante
      coord_linear[1][i] = p2_vector[i];   //p^2
    }

  ///************************///

  // cout<<"DEBUG---(C)"<<endl;
    
  valarray<vXd_t> jZ_chiral_par=fit_chiral_Z_RIp_jackknife(coord_linear,Z_chiral[1],jZ_chiral,p2_min,p2_max,p_min_value);  //jZ_chiral_par[ibil][ijack][ipar]

  // cout<<"DEBUG---(D)"<<endl;

  int nbil=jZ_chiral_par.size();
  int njacks=jZ_chiral_par[0].size();
  int pars=jZ_chiral_par[0][0].size();
    
  vvd_t Z_par_ave(vd_t(0.0,pars),nbil), sqr_Z_par_ave(vd_t(0.0,pars),nbil), Z_par_err(vd_t(0.0,pars),nbil); //Z
  vvvd_t Z_par_ave_err(vvd_t(vd_t(0.0,pars),nbil),2);  //Zq_par_ave_err[ave/err][ibil][par]

  // cout<<"DEBUG---(E)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ipar=0;ipar<pars;ipar++)
      for(int ijack=0;ijack<njacks;ijack++)
	{
	  Z_par_ave[ibil][ipar]+=jZ_chiral_par[ibil][ijack](ipar)/njacks;
	  sqr_Z_par_ave[ibil][ipar]+=jZ_chiral_par[ibil][ijack](ipar)*jZ_chiral_par[ibil][ijack](ipar)/njacks;
	}

  //  cout<<"DEBUG---(F)"<<endl;

  for(int ibil=0; ibil<nbil;ibil++)
    for(int ipar=0;ipar<pars;ipar++)
      Z_par_err[ibil][ipar]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_par_ave[ibil][ipar]-Z_par_ave[ibil][ipar]*Z_par_ave[ibil][ipar]));  //ibil={S,A,P,V,T}

  // cout<<"DEBUG---(G)"<<endl;
    
  Z_par_ave_err[0]=Z_par_ave; //Z_par_ave_err[ave/err][ibil][par]
  Z_par_ave_err[1]=Z_par_err;

  // cout<<"DEBUG---(H)"<<endl;

  vd_t A(0.0,nbil),A_err(0.0,nbil),B(0.0,nbil),B_err(0.0,nbil);

  for(int ibil=0; ibil<nbil;ibil++)
    {
      A[ibil]=Z_par_ave_err[0][ibil][0];
      A_err[ibil]=Z_par_ave_err[1][ibil][0];
      B[ibil]=Z_par_ave_err[0][ibil][1];
      B_err[ibil]=Z_par_ave_err[1][ibil][1];
    }
    
  ///*****************************///

  //////////////////////
    
  vector<string> bil={"S","A","P","V","T"};
    
  vector<ofstream> datafile(5);
  vector<ofstream> datafile_fit(5);
    
  for(int i=0;i<5;i++)
    {
      //   cout<<endl;
      cout<<"Z"<<bil[i]<<" = "<<A[i]<<" +/- "<<A_err[i]<<endl; 
      
      datafile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_data.txt");
      
      for(size_t imom=0;imom<p2_vector.size();imom++)
	datafile[i]<<p2_vector[imom]<<"\t"<<Z_chiral[0][imom][i]<<"\t"<<Z_chiral[1][imom][i]<<endl;
      
      datafile[i].close();
      
      datafile_fit[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_data_fit.txt");
      datafile_fit[i]<<"0"<<"\t"<<A[i]<<"\t"<<A_err[i]<<endl;
      datafile_fit[i].close();
    }
  cout<<endl;
    
  vector<ofstream> scriptfile(5);
    
    
  for(int i=0;i<5;i++)
    {
      scriptfile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_script.txt");
      scriptfile[i]<<"set autoscale xy"<<endl;
      scriptfile[i]<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;
      if(i==0 && name=="ZO_RIp_ainv") scriptfile[i]<<"set yrange [0.55:0.78]"<<endl; //S
      if(i==1 && name=="ZO_RIp_ainv") scriptfile[i]<<"set yrange [0.73:0.83]"<<endl; //A
      if(i==2 && name=="ZO_RIp_ainv") scriptfile[i]<<"set yrange [0.2:1]"<<endl; //P
      if(i==3 && name=="ZO_RIp_ainv") scriptfile[i]<<"set yrange [0.6:0.7]"<<endl; //V
      if(i==4 && name=="ZO_RIp_ainv") scriptfile[i]<<"set yrange [0.70:0.85]"<<endl; //T
      if(i==0 && name=="ZO_em_RIp_ainv") scriptfile[i]<<"set yrange [-0.3:0.3]"<<endl; //S
      if(i==1 && name=="ZO_em_RIp_ainv") scriptfile[i]<<"set yrange [-0.09:0]"<<endl; //A
      if(i==2 && name=="ZO_em_RIp_ainv") scriptfile[i]<<"set yrange [-1:0.6]"<<endl; //P
      if(i==3 && name=="ZO_em_RIp_ainv") scriptfile[i]<<"set yrange [-0.16:-0.05]"<<endl; //V
      if(i==4 && name=="ZO_em_RIp_ainv") scriptfile[i]<<"set yrange [-0.125:0.025]"<<endl; //T
      scriptfile[i]<<"set xrange [-0.05:2.3]"<<endl;
      if(name=="ZO_RIp_ainv") scriptfile[i]<<"set ylabel '$Z_"<<bil[i]<<"$'"<<endl;
      if(name=="ZO_em_RIp_ainv") scriptfile[i]<<"set ylabel '$\\delta Z_"<<bil[i]<<"$'"<<endl;
      if(name=="ZO_RIp_ainv") scriptfile[i]<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_"<<bil[i]<<"$'"<<endl;
      if(name=="ZO_em_RIp_ainv") scriptfile[i]<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$\\delta Z_"<<bil[i]<<"$'"<<endl;
      scriptfile[i]<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data_fit.txt' u 1:2:3 with errorbars pt 7 lt 1 lc rgb 'red' ps 1 title 'extrapolation'"<<endl;
      scriptfile[i]<<"f(x)="<<A[i]<<"+"<<B[i]<<"*x"<<endl;
      scriptfile[i]<<"replot f(x) lw 3 title 'linear fit'"<<endl;
      scriptfile[i]<<"set terminal epslatex color"<<endl;
      if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile[i]<<"set output 'allmoms/"<<name<<"_"<<bil[i]<<".tex'"<<endl;
      else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile[i]<<"set output 'eqmoms/"<<name<<"_"<<bil[i]<<".tex'"<<endl;
      scriptfile[i]<<"replot"<<endl;
      scriptfile[i]<<"set term unknown"<<endl;
        
      scriptfile[i].close();
        
      string command="gnuplot plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_script.txt";
        
      system(command.c_str());
    }
    
  int moms=p2_vector.size();
  // int njacks=jZ_chiral[0].size();
  vector<vd_t> jZP_over_S(moms,vd_t(0.0,njacks));
    
#pragma omp parallel for collapse(2)
  for(int imom=0;imom<moms;imom++)
    for(int ijack=0;ijack<njacks;ijack++)
      jZP_over_S[imom][ijack]=jZ_chiral[imom][ijack][2]/jZ_chiral[imom][ijack][0];
    
  vvd_t ZP_over_S=average_Zq_chiral(jZP_over_S);
    
  ofstream datafile2;
    
    
  datafile2.open("plot_data_and_script/plot_"+name+"_P_over_S_"+all_or_eq_moms+"_data.txt");
    
  for(size_t imom=0;imom<p2_vector.size();imom++)
    datafile2<<p2_vector[imom]<<"\t"<<ZP_over_S[0][imom]<<"\t"<<ZP_over_S[1][imom]<<endl;
    
  datafile2.close();
    
    
  ofstream scriptfile2;
    
  scriptfile2.open("plot_data_and_script/plot_"+name+"_P_over_S_"+all_or_eq_moms+"_script.txt");
  scriptfile2<<"set autoscale xy"<<endl;
  scriptfile2<<"set xlabel '$a^2\\tilde{p}^2$'"<<endl;
  // scriptfile2<<"set yrange [0.7:0.9]"<<endl;
  scriptfile2<<"set ylabel '$Z_P/Z_S$'"<<endl;
  scriptfile2<<"plot 'plot_data_and_script/plot_"<<name<<"_P_over_S_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_P/Z_S$ chiral'"<<endl;
  scriptfile2<<"set terminal epslatex color"<<endl;
  if(strcmp(all_or_eq_moms.c_str(),"allmoms")==0) scriptfile2<<"set output 'allmoms/"<<name<<"_P_over_S.tex'"<<endl;
  else if(strcmp(all_or_eq_moms.c_str(),"eqmoms")==0) scriptfile2<<"set output 'eqmoms/"<<name<<"_P_over_S.tex'"<<endl;
  scriptfile2<<"replot"<<endl;
  scriptfile2<<"set term unknown"<<endl;
    
  scriptfile2.close();
    
  string command2="gnuplot plot_data_and_script/plot_"+name+"_P_over_S_"+all_or_eq_moms+"_script.txt";
    
  system(command2.c_str());
    
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

    vector<vd_t> jSigma1_RIp_ainv_allmoms(moms,vd_t(0.0,njacks)),jSigma1_em_RIp_ainv_allmoms(moms,vd_t(0.0,njacks));
    vector<vvd_t> jZO_RIp_ainv_allmoms(moms,vvd_t(vd_t(5),njacks)),jZO_em_RIp_ainv_allmoms(moms,vvd_t(vd_t(5),njacks));
    
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
    
    vector<vvd_t> jZq_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks)), jSigma1_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks)),\
    jZq_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks)), jSigma1_em_equivalent_eqmoms(neq_moms,vvd_t(vd_t(neq2),njacks));
    
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

    vector<vd_t> jSigma1_RIp_ainv_eqmoms(neq_moms,vd_t(0.0,njacks)),jSigma1_em_RIp_ainv_eqmoms(neq_moms,vd_t(0.0,njacks));
    vector<vvd_t> jZO_RIp_ainv_eqmoms(neq_moms,vvd_t(vd_t(5),njacks)),jZO_em_RIp_ainv_eqmoms(neq_moms,vvd_t(vd_t(5),njacks));
 

    
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
    READ(jSigma1_RIp_ainv);
    READ(jSigma1_em_RIp_ainv);
    READ(jZO_RIp_ainv);
    READ(jZO_em_RIp_ainv);
    
#undef READ
    
    read_vec(m_eff_equivalent,"allmoms/m_eff_equivalent");
    read_vec(m_eff_equivalent_Zq,"allmoms/m_eff_equivalent_Zq");
    
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zq with subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
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
    
    plot_Zq_sub(jZq_em_eqmoms,jZq_em_sub_eqmoms,p2_vector_eqmoms,"Zq_em_correction","eqmoms");
    plot_Zq_sub(jSigma1_em_eqmoms,jSigma1_em_sub_eqmoms,p2_vector_eqmoms,"Sigma1_em_correction","eqmoms");


    vvvd_t Sigma1 = average_Zq(jSigma1_em_eqmoms);  //Zq[ave/err][imom][nm]
    vvvd_t Sigma1_sub = average_Zq(jSigma1_em_sub_eqmoms);  //Zq[ave/err][imom][nm]

    cout<<"Zq"<<endl<<"------------------------------"<<endl<<endl;
    for(size_t imom=0;imom<p2_vector_eqmoms.size();imom++)
    {
        cout<<p2_vector_eqmoms[imom]<<"\t"<<Sigma1[0][imom][0]<<"\t"<<Sigma1[1][imom][0]<<endl;  //print only for M0R0
    }
    cout<<endl<<"Zq_SUB"<<endl<<"------------------------------"<<endl;
    for(size_t imom=0;imom<p2_vector_eqmoms.size();imom++)
    {
       cout<<p2_vector_eqmoms[imom]<<"\t"<<Sigma1_sub[0][imom][0]<<"\t"<<Sigma1_sub[1][imom][0]<<endl;  //print only for M0R0
    }




    
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zq chiral extrapolation  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    // plot_Zq_chiral_extrapolation(jZq_equivalent_eqmoms,jZq_pars_eqmoms,m_eff_equivalent_Zq,"Zq_chiral_extrapolation","eqmoms");

    plot_Zq_chiral_extrapolation(jSigma1_equivalent_eqmoms,jSigma1_pars_eqmoms,m_eff_equivalent_Zq,"Sigma1_chiral_extrapolation","eqmoms");
    plot_Zq_chiral_extrapolation(jSigma1_em_equivalent_eqmoms,jSigma1_em_pars_eqmoms,m_eff_equivalent_Zq,"Sigma1_em_chiral_extrapolation","eqmoms");
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zq chiral ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    // cout<<"Zq chiral"<<endl;
    //plot_Zq_chiral(jZq_chiral_eqmoms,p2_vector_eqmoms,"Zq_chiral","eqmoms");
    cout<<"Sigma1 chiral"<<endl;
    plot_Zq_chiral(jSigma1_chiral_eqmoms,p2_vector_eqmoms,"Sigma1_chiral","eqmoms");
    
    vector<vd_t> jZq_chiral_with_em_eqmoms(neq_moms,vd_t(njacks)), jSigma1_chiral_with_em_eqmoms(neq_moms,vd_t(njacks));
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<neq_moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZq_chiral_with_em_eqmoms[imom][ijack]=jZq_chiral_eqmoms[imom][ijack]+jZq_em_chiral_eqmoms[imom][ijack];
            jSigma1_chiral_with_em_eqmoms[imom][ijack]=jSigma1_chiral_eqmoms[imom][ijack]+jSigma1_em_chiral_eqmoms[imom][ijack];
        }
    
    //cout<<"Zq chiral with em"<<endl;
    // plot_Zq_chiral(jZq_chiral_with_em_eqmoms,p2_vector_eqmoms,"Zq_chiral_with_em","eqmoms");
    //cout<<"Sigma1 chiral with em"<<endl;
    //plot_Zq_chiral(jSigma1_chiral_with_em_eqmoms,p2_vector_eqmoms,"Sigma1_chiral_with_em","eqmoms");

    // cout<<"Zq chiral em correction"<<endl;
    // plot_Zq_chiral(jZq_em_chiral_eqmoms,p2_vector_eqmoms,"Zq_chiral_em_correction","eqmoms");
    cout<<"Sigma1 chiral em correction"<<endl;
    plot_Zq_chiral(jSigma1_em_chiral_eqmoms,p2_vector_eqmoms,"Sigma1_chiral_em_correction","eqmoms");
    
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zq over Sigma1 chiral ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    vector<vd_t> jZq_over_Sigma1_eqmoms(neq_moms,vd_t(njacks)), jZq_over_Sigma1_with_em_eqmoms(neq_moms,vd_t(njacks));
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<neq_moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZq_over_Sigma1_eqmoms[imom][ijack]=jZq_chiral_eqmoms[imom][ijack]/jSigma1_chiral_eqmoms[imom][ijack];
            jZq_over_Sigma1_with_em_eqmoms[imom][ijack]=jZq_chiral_with_em_eqmoms[imom][ijack]/jSigma1_chiral_with_em_eqmoms[imom][ijack];
        }

    cout<<"Zq over Sigma1 chiral"<<endl;
    plot_Zq_chiral(jZq_over_Sigma1_eqmoms,p2_vector_eqmoms,"Zq_over_Sigma1_chiral","eqmoms");
    cout<<"Zq over Sigma1 chiral with em"<<endl;
    plot_Zq_chiral(jZq_over_Sigma1_with_em_eqmoms,p2_vector_eqmoms,"Zq_over_Sigma1_chiral_with_em","eqmoms");
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Z with subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    plot_Z_sub(jZ_eqmoms,jZ_sub_eqmoms,p2_vector_eqmoms,"Z","eqmoms");
    plot_Z_sub(jZ1_eqmoms,jZ1_sub_eqmoms,p2_vector_eqmoms,"Z1","eqmoms");
    
    vector<jZbil_t> jZ_with_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_with_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks));
    vector<jZbil_t> jZ_sub_with_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks)), jZ1_sub_with_em_eqmoms(neq_moms,jZbil_t(vvvd_t(vvd_t(vd_t(5),nmr),nmr),njacks));
    
#pragma omp parallel for collapse(5)
    for(int imom=0;imom<neq_moms;imom++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int k=0;k<5;k++)
                    {
                        jZ_with_em_eqmoms[imom][ijack][mr_fw][mr_bw][k]=jZ_eqmoms[imom][ijack][mr_fw][mr_bw][k]+jZ_em_eqmoms[imom][ijack][mr_fw][mr_bw][k];
                        jZ1_with_em_eqmoms[imom][ijack][mr_fw][mr_bw][k]=jZ1_eqmoms[imom][ijack][mr_fw][mr_bw][k]+jZ1_em_eqmoms[imom][ijack][mr_fw][mr_bw][k];
                        jZ_sub_with_em_eqmoms[imom][ijack][mr_fw][mr_bw][k]=jZ_sub_eqmoms[imom][ijack][mr_fw][mr_bw][k]+jZ_em_sub_eqmoms[imom][ijack][mr_fw][mr_bw][k];
                        jZ1_sub_with_em_eqmoms[imom][ijack][mr_fw][mr_bw][k]=jZ1_sub_eqmoms[imom][ijack][mr_fw][mr_bw][k]+jZ1_em_sub_eqmoms[imom][ijack][mr_fw][mr_bw][k];
                    }
    
    plot_Z_sub(jZ_with_em_eqmoms,jZ_sub_with_em_eqmoms,p2_vector_eqmoms,"Z_with_em","eqmoms");
    plot_Z_sub(jZ1_with_em_eqmoms,jZ1_sub_with_em_eqmoms,p2_vector_eqmoms,"Z1_with_em","eqmoms");
    
    plot_Z_sub(jZ_em_eqmoms,jZ_em_sub_eqmoms,p2_vector_eqmoms,"Z_em_correction","eqmoms");
    plot_Z_sub(jZ1_em_eqmoms,jZ1_em_sub_eqmoms,p2_vector_eqmoms,"Z1_em_correction","eqmoms");
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Z chiral extrapolation  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    plot_ZPandS_chiral_extrapolation("P",jGp_equivalent_eqmoms,jGp_subpole_eqmoms,jGp_pars_eqmoms,m_eff_equivalent,"Gp_chiral_extrapolation","eqmoms");
    plot_ZPandS_chiral_extrapolation("S",jGs_equivalent_eqmoms,jGs_subpole_eqmoms,jGs_pars_eqmoms,m_eff_equivalent,"Gs_chiral_extrapolation","eqmoms");
    plot_ZVAT_chiral_extrapolation("V",jGv_equivalent_eqmoms,jGv_pars_eqmoms,m_eff_equivalent,"Gv_chiral_extrapolation","eqmoms");
    plot_ZVAT_chiral_extrapolation("A",jGa_equivalent_eqmoms,jGa_pars_eqmoms,m_eff_equivalent,"Ga_chiral_extrapolation","eqmoms");
    plot_ZVAT_chiral_extrapolation("T",jGt_equivalent_eqmoms,jGt_pars_eqmoms,m_eff_equivalent,"Gt_chiral_extrapolation","eqmoms");
    
    plot_ZPandS_chiral_extrapolation("P",jGp_em_equivalent_eqmoms,jGp_em_subpole_eqmoms,jGp_em_pars_eqmoms,m_eff_equivalent,"Gp_em_chiral_extrapolation","eqmoms");
    plot_ZPandS_chiral_extrapolation("S",jGs_em_equivalent_eqmoms,jGs_em_subpole_eqmoms,jGs_em_pars_eqmoms,m_eff_equivalent,"Gs_em_chiral_extrapolation","eqmoms");
    plot_ZVAT_chiral_extrapolation("V",jGv_em_equivalent_eqmoms,jGv_em_pars_eqmoms,m_eff_equivalent,"Gv_em_chiral_extrapolation","eqmoms");
    plot_ZVAT_chiral_extrapolation("A",jGa_em_equivalent_eqmoms,jGa_em_pars_eqmoms,m_eff_equivalent,"Ga_em_chiral_extrapolation","eqmoms");
    plot_ZVAT_chiral_extrapolation("T",jGt_em_equivalent_eqmoms,jGt_em_pars_eqmoms,m_eff_equivalent,"Gt_em_chiral_extrapolation","eqmoms");
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Z chiral ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    // cout<<"Z chiral"<<endl;
    // plot_Z_chiral(jZ_chiral_eqmoms,p2_vector_eqmoms,"Z_chiral","eqmoms");
    cout<<"Z1 chiral"<<endl;
    plot_Z_chiral(jZ1_chiral_eqmoms,p2_vector_eqmoms,"Z1_chiral","eqmoms");
    
    vector<vvd_t> jZ_chiral_with_em_eqmoms(neq_moms,vvd_t(vd_t(5),njacks)), jZ1_chiral_with_em_eqmoms(neq_moms,vvd_t(vd_t(5),njacks));
    
#pragma omp parallel for collapse(3)
    for(int imom=0;imom<neq_moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int k=0;k<5;k++)
            {
                jZ_chiral_with_em_eqmoms[imom][ijack][k]=jZ_chiral_eqmoms[imom][ijack][k]+jZ_em_chiral_eqmoms[imom][ijack][k];
                jZ1_chiral_with_em_eqmoms[imom][ijack][k]=jZ1_chiral_eqmoms[imom][ijack][k]+jZ1_em_chiral_eqmoms[imom][ijack][k];
            }

    // cout<<"Z chiral with em"<<endl;
    // plot_Z_chiral(jZ_chiral_with_em_eqmoms,p2_vector_eqmoms,"Z_chiral_with_em","eqmoms");
    //cout<<"Z1 chiral with em"<<endl;
    //plot_Z_chiral(jZ1_chiral_with_em_eqmoms,p2_vector_eqmoms,"Z1_chiral_with_em","eqmoms");

    //cout<<"Z chiral em correction"<<endl;
    //plot_Z_chiral(jZ_em_chiral_eqmoms,p2_vector_eqmoms,"Z_chiral_em_correction","eqmoms");
    cout<<"Z1 chiral em correction"<<endl;
    plot_Z_chiral(jZ1_em_chiral_eqmoms,p2_vector_eqmoms,"Z1_chiral_em_correction","eqmoms");
    

      
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Z RIp_ainv  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    cout<<"Z1(1/a) -- p_min>1"<<endl;
    plot_ZO_RIp_ainv(jZO_RIp_ainv_eqmoms,p2_vector_eqmoms,"ZO_RIp_ainv","eqmoms",1.0);

    cout<<"Z1(1/a) em correction -- p_min>1"<<endl;
    plot_ZO_RIp_ainv(jZO_em_RIp_ainv_eqmoms,p2_vector_eqmoms,"ZO_em_RIp_ainv","eqmoms",1.0);

    cout<<"Z1(1/a) -- p_min>0.9"<<endl;
    plot_ZO_RIp_ainv(jZO_RIp_ainv_eqmoms,p2_vector_eqmoms,"ZO_RIp_ainv","eqmoms",0.9);

    cout<<"Z1(1/a) em correction -- p_min>0.9"<<endl;
    plot_ZO_RIp_ainv(jZO_em_RIp_ainv_eqmoms,p2_vector_eqmoms,"ZO_em_RIp_ainv","eqmoms",0.9);

    cout<<"Z1(1/a) -- p_min>1.1"<<endl;
    plot_ZO_RIp_ainv(jZO_RIp_ainv_eqmoms,p2_vector_eqmoms,"ZO_RIp_ainv","eqmoms",1.1);

    cout<<"Z1(1/a) em correction -- p_min>1.1"<<endl;
    plot_ZO_RIp_ainv(jZO_em_RIp_ainv_eqmoms,p2_vector_eqmoms,"ZO_em_RIp_ainv","eqmoms",1.1);

    

    cout<<"Sigma1(1/a)"<<endl;
    plot_Zq_RIp_ainv(jSigma1_RIp_ainv_eqmoms,p2_vector_eqmoms,"Sigma1_RIp_ainv","eqmoms");

    cout<<"Sigma1(1/a) em correction"<<endl;
    plot_Zq_RIp_ainv(jSigma1_em_RIp_ainv_eqmoms,p2_vector_eqmoms,"Sigma1_em_RIp_ainv","eqmoms");



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

    
    // cout<<"***DEBUG***"<<endl;
    // for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    //   for(int mr_bw=0;mr_bw<nmr;mr_bw++)
    // 	for(int i=0;i<2;i++)
    // 	  cout<<eff_mass_array[mr_fw][mr_bw][i]<<endl;
    // cout<<"***DEBUG***"<<endl;
  
    vvd_t eff_mass(vd_t(0.0,nmr),nmr);
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
      for(int mr_bw=0;mr_bw<nmr;mr_bw++)
	eff_mass[mr_fw][mr_bw] = eff_mass_array[mr_fw][mr_bw][0];

      
    cout<<endl;
    cout<<"r_fw \t m_fw \t r_bw \t m_bw \t  eff_mass"<<endl;
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
      for(int mr_bw=0;mr_bw<nmr;mr_bw++)
	cout<<mr_fw%nr<<"\t"<<(mr_fw - mr_fw%nr)/nr<<"\t"<<mr_bw%nr<<"\t"<<(mr_bw - mr_bw%nr)/nr<<"\t"<<eff_mass[mr_fw][mr_bw]<<endl;
 
    // cout<<"eff_mass: "<<eff_mass_array[0]<<" +- "<<eff_mass_array[1]<<endl;
    
    
    
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



