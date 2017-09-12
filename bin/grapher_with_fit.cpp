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

valarray<VectorXd> fit_chiral_jackknife(const vvd_t &coord, const vd_t &error, const vector<vd_t> &y, const int range_min, const int range_max)
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
    for(int i=range_min; i<=range_max; i++)
    {
        for(int j=0; j<n_par; j++)
            for(int k=0; k<n_par; k++)
                if(isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
        
        for(int ijack=0; ijack<njacks; ijack++)
            for(int k=0; k<n_par; k++)
                if(isnan(error[i])==0) Sy[ijack](k) += y[i][ijack]*coord[k][i]/(error[i]*error[i]);
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
    
    return jpars;
    
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
            Zq_err[imom][mr]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_ave[imom][mr]-Zq_ave[imom][mr]*Zq_ave[imom][mr]);
    
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
        Zq_err[imom]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_ave[imom]-Zq_ave[imom]*Zq_ave[imom]);
    
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
            Zq_par_err[imom][ipar]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_par_ave[imom][ipar]-Zq_par_ave[imom][ipar]*Zq_par_ave[imom][ipar]);
    
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
                    Z_err[imom][mr_fw][mr_bw][k]=sqrt((double)(njacks-1))*sqrt(sqr_Z_ave[imom][mr_fw][mr_bw][k]-Z_ave[imom][mr_fw][mr_bw][k]*Z_ave[imom][mr_fw][mr_bw][k]);
    
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
            Z_chiral_err[imom][k]=sqrt((double)(njacks-1))*sqrt(sqr_Z_chiral_ave[imom][k]-Z_chiral_ave[imom][k]*Z_chiral_ave[imom][k]);
    
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
    scriptfile<<"set xlabel '$\\tilde{p}^2$'"<<endl;
    scriptfile<<"set ylabel '$Z_q$'"<<endl;
    // scriptfile<<"set yrange [0.7:0.9]"<<endl;
    scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q$'"<<endl;
    scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_sub_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'red' title '$Z_q$ corrected'"<<endl;
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
    
    vvvd_t Zq_equivalent = average_Zq(jZq_equivalent_and_chiral_extr);  //Zq[ave/err][imom][ieq]
    vvvd_t Zq_pars=average_pars(jZq_pars);
    
    ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
    
    //datafile1<<0<<"\t"<<Zq_pars[0][0]<<"\t"<<Zq_pars[1][0]<<endl;
    for(size_t ieq=0;ieq<m_eff_equivalent_Zq.size()+1;ieq++)
    {
        if(ieq==0)
            datafile1<<0<<"\t"<<Zq_equivalent[0][4][ieq]<<"\t"<<Zq_equivalent[1][4][ieq]<<endl;  //print only for p2~1
        else
            datafile1<<m_eff_equivalent_Zq[ieq-1]*m_eff_equivalent_Zq[ieq-1]<<"\t"<<Zq_equivalent[0][4][ieq]<<"\t"<<Zq_equivalent[1][4][ieq]<<endl;  //print only for p2~1
    }
    datafile1.close();
    
    double A=Zq_pars[0][4][0];
    double B=Zq_pars[0][4][1];
    
    ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");
    
    scriptfile<<"set autoscale xy"<<endl;
    scriptfile<<"set xlabel '$M_{eff}^2$'"<<endl;
    scriptfile<<"set ylabel '$Z_Q$'"<<endl;
    scriptfile<<"set xrange [-0.003:0.05]"<<endl;
    scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q$'"<<endl;
    scriptfile<<"replot '< head -1 plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 7 lc rgb 'black' title '$Z_q$ chiral extr.'"<<endl;
    scriptfile<<"f(x)="<<A<<"+"<<B<<"*x"<<endl;
    scriptfile<<"replot f(x) notitle"<<endl;
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
    vvd_t Zq_chiral = average_Zq_chiral(jZq_chiral);  //Zq[ave/err][imom][nm]
    
    ///**************************///
    //linear fit
    int p2_min=4;  //a2p2~1
    int p2_max=p2_vector.size();
    
    vvd_t coord_linear(vd_t(0.0,p2_vector.size()),2);
    
    for(int i=0; i<p2_vector.size(); i++)
    {
        coord_linear[0][i] = 1.0;  //costante
        coord_linear[1][i] = p2_vector[i];   //p^2
    }
    
    vXd_t jZq_chiral_par=fit_chiral_jackknife(coord_linear,Zq_chiral[1],jZq_chiral,p2_min,p2_max);  //jZq_chiral_par[ijack][par]

    //   vvvd_t Zq_chiral_par=average_pars(jZq_chiral_par); //Zq[ave/err][imom][ieq]

    int njacks=jZq_chiral_par.size();
    int pars=jZq_chiral_par[0].size();
    
    vd_t Zq_par_ave(0.0,pars), sqr_Zq_par_ave(0.0,pars), Zq_par_err(0.0,pars);
    vvd_t Zq_par_ave_err(vd_t(0.0,pars),2);
    
    for(int ipar=0;ipar<pars;ipar++)
      for(int ijack=0;ijack<njacks;ijack++)
	{
	  Zq_par_ave[ipar]+=jZq_pars[ijack](ipar)/njacks;
	  sqr_Zq_par_ave[ipar]+=jZq_pars[ijack](ipar)*jZq_pars[ijack](ipar)/njacks;
	}
  
    for(int ipar=0;ipar<pars;ipar++)
      Zq_par_err[ipar]=sqrt((double)(njacks-1))*sqrt(sqr_Zq_par_ave[ipar]-Zq_par_ave[ipar]*Zq_par_ave[ipar]);
    
    Zq_par_ave_err[0]=Zq_par_ave; //Zq_par_ave_err[ave/err][par]
    Zq_par_ave_err[1]=Zq_par_err;

    
    
    double A=Zq_chiral_fit_par[0][0];
    double A_err=Zq_chiral_fit_par[1][0];
    double B=Zq_chiral_fit_par[0][1];
    double B_err=Zq_chiral_fit_par[1][1];


    cout<<endl;
    cout<<"ZQ continuum limit extrapolation: Zq = "<<A<<" +/- "<<A_err<<endl<<endl; 
    
    ///*****************************///
                    
    
    ofstream datafile1("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data.txt");
    
    for(size_t imom=0;imom<p2_vector.size();imom++)
    {
        datafile1<<p2_vector[imom]<<"\t"<<Zq_chiral[0][imom]<<"\t"<<Zq_chiral[1][imom]<<endl;  //print only for M0R0
    }
    datafile1.close();

    ofstream datafile2("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_data_fit.txt");
    datafile2<<0<<"\t"<<A<<"\t"<<A_err<<endl;
    datafile2.close();    
    
    ofstream scriptfile("plot_data_and_script/plot_"+name+"_"+all_or_eq_moms+"_script.txt");
    
    scriptfile<<"set autoscale xy"<<endl;
    scriptfile<<"set xlabel '$\\tilde{p}^2$'"<<endl;
    scriptfile<<"set ylabel '$Z_q$'"<<endl;
    // scriptfile<<"set yrange [0.7:0.9]"<<endl;
    scriptfile<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_q$ chiral'"<<endl;
    scriptfile<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<all_or_eq_moms<<"_data_fit.txt' u 1:2:3 with errorbars pt 5 lc rgb 'red' notitle"<<endl;
    scriptfile<<"f(x)="<<A<<"+"<<B<<"*x"<<endl;
    scriptfile<<"replot f(x) notitle"<<endl;
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
        scriptfile[i]<<"set xlabel '$\\tilde{p}^2$'"<<endl;
        // scriptfile[i]<<"set yrange [0.7:0.9]"<<endl;
        scriptfile[i]<<"set ylabel '$Z_"<<bil[i]<<"$'"<<endl;
        scriptfile[i]<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_"<<bil[i]<<"$'"<<endl;
        scriptfile[i]<<"replot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_sub_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'red' title '$Z_"<<bil[i]<<"$ corrected'"<<endl;
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

void plot_Z_chiral(vector<vvd_t> &jZ_chiral, vector<double> &p2_vector, const string &name, const string &all_or_eq_moms)
{
    vvvd_t Z_chiral = average_Z_chiral(jZ_chiral);  //Z_chiral[ave/err][imom][k]
    
    vector<string> bil={"S","A","P","V","T"};
    
    vector<ofstream> datafile(5);
    
    for(int i=0;i<5;i++)
    {
        datafile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_data.txt");
        
        for(size_t imom=0;imom<p2_vector.size();imom++)
            datafile[i]<<p2_vector[imom]<<"\t"<<Z_chiral[0][imom][i]<<"\t"<<Z_chiral[1][imom][i]<<endl;
        
        datafile[i].close();
    }
    
    vector<ofstream> scriptfile(5);
    
    
    for(int i=0;i<5;i++)
    {
        scriptfile[i].open("plot_data_and_script/plot_"+name+"_"+bil[i]+"_"+all_or_eq_moms+"_script.txt");
        scriptfile[i]<<"set autoscale xy"<<endl;
        scriptfile[i]<<"set xlabel '$\\tilde{p}^2$'"<<endl;
        // scriptfile[i]<<"set yrange [0.7:0.9]"<<endl;
        scriptfile[i]<<"set ylabel '$Z_"<<bil[i]<<"$'"<<endl;
        scriptfile[i]<<"plot 'plot_data_and_script/plot_"<<name<<"_"<<bil[i]<<"_"<<all_or_eq_moms<<"_data.txt' u 1:2:3 with errorbars pt 6 lc rgb 'blue' title '$Z_"<<bil[i]<<"$ chiral'"<<endl;
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
    int njacks=jZ_chiral[0].size();
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
    scriptfile2<<"set xlabel '$\\tilde{p}^2$'"<<endl;
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
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zq chiral extrapolation  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    plot_Zq_chiral_extrapolation(jZq_equivalent_eqmoms,jZq_pars_eqmoms,m_eff_equivalent_Zq,"Zq_chiral_extrapolation","eqmoms");
    plot_Zq_chiral_extrapolation(jSigma1_equivalent_eqmoms,jSigma1_pars_eqmoms,m_eff_equivalent_Zq,"Sigma1_chiral_extrapolation","eqmoms");
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zq chiral ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    plot_Zq_chiral(jZq_chiral_eqmoms,p2_vector_eqmoms,"Zq_chiral","eqmoms");
    plot_Zq_chiral(jSigma1_chiral_eqmoms,p2_vector_eqmoms,"Sigma1_chiral","eqmoms");
    
    vector<vd_t> jZq_chiral_with_em_eqmoms(neq_moms,vd_t(njacks)), jSigma1_chiral_with_em_eqmoms(neq_moms,vd_t(njacks));
    
#pragma omp parallel for collapse(2)
    for(int imom=0;imom<neq_moms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            jZq_chiral_with_em_eqmoms[imom][ijack]=jZq_chiral_eqmoms[imom][ijack]+jZq_em_chiral_eqmoms[imom][ijack];
            jSigma1_chiral_with_em_eqmoms[imom][ijack]=jSigma1_chiral_eqmoms[imom][ijack]+jSigma1_em_chiral_eqmoms[imom][ijack];
        }
    
    plot_Zq_chiral(jZq_chiral_with_em_eqmoms,p2_vector_eqmoms,"Zq_chiral_with_em","eqmoms");
    plot_Zq_chiral(jSigma1_chiral_with_em_eqmoms,p2_vector_eqmoms,"Sigma1_chiral_with_em","eqmoms");
    
    plot_Zq_chiral(jZq_em_chiral_eqmoms,p2_vector_eqmoms,"Zq_chiral_em_correction","eqmoms");
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
    
    plot_Zq_chiral(jZq_over_Sigma1_eqmoms,p2_vector_eqmoms,"Zq_over_Sigma1_chiral","eqmoms");
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
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Z chiral ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
    plot_Z_chiral(jZ_chiral_eqmoms,p2_vector_eqmoms,"Z_chiral","eqmoms");
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
    
    plot_Z_chiral(jZ_chiral_with_em_eqmoms,p2_vector_eqmoms,"Z_chiral_with_em","eqmoms");
    plot_Z_chiral(jZ1_chiral_with_em_eqmoms,p2_vector_eqmoms,"Z1_chiral_with_em","eqmoms");
    
    plot_Z_chiral(jZ_em_chiral_eqmoms,p2_vector_eqmoms,"Z_chiral_em_correction","eqmoms");
    plot_Z_chiral(jZ1_em_chiral_eqmoms,p2_vector_eqmoms,"Z1_chiral_em_correction","eqmoms");
    
    
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    
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



