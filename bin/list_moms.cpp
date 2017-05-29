#include <cmath>
#include <set>
#include <iostream>
#include <array>

using namespace std;

int L,T;
using coords_t=array<int,4>;

coords_t c_of_m(int m)
{
  coords_t out;
  for(int mu=3;mu>=1;mu--)
    {
      out[mu]=m%(2*L+1)-L;
      m/=2*L+1;
    }
  out[0]=m%(2*T+1)-T;
  
  return out;
}

ostream& operator<<(ostream &out,const coords_t &i)
{
  for(int mu=0;mu<4;mu++) out<<i[mu]<<" ";
  return out;
}

coords_t operator-(const coords_t &a,const coords_t &b)
{
  coords_t c;
  for(int mu=0;mu<4;mu++) c[mu]=a[mu]-b[mu];
  return c;
}

int norm2(const coords_t &cm)
{
  int s=0;
  for(auto c : cm) s+=c*c;
  return s;
}

int main(int narg,char **arg)
{
  L=T=5;
  if(narg>=2) L=atoi(arg[1]);
  if(narg>=3) T=atoi(arg[2]);
  
  int NM=(2*T+1)*pow(2*L+1,3);
  
  set<coords_t> list;
  
  for(int im1=0;im1<NM;im1++)
    for(int im2=0;im2<NM;im2++)
      {
	coords_t cm1=c_of_m(im1);
	coords_t cm2=c_of_m(im2);
	coords_t cQ=cm1-cm2;
	
	int n1=norm2(cm1);
	int n2=norm2(cm2);
	int nQ=norm2(cQ);
	
	if(n1==n2 and n1==nQ)
	  {
	    list.insert(cm1);
	    //cout<<cm1<<"    "<<cm2<<"   "<<sqrt(nQ)<<endl;
	  }
      }
  
  for(auto c : list) cout<<c<<endl;
  
  cout<<list.size()<<endl;
  cout<<NM<<endl;
  
  return 0;
}
