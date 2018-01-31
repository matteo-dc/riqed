#include "aliases.hpp"
#include <iostream>
#include <fstream>

//print of file
void print_internal(double t,ofstream& outfile)
{
//    outfile.write((char*) &t,sizeof(double));
    outfile<<t<<endl;
}
//template <class T>
void print_internal(VectorXd &V, ofstream& outfile)
{
    for(int i=0; i<V.size();i++) print_internal(V(i),outfile);
}

////print of file binary
//void print_internal_bin(double t,ofstream& outfile)
//{
//    outfile.write((char*) &t,sizeof(double));
////    outfile<<t<<endl;
//}
//void print_internal_bin(int t,ofstream& outfile)
//{
//    outfile.write((char*) &t,sizeof(int));
//    //    outfile<<t<<endl;
//}

//template <class T>
void print_internal_bin(VectorXd &V, ofstream& outfile)
{
    for(int i=0; i<V.size();i++) print_internal(V(i),outfile);
}
