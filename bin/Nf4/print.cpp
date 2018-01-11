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

//template <class T>
//void print_internal(valarray<T> &v, ofstream& outfile)
//{
//    for(auto &i : v) print_internal(i,outfile);
//}
//
//template <class T>
//void print_vec( T &vec, const char* path)
//{
//    ofstream outfile(path/*,ofstream::binary*/,ofstream::out);
//     
//    if (outfile.is_open())
//    {
//        for(auto &i : vec)
//            print_internal(i,outfile);
//        
//        outfile.close();
//        
//    }
//    else cout << "Unable to open the output file "<<path<<endl;
//}