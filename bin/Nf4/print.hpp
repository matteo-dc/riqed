#ifndef PRINT_HPP
#define PRINT_HPP

template <class T>
void print_vec( T &vec, const char* path);

#define PRINT(NAME)				\
print_vec(NAME##_##allmoms,"allmoms/"#NAME);	\
print_vec(NAME##_##eqmoms,"eqmoms/"#NAME)


#endif
