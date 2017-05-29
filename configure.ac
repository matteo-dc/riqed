AC_INIT([template],[1.0],[fr.sanfilippo@gmail.com])

AC_CONFIG_SRCDIR([bin])
AC_CONFIG_AUX_DIR([config])
AM_INIT_AUTOMAKE([1.0 no-define foreign subdir-objects])
AM_CONFIG_HEADER(bin/config.hpp)
AC_CONFIG_MACRO_DIR([m4])
AC_CANONICAL_HOST

AC_PROG_RANLIB

AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])

#silent automake
AM_SILENT_RULES([yes])

AX_CXX_COMPILE_STDCXX_11(noext,mandatory)
AX_CXXFLAGS_WARN_ALL

#add Eigen path, search it and enable
AC_ARG_WITH(eigen,
	AS_HELP_STRING([--with-eigen],[Specify where to find Eigen/Dense]),
	with_eigen=${withval}
	CPPFLAGS="$CPPFLAGS -I$with_eigen",
	with_eigen=no)
AC_MSG_RESULT([with Eigen... ${with_eigen}])
AC_LANG_PUSH([C++])
AC_CHECK_HEADERS([Eigen/Dense])
AC_LANG_POP([C++])
if test "$ac_cv_header_Eigen_Dense" == "no"
then
	AC_MSG_ERROR(["Unable to find Eigen"])
else
	AC_DEFINE_UNQUOTED([USE_EIGEN],[1],[Enable Eigen])
fi
AC_MSG_RESULT(enabling Eigen ... ${enable_eigen})

#openMP
AC_OPENMP
CFLAGS="$CXXFLAGS $OPENMP_CFLAGS"
CPPFLAGS="$CPPFLAGS $OPENMP_CPPFLAGS"
CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"

CXXFLAGS="-O3 $CXXFLAGS"

AC_CONFIG_FILES(Makefile bin/Makefile)

AC_OUTPUT
