#ifndef BASIC_DEFS_HPP
#define BASIC_DEFS_HPP

#include <map>
#include <vector>
#include <algorithm>

#include <armadillo>
#include <boost/numeric/odeint.hpp>

#include "lattice.hpp"

#define STATUS_OKAY 0
#define STATUS_ERROR 1

/**
 * 
 * defs for ftwa
 * 
 **/

/**
 * Indices in fTWA rho vectors are sorted wrt to site indices:
 * 0 00
 * 1 01
 * 2 11
 * 3 02
 * ...
 * Compute the corresponding position in this vector for given site index.
 */
inline unsigned int ftwa_index(unsigned int i, unsigned int j) {
    return j*(j+1)/2 + i;
}

inline std::complex<double> rhoVal(const arma::cx_vec& x, unsigned int i, unsigned int j) {
    return (i > j) ? std::conj(x(ftwa_index(j, i))) : x(ftwa_index(i, j));
}

inline std::complex<double> DmatVal(unsigned int V, const arma::cx_mat& Dmat, arma::uword i, arma::uword j, arma::uword m, arma::uword n) {
    return (i > j) ? std::conj(Dmat(ftwa_index(j, i), n+V*m)) : Dmat(ftwa_index(i, j), m+V*n);
}

inline std::complex<double> DmatValFact(unsigned int V, const arma::cx_vec& rho, arma::uword i, arma::uword j, arma::uword m, arma::uword n) {
    //~ return rhoVal(rho, i, j)*rhoVal(rho, m, n);
    
    return rhoVal(rho, i, j)*rhoVal(rho, m, n);
}

inline std::complex<double> TmatDecouplVal(unsigned int V, const arma::cx_vec& rho, const arma::cx_mat& Dmat,
    arma::uword i, arma::uword j, arma::uword m, arma::uword n, arma::uword p, arma::uword q) {
    
    //~ std::cout << rhoVal(rho, i, j)*DmatVal(V, Dmat, m, n, p, q)
        //~ + rhoVal(rho, m, n)*DmatVal(V, Dmat, i, j, p, q)
        //~ + rhoVal(rho, p, q)*DmatVal(V, Dmat, i, j, m, n) << std::endl;
    //~ std::cout <<  rhoVal(rho, i, j)*rhoVal(rho, m, n)*rhoVal(rho, p, q) << std::endl << std::endl;
    
    //~ return 1.0/3.0*(rhoVal(rho, i, j)*DmatVal(V, Dmat, m, n, p, q)
                  //~ + rhoVal(rho, m, n)*DmatVal(V, Dmat, i, j, p, q)
                  //~ + rhoVal(rho, p, q)*DmatVal(V, Dmat, i, j, m, n);
                  //~ - ((i == j) ? (DmatVal(V, Dmat, m, n, p, q) - rhoVal(rho, m, n)*rhoVal(rho, p, q)) : 0.0)
                  //~ - ((m == n) ? (DmatVal(V, Dmat, i, j, p, q) - rhoVal(rho, i, j)*rhoVal(rho, p, q)) : 0.0)
                  //~ - ((p == q) ? (DmatVal(V, Dmat, i, j, m, n) - rhoVal(rho, i, j)*rhoVal(rho, m, n)) : 0.0));
    
    //~ return rhoVal(rho, i, j)*rhoVal(rho, m, n)*rhoVal(rho, p, q)
        //~ + 1.0/3.0*((rhoVal(rho, i, j) - ((i == j) ? 1.0 : 0.0))*(DmatVal(V, Dmat, m, n, p, q) - rhoVal(rho, m, n)*rhoVal(rho, p, q))
                 //~ + (rhoVal(rho, m, n) - ((m == n) ? 1.0 : 0.0))*(DmatVal(V, Dmat, p, q, i, j) - rhoVal(rho, p, q)*rhoVal(rho, i, j))
                 //~ + (rhoVal(rho, p, q) - ((p == q) ? 1.0 : 0.0))*(DmatVal(V, Dmat, i, j, m, n) - rhoVal(rho, i, j)*rhoVal(rho, m, n)));
    
    return rhoVal(rho, i, j)*DmatVal(V, Dmat, m, n, p, q)
         + rhoVal(rho, m, n)*DmatVal(V, Dmat, i, j, p, q)
         + rhoVal(rho, p, q)*DmatVal(V, Dmat, i, j, m, n)
     - 2.0*rhoVal(rho, i, j)*rhoVal(rho, m, n)*rhoVal(rho, p, q);
    
    //~ return rhoVal(rho, i, j)*rhoVal(rho, m, n)*rhoVal(rho, p, q);
}

struct SimulationParameters {
    double start_time;
    double end_time;
    unsigned int num_tsteps;
    double checkpoint_dt;
};

struct HubbardParameters {
    double t;
    double U;
};

struct HubbardHeisenbergParameters {
    double t;
    double J;
    double U;
    double doping;
};

struct PeierlsPulseParameters {
    double ampl_x;
    double ampl_y;
    double freq;
    double phase;
    double temp_centr;
    double temp_width;
    bool is_spat_uniform;
    double spat_centr_x;
    double spat_centr_y;
    double spat_width;
};

/**
 * Tight-binding energy dispersion of a square lattice with periodic boundary conditions.
 * @param L is the length of the lattice in one dimension.
 * @param mx and @param my are the indices of the momentum (within the interval [0, L-1])
 * in x and y direction.
 */
inline double dispTightBinding2d(unsigned int L, unsigned int mx, unsigned int my) {
    // return -2.0*cos((2*M_PI/(double) L) * ((double) mx + 0.5)) - 2.0*cos((2*M_PI/(double) L) * ((double) my + 0.5));
    return -2.0*cos((2*M_PI/(double) L) * ((double) mx)) - 2.0*cos((2*M_PI/(double) L) * ((double) my));
}

/**
 * Return true if eps( m1 ) < eps( m2 ), where eps is the energy dispersion
 * of the tight-binding model on a square lattice with periodic boundary conditions.
 * @param L is the length of the lattice in one dimension.
 * @param m1 and @param m2 are the number of the momentum points on the lattice.
 * 
 * Note: This function is used to sort the momenta in ascending order of their energy.
 * 
 * @see dispTightBinding2d
 */
inline bool sortFermi2D(unsigned int L, unsigned int m1, unsigned int m2) {
    return dispTightBinding2d(L, m1 % L, m1 / L) < dispTightBinding2d(L, m2 % L, m2 / L);
}

/**
 * Fermi-Dirac distribution function.
 * @param en is the energy of the state.
 * @param chem_pot is the chemical potential.
 * @param temp is the temperature.
 */
inline double fermiDirac(double en, double chem_pot, double temp) {
    return 1.0/(1.0 + std::exp((en - chem_pot)/temp));
}

void halfFilling2D(const Lattice& lattice, arma::vec& kvals);

void fermiSurface2D(const Lattice& lattice, unsigned int N, arma::vec& kvals);

void plotFermiSurface2D(const Lattice& lattice, const arma::vec& kvals);

inline double dispTightBinding1dLR(unsigned int L, unsigned int mx);

void halfFilling1dLR(const Lattice& lattice, arma::vec& kvals);

////////////////////////////////////////////////////////////////////////
//////////////////////// useful functions //////////////////////////////
////////////////////////////////////////////////////////////////////////

/**
 * Compute the distance R = r_i - r_j of two points on the lattice
 * in x-direction.
 * 
 * Ordering of the lattice sites:
 * 0  1  2  3
 * 4  5  6  7
 * 9  9  10 11
 * 12 13 14 15
 */
inline int xdist2dPBC(unsigned int L, unsigned int i, unsigned int j) {
    unsigned int ix = i % L, jx = j % L;
    bool f = (ix < jx);
    unsigned int d = (f ? jx - ix : ix - jx);
    int sL = (int) L, sd = (int) d;
    
    return d > L-d ? (f ? sL-sd : sd-sL) : (f ? -sd : sd);
}

/**
 * Compute the distance R = r_i - r_j of two points on the lattice
 * in y-direction.
 * 
 * Ordering of the lattice sites:
 * 0  1  2  3
 * 4  5  6  7
 * 9  9  10 11
 * 12 13 14 15
 */
inline int ydist2dPBC(unsigned int L, unsigned int i, unsigned int j) {
    unsigned int iy = i / L, jy = j / L;
    bool f = (iy < jy);
    unsigned int d = (f ? (jy - iy) : (iy - jy));
    int sL = (int) L, sd = (int) d;
    
    return d > L-d ? (f ? sL-sd : sd-sL) : (f ? -sd : sd);
}

inline constexpr unsigned int pow2(unsigned int i) {
    return 1 << i;
}

template<typename T>
inline int bitAtPos(const T& n, int pos) {
    T bit = n & (1ULL << pos);
    return bit == 0 ? 0 : +1;
}

////////////////////////////////////////////////////////////////////////
///////////////////////////// type traits //////////////////////////////
////////////////////////////////////////////////////////////////////////

// number types
template <typename T>
struct NumberTypeTrait{};

// specialize for real types
template <>
struct NumberTypeTrait<float>
{
    typedef float NumberType;
    typedef float RealType;
};

template <>
struct NumberTypeTrait<double>
{
    typedef double NumberType;
    typedef double RealType;
};

// specialize for complex types
template <>
struct NumberTypeTrait<std::complex<float>>
{
    typedef std::complex<float> NumberType;
    typedef float RealType;
};

template <>
struct NumberTypeTrait<std::complex<double>>
{
    typedef std::complex<double> NumberType;
    typedef double RealType;
};

////////////////////////////////////////////////////////////////////////
///////////////////// Armadillo-related stuff //////////////////////////
////////////////////////////////////////////////////////////////////////

namespace boost { namespace numeric { namespace odeint {

template <>
struct is_resizeable<arma::cx_vec>
{
    typedef boost::true_type type;
    const static bool value = type::value;
};

template <>
struct same_size_impl<arma::cx_vec, arma::cx_vec>
{
    static bool same_size(const arma::cx_vec& x, const arma::cx_vec& y)
    {
        return x.size() == y.size();   // not sure if this is correct for arma
    }
};

template<>
struct resize_impl<arma::cx_vec, arma::cx_vec>
{
    static void resize(arma::cx_vec& v1, const arma::cx_vec& v2)
    {
        v1.resize(v2.size());     // not sure if this is correct for arma
    }
};

template <>
struct is_resizeable<arma::cx_mat> {
    typedef boost::true_type type;
    const static bool value = type::value;
};

template <>
struct same_size_impl<arma::cx_mat, arma::cx_mat> {
    static bool same_size(const arma::cx_mat& x, const arma::cx_mat& y) {
        return arma::size(x) == arma::size(y);
    }
};

template<>
struct resize_impl<arma::cx_mat, arma::cx_mat> {
    static void resize(arma::cx_mat& v1, const arma::cx_mat& v2) {
        v1.set_size(arma::size(v2));
    }
};

} } } // namespace boost::numeric::odeint

#endif // BASIC_DEFS_HPP
