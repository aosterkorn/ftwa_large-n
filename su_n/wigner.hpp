/** 
 *  @file   wigner.hpp 
 *  @brief  Gaussian Wigner function models
 *  @author Alexander Osterkorn
 *  @date   2022-05-23 
 */

#ifndef SU_N_WIGNER_HPP
#define SU_N_WIGNER_HPP

#ifdef FTWA_WITH_TIMER
#include <chrono>
#endif

#include <sstream>
#include <fstream>

#include <armadillo>

#include "../base/basic_defs.hpp"
#include "../base/lattice.hpp"
#include "../base/fourier.hpp"

namespace ftwa_su_n {

enum WignerFuncModel { Gaussian = 1, TwoPoint = 2 };

class WignerFuncProduct {
  public:
    WignerFuncProduct(const Lattice& lattice);
    WignerFuncProduct(const Lattice& lattice,
                      unsigned int nf,
                      WignerFuncModel wignerFuncModel);
    WignerFuncProduct(const Lattice& lattice,
                      unsigned int nf,
                      WignerFuncModel wignerFuncModel,
                      const arma::vec& occupations);
    
    virtual void generate(std::mt19937& generator, arma::cx_vec& res) const;
    
  protected:
    const Lattice& _lattice;
    unsigned int _nf;
    WignerFuncModel _wignerFuncModel;
    arma::vec _occupations;
};

/**
 * Gaussian Wigner function for a many-body product state.
 * It corresponds to a Slater determinant
 * with the occupied modes listed in @param occupations.
 * Note that the distributions for the real and imaginary part are
 * 0.5 of the covariance matrix entries.
 */

/**
 * Gaussian Wigner function for a temperature zero Fermi sea state in 2D.
 * The number of particles needs to be provided in @param num_particles
 */
class WignerFuncFermiSeaGS : public WignerFuncProduct {
  public:
    WignerFuncFermiSeaGS(const Lattice& lattice,
                         unsigned int nf,
                         WignerFuncModel wignerFuncModel,
                         unsigned int num_particles);
    WignerFuncFermiSeaGS(const Lattice& lattice,
                         unsigned int nf,
                         WignerFuncModel wignerFuncModel,
                         const arma::vec& occupations);
    
    void generate(std::mt19937& generator, arma::cx_vec& res) const;
};

/**
 * Gaussian Wigner function for a Fermi sea state in two dimensions
 * and at non-zero temperature @param temp
 * Particle number needs to be provided in @param num_particles
 */
class WignerFuncFermiSeaTemp : public WignerFuncProduct {
  public:
    WignerFuncFermiSeaTemp(const Lattice& lattice,
                           unsigned int nf,
                           WignerFuncModel wignerFuncModel,
                           unsigned int numParticles,
                           double temp);
    
    // void generate(std::mt19937& generator, arma::cx_vec& res) const;
    
  private:
    unsigned int _numParticles;
    double _temp;
    double _chemPot;
    arma::vec _energies;
    
    double _particleNumberFromChemPot(double chemPot);
};

class HubHieGaussWigner : public WignerFuncFermiSeaGS {
  public:
    HubHieGaussWigner(const Lattice& lattice,
                      unsigned int nf,
                      WignerFuncModel wignerFuncModel,
                      unsigned int num_particles,
                      const FourierTransformer2dPBC& ft);
    
    virtual void generate(std::mt19937& generator, arma::cx_vec& res) const;

  private:
    const FourierTransformer2dPBC& _ft;
};

////////////////////////////////////////////////////////////////////////
////////////////////// Hubbard-Heisenberg model ////////////////////////
////////////////////////////////////////////////////////////////////////

/**
 * Gaussian Wigner function for the mean-field ground state of the
 * Hubbard-Heisenberg model with the tilted unit cell of two sites
 * Values of the unit cell bonds: @param ucv
 * Fourier transformation for the lattice: @param ft
 */
class HubHeiGaussWignerInfty {
  public:
    HubHeiGaussWignerInfty(
        const Lattice& lattice,
        const HubbardHeisenbergParameters& params,
        const struct unitCellValues& ucv,
        const ftwa_su_n::FTSqrt2CellLattice& ft
    ) : _lattice(lattice), _params(params), _ucv(ucv), _ft(ft) { };
    
    virtual void generate(arma::cx_vec& res) const;
    
    void generateMom(arma::cx_vec& res) const;
    
  protected:
    const Lattice& _lattice;
    const HubbardHeisenbergParameters& _params;
    const struct unitCellValues& _ucv;
    const FTSqrt2CellLattice& _ft;
};

/**
 * Gaussian Wigner function for the mean-field ground state of the
 * Hubbard-Heisenberg model with the tilted unit cell of two sites
 * quenched to a finite value of $N$.
 * Values of the unit cell bonds: @param ucv
 * Fourier transformation for the lattice: @param ft
 */
class HubHeiGaussWignerFiniteN : public HubHeiGaussWignerInfty {
  public:
    HubHeiGaussWignerFiniteN(
        const Lattice& lattice,
        const HubbardHeisenbergParameters& params,
        const struct unitCellValues& ucv,
        const ftwa_su_n::FTSqrt2CellLattice& ft,
        const unsigned int n,
        std::mt19937& generator
    ) : HubHeiGaussWignerInfty(lattice, params, ucv, ft), _n(n), _generator(generator) { };
    
    void generate(arma::cx_vec& res) const;
    
  private:
    const unsigned int _n;
    std::mt19937& _generator;
};

/**
 * Gaussian Wigner function for the mean-field ground state of the
 * Hubbard-Heisenberg model with the tilted unit cell of two sites
 * quenched to a finite value of $N$ BUT only including fluctuations
 * of the momentum-diagonal variables $\rho_{k k}$.
 * This is mainly for experimental purposes.
 * Values of the unit cell bonds: @param ucv
 * Fourier transformation for the lattice: @param ft
 */
class HubHeiGaussWignerFiniteNDiagFluct : public HubHeiGaussWignerInfty {
  public:
    HubHeiGaussWignerFiniteNDiagFluct(
        const Lattice& lattice,
        const HubbardHeisenbergParameters& params,
        const struct unitCellValues& ucv,
        const ftwa_su_n::FTSqrt2CellLattice& ft,
        const unsigned int n,
        std::mt19937& generator
    ) : HubHeiGaussWignerInfty(lattice, params, ucv, ft), _n(n), _generator(generator) { };
    
    void generate(arma::cx_vec& res) const;
    
  private:
    const unsigned int _n;
    std::mt19937& _generator;
};

//~ class HubHeiGaussWignerInfty {
  //~ public:
    //~ HubHeiGaussWignerInfty(
        //~ const Lattice& lattice,
        //~ const HubbardHeisenbergParameters& params,
        //~ const struct unitCellValues& ucv,
        //~ const ftwa_su_n::FourierTransformer2dPBC& ft
    //~ ) : _lattice(lattice), _params(params), _ucv(ucv), _ft(ft) { };
    
    //~ virtual void generate(arma::cx_vec& res) const;
    
  //~ protected:
    //~ const Lattice& _lattice;
    //~ const HubbardHeisenbergParameters& _params;
    //~ const struct unitCellValues& _ucv;
    //~ const ftwa_su_n::FourierTransformer2dPBC& _ft;
//~ };

//~ class HubHeiGaussWignerFiniteN : public HubHeiGaussWignerInfty {
  //~ public:
    //~ HubHeiGaussWignerFiniteN(
        //~ const Lattice& lattice,
        //~ const HubbardHeisenbergParameters& params,
        //~ const struct unitCellValues& ucv,
        //~ const ftwa_su_n::FourierTransformer2dPBC& ft,
        //~ const unsigned int n,
        //~ std::mt19937& generator
    //~ ) : HubHeiGaussWignerInfty(lattice, params, ucv, ft), _n(n), _generator(generator) { };
    
    //~ void generate(arma::cx_vec& res) const;
    
  //~ private:
    //~ const unsigned int _n;
    //~ std::mt19937& _generator;
//~ };

}

#endif // SU_N_WIGNER_HPP
