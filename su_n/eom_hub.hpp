#ifndef SU_N_EOM_HUB_HPP
#define SU_N_EOM_HUB_HPP

#ifdef FTWA_WITH_TIMER
#include <chrono>
#endif

#include <armadillo>

#include "../base/basic_defs.hpp"
#include "../base/lattice.hpp"
#include "../base/fourier.hpp"
#include "../base/checkpoint_manager.hpp"

namespace ftwa_su_n {

/**
 * Equation of motion for the Hubbard model in 2D with periodic boundary conditions.
 * system() computes the time derivative of the state vector (to be used with odeint)
 * observer() computes observables and (if FTWA_CACHE_CHECKPOINTS is defined)
 * stores the state vector as a checkpoint.
 */
class ODEHub2dPBC {
  public:
    ODEHub2dPBC(
        const Lattice& lattice,
        const HubbardParameters& params,
        const SimulationParameters& simParams,
        CheckpointManager& cm,
        const ftwa_su_n::FourierTransformer2dPBC& ft,
        unsigned int verbosity
    );
    
    ~ODEHub2dPBC();
    
    void system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const;
    
    void observer(const arma::cx_vec& x, const double t);
    
  private:
    const Lattice& _lattice;
    const HubbardParameters& _params;
    const SimulationParameters& _simParams;
    CheckpointManager& _cm;
    const ftwa_su_n::FourierTransformer2dPBC& _ft;
    
#ifdef FTWA_CACHE_CHECKPOINTS
    arma::cx_cube _checkpoints;
#endif
    bool _fourierOutput;
    unsigned int _verbosity;
};

class ODEHub2dMomPBC {
  public:
    ODEHub2dMomPBC(
        const Lattice& lattice,
        const HubbardParameters& params,
        const SimulationParameters& simParams,
        CheckpointManager& cm,
        const ftwa_su_n::FourierTransformer2dPBC& ft,
        unsigned int verbosity,
        double en_cutoff,
        bool use_cutoff
    );
    
    ~ODEHub2dMomPBC();
    
    void system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const;
    
    void observer(const arma::cx_vec& x, const double t);
    
  private:
    const Lattice& _lattice;
    const HubbardParameters& _params;
    const SimulationParameters& _simParams;
    CheckpointManager& _cm;
    const ftwa_su_n::FourierTransformer2dPBC& _ft;
    
#ifdef FTWA_CACHE_CHECKPOINTS
    arma::cx_cube _checkpoints;
#endif
    bool _fourierOutput;
    unsigned int _verbosity;
    
    arma::vec _en_diffs;
    double _en_cutoff;
    bool _use_cutoff;
};

class ODEHub2dHiePBC {
  public:
    ODEHub2dHiePBC(
        const SquareLattice& lattice,
        const HubbardParameters& params,
        const SimulationParameters& simParams,
        CheckpointManager& cm,
        const ftwa_su_n::FourierTransformer2dPBC& ft,
        unsigned int verbosity
    );
    
    ~ODEHub2dHiePBC();
    
    void system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const;
    
    void observer(const arma::cx_vec& x, const double t);
    
  private:
    const SquareLattice& _lattice;
    const HubbardParameters& _params;
    const SimulationParameters& _simParams;
    CheckpointManager& _cm;
    const ftwa_su_n::FourierTransformer2dPBC& _ft;
    
    bool _fourierOutput;
    unsigned int _verbosity;
};

}

#endif // SU_N_EOM_HUB_HPP
