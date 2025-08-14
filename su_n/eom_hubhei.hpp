#ifndef SU_N_EOM_HUBHEI_HPP
#define SU_N_EOM_HUBHEI_HPP

#ifdef FTWA_WITH_TIMER
#include <chrono>
#endif

#include <armadillo>

#include "../base/basic_defs.hpp"
#include "../base/lattice.hpp"
#include "../base/fourier.hpp"
#include "../base/checkpoint_manager.hpp"

namespace ftwa_su_n {

class ODEHubHei2dPBC {
  public:
    ODEHubHei2dPBC(
        const Lattice& lattice,
        const HubbardHeisenbergParameters& params,
        const SimulationParameters& simParams,
        const unitCellValues& ucv_tevol,
        CheckpointManager& cm,
        const FTSqrt2CellLattice& ft,
        double symmBreakStrength,
        bool mom_space,
        unsigned int verbosity
    );
    
    virtual ~ODEHubHei2dPBC();
    
    virtual void system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const;
    
    virtual void observer(const arma::cx_vec& x, const double t);
    
  protected:
    const Lattice& _lattice;
    HubbardHeisenbergParameters _params;
    SimulationParameters _simParams;
    unitCellValues _ucv_tevol;
    CheckpointManager& _cm;
    const FTSqrt2CellLattice& _ft;
    double _symmBreakStrength;
    
#ifdef FTWA_CACHE_CHECKPOINTS
    arma::cx_cube _checkpoints;
#endif
    bool _mom_space;
    unsigned int _verbosity;
    
    arma::umat _nns;
    arma::mat _symmBreakEps;
    
    //~ arma::uvec _indices_ai_j;
    //~ arma::uvec _indices_aj_i;
    //~ arma::vec _rho_conj_ai_j;
    //~ arma::vec _rho_conj_aj_i;
    //~ arma::uvec _indices_i;
    //~ arma::uvec _indices_j;
};

class ODEHubHei2dSwitchJPBC : public ODEHubHei2dPBC {
  public:
    ODEHubHei2dSwitchJPBC(
        const Lattice& lattice,
        const HubbardHeisenbergParameters& params,
        const SimulationParameters& simParams,
        const unitCellValues& _ucv_ini,
        CheckpointManager& cm,
        const FTSqrt2CellLattice& ft,
        double symmBreakStrength,
        bool mom_space,
        unsigned int verbosity,
        const HubbardHeisenbergParameters& params_fin,
        const unitCellValues& ucv_fin,
        double switch_time_start,
        double switch_time_end,
        int switch_order,
        double switch_symm_break_strength
    );
    
    ~ODEHubHei2dSwitchJPBC();
    
    void system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const override;
    
    // void observer(const arma::cx_vec& x, const double t);
    
  protected:
    HubbardHeisenbergParameters _params_fin;
    unitCellValues _ucv_fin;
    double _switch_time_start;
    double _switch_time_end;
    int _switch_order;
    double _switch_symm_break_strength;
};

class ODEHubHei2dPeierlsPBC : public ODEHubHei2dPBC {
  public:
    ODEHubHei2dPeierlsPBC(
        const Lattice& lattice,
        const HubbardHeisenbergParameters& params,
        const SimulationParameters& simParams,
        const unitCellValues& _ucv_tevol,
        CheckpointManager& cm,
        const FTSqrt2CellLattice& ft,
        double symmBreakStrength,
        bool mom_space,
        unsigned int verbosity,
        const PeierlsPulseParameters& pulseParams
    );
    
    ~ODEHubHei2dPeierlsPBC();
    
    void system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const override;
    
    // void observer(const arma::cx_vec& x, const double t);
    
  protected:
    PeierlsPulseParameters _pulseParams;
    arma::mat _peierlsExpArgs;
};


class ODEHubHei2dMomPeierlsPBC : public ODEHubHei2dPeierlsPBC {
  public:
    ODEHubHei2dMomPeierlsPBC(
        const Lattice& lattice,
        const HubbardHeisenbergParameters& params,
        const SimulationParameters& simParams,
        const unitCellValues& _ucv_tevol,
        CheckpointManager& cm,
        const FTSqrt2CellLattice& ft,
        bool mom_space,
        unsigned int verbosity,
        const PeierlsPulseParameters& pulseParams
    );
    
    ~ODEHubHei2dMomPeierlsPBC();
    
    void system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const override;
    
    void observer(const arma::cx_vec& x, const double t) override;
    
  protected:
    arma::cx_mat _bondFourierFactors;
};


}

#endif // SU_N_EOM_HUBHEI_HPP
