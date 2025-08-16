#ifndef CHECKPOINT_MANAGER_HPP
#define CHECKPOINT_MANAGER_HPP

#include <string>

#include <armadillo>

#include "basic_defs.hpp"
#include "lattice.hpp"

/**
 * Storage class for observables evaluated during the simulation.
 * update functions are used to update the observables
 * with the current state vector after each of the TWA runs.
 */
class CheckpointManager {
  public:
    CheckpointManager(const Lattice& lattice, const std::string& runName) : write_flag(false), _lattice(lattice), _runName(runName) { }
    // virtual int init(unsigned int num_spins, const SimulationParameters& simParams, bool mom_space, bool with_cov, bool with_abs_offdiag, bool write_to_disk) = 0;
    
    // compatibility with sqlite
    virtual int beginTransaction() = 0;
    virtual int endTransaction() = 0;
    
    int initDensities(
        unsigned int num_spins,
        const SimulationParameters& simParams,
        bool mom_space,
        bool with_cov_densdens,
        bool with_cov_densdens_diag,
        bool write_to_file
    );
    
    int initOffdiag(
        unsigned int num_spins,
        const SimulationParameters& simParams,
        bool mom_space,
        bool with_cov,
        bool with_abs_offdiag,
        bool write_to_file
    );
    
    // only for su_n model
    int initBonds(const SimulationParameters& simParams, bool with_abs, bool with_abs2, bool write_to_file);
    int initFluxes(const SimulationParameters& simParams, bool write_to_file);
    int initModes(const SimulationParameters& simParams, bool write_to_file);
    
    // for su_n state vectors
    int updateDensities(const arma::cx_vec& vals, unsigned int i_t, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag);
    int updateOffdiag(const arma::cx_vec& vals, unsigned int i_t, bool mom_space, bool with_cov, bool with_abs_offdiag);
    int updateBonds(const arma::cx_vec& vals, unsigned int i_t, bool with_abs, bool with_abs2);
    int updateFluxes(const arma::cx_vec& vals, unsigned int i_t);
    int updateModes(const arma::cx_mat& vals, unsigned int i_t);
    
    // for checkpoint cubes
    int updateDensities(const arma::cx_cube& vals, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag);
    int updateOffdiag(const arma::cx_cube& vals, bool mom_space, bool with_cov, bool with_abs_offdiag);
    
    virtual int loadDensities(bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag) = 0;
    virtual int loadOffdiag(bool mom_space, bool with_cov, bool with_abs2) = 0;
    virtual int loadBonds(bool with_abs, bool with_abs2) = 0;
    virtual int loadFluxes() = 0;
    virtual int loadModes() = 0;
    
    virtual int writeDensities(bool initialize, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag) = 0;
    virtual int writeOffdiag(bool initialize, bool mom_space, bool with_cov, bool with_abs2) = 0;
    virtual int writeBonds(bool initialize, bool with_abs, bool with_abs2) = 0;
    virtual int writeFluxes(bool initialize) = 0;
    virtual int writeModes(bool initialize) = 0;
    
    bool write_flag;
    
  protected:
    const Lattice& _lattice;
    const std::string& _runName;
    
    arma::cube _rho_dens_i;
    arma::cube _rho_cov_densdens_ij;
    arma::cube _rho_cov_densdens_ii;
    arma::umat _n_samples_dens_i;
    arma::cx_cube _rho_offdiag_ij;
    arma::umat _n_samples_offdiag_ij;
    arma::cube _rho_abs_offdiag_ij;
    
    arma::cube _rho_dens_k;
    arma::cube _rho_cov_densdens_kl;
    arma::cube _rho_cov_densdens_kk;
    arma::umat _n_samples_dens_k;
    arma::cx_cube _rho_offdiag_kl;
    arma::umat _n_samples_offdiag_kl;
    arma::cube _rho_abs_offdiag_kl;
    
    arma::cx_mat _rho_bonds;
    arma::mat _rho_abs_bonds;
    arma::mat _rho_abs2_bonds;
    arma::mat _rho_bondbond_cell0;
    arma::uvec _n_samples_bonds;

    arma::cx_mat _rho_pi_fluxes_plaq;
    arma::mat _rho_abs_fluxes_plaq;
    arma::cx_mat _rho_phase_fluxes_plaq;
    arma::mat _rho_op_fluxes_plaq;
    arma::mat _rho_op_abs_fluxes_plaq;
    arma::mat _rho_op_squared_fluxes_plaq;
    arma::cx_mat _rho_pi_fluxes_vert;
    arma::mat _rho_abs_fluxes_vert;
    arma::cx_mat _rho_phase_fluxes_vert;
    arma::mat _rho_op_fluxes_vert;
    arma::mat _rho_op_abs_fluxes_vert;
    arma::mat _rho_op_squared_fluxes_vert;
    arma::uvec _n_samples_fluxes;
    
    arma::cx_cube _rho_k_diag_PM;
    arma::uvec _n_samples_k_diag_PM;
};

/**
 * CheckpointManager implementation for HDF5 output.
 */
class HDF5CheckpointManager : public CheckpointManager {
  public:
    HDF5CheckpointManager(const Lattice& lattice, const std::string& runName);
    
    ~HDF5CheckpointManager();
    
    // int init(unsigned int num_spins, const SimulationParameters& simParams, bool mom_space, bool with_cov, bool with_abs_offdiag, bool write_to_disk);
    
    // int load(const SimulationParameters& simParams, bool mom_space, bool with_cov, bool with_abs_offdiag);
    
    int beginTransaction();
    int endTransaction();
    
    int loadDensities(bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag);
    int loadOffdiag(bool mom_space, bool with_cov, bool with_abs2);
    int loadBonds(bool with_abs, bool with_abs2);
    int loadFluxes();
    int loadModes();
    
    // int updateDensities(const arma::cx_cube& vals, bool mom_space, bool with_cov);
    // int updateOffdiag(const arma::cx_cube& vals, bool mom_space, bool with_cov, bool with_abs_offdiag);
    
    int writeDensities(bool initialize, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag);
    int writeOffdiag(bool initialize, bool mom_space, bool with_cov, bool with_abs2);
    int writeBonds(bool initialize, bool with_abs, bool with_abs2);
    int writeFluxes(bool initialize);
    int writeModes(bool initialize);
};

#endif // CHECKPOINT_MANAGER_HPP
