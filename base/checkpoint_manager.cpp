// checkpoint_manager.cpp

#include "checkpoint_manager.hpp"

HDF5CheckpointManager::HDF5CheckpointManager(const Lattice& lattice, const std::string& runName) : CheckpointManager(lattice, runName) { }

HDF5CheckpointManager::~HDF5CheckpointManager() { }

//~ int HDF5CheckpointManager::init(unsigned int num_spins, const SimulationParameters& simParams, bool mom_space, bool with_cov, bool with_abs_offdiag, bool write_to_file) {
    //~ unsigned int V = _lattice.numSites(), rhoLen = _lattice.rhoLen();
    //~ unsigned int num_i_t = simParams.num_tsteps + 1;
    //~ unsigned int num_spin_confs = num_spins*num_spins;
    
    //~ arma::cube& rho_dens = mom_space ? this->_rho_dens_k : this->_rho_dens_i;
    //~ arma::cube& rho_cov_densdens = mom_space ? this->_rho_cov_densdens_kl : this->_rho_cov_densdens_ij;
    //~ arma::umat& n_samples_dens = mom_space ? this->_n_samples_dens_k : this->_n_samples_dens_i;
    //~ arma::cx_cube& rho_offdiag = mom_space ? this->_rho_offdiag_kl : this->_rho_offdiag_ij;
    //~ arma::umat& n_samples_offdiag = mom_space ? this->_n_samples_offdiag_kl : this->_n_samples_offdiag_ij;
    //~ arma::cube& rho_abs_offdiag = mom_space ? this->_rho_abs_offdiag_kl : this->_rho_abs_offdiag_ij;
        
    //~ rho_dens = arma::zeros<arma::cube>(V, num_spins, num_i_t);
    //~ n_samples_dens = arma::zeros<arma::umat>(num_i_t, num_spins);
    //~ if (with_cov) {
        //~ rho_cov_densdens = arma::zeros<arma::cube>(rhoLen, num_spin_confs, num_i_t);
    //~ }
    
    //~ rho_offdiag = arma::zeros<arma::cx_cube>(rhoLen, num_spin_confs, num_i_t);
    //~ n_samples_offdiag = arma::zeros<arma::umat>(num_i_t, num_spin_confs);
    //~ if (with_abs_offdiag) {
        //~ rho_abs_offdiag = arma::zeros<arma::cube>(rhoLen, num_spin_confs, num_i_t);
    //~ }
        
    //~ if (write_to_file) {
        //~ this->writeDensities(true, mom_space, with_cov);
        //~ this->writeOffdiag(true, mom_space, with_cov, with_abs_offdiag);
    //~ }
    
    //~ return STATUS_OKAY;
//~ }

int CheckpointManager::initDensities(unsigned int num_spins, const SimulationParameters& simParams, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag, bool write_to_file) {
    unsigned int V = _lattice.numSites(), rhoLen = _lattice.rhoLen();
    unsigned int num_i_t = simParams.num_tsteps + 1;
    unsigned int num_spin_confs = num_spins*num_spins;
    
    arma::cube& rho_dens = mom_space ? this->_rho_dens_k : this->_rho_dens_i;
    arma::cube& rho_cov_densdens = mom_space ? this->_rho_cov_densdens_kl : this->_rho_cov_densdens_ij;
    arma::cube& rho_cov_densdens_diag = mom_space ? this->_rho_cov_densdens_kk : this->_rho_cov_densdens_ii;
    arma::umat& n_samples_dens = mom_space ? this->_n_samples_dens_k : this->_n_samples_dens_i;
        
    rho_dens = arma::zeros<arma::cube>(V, num_spins, num_i_t);
    n_samples_dens = arma::zeros<arma::umat>(num_i_t, num_spins);
    
    if (with_cov_densdens) {
        rho_cov_densdens = arma::zeros<arma::cube>(rhoLen, num_spin_confs, num_i_t);
    }
    
    if (with_cov_densdens_diag) {
        rho_cov_densdens_diag = arma::zeros<arma::cube>(V, num_spin_confs, num_i_t);
    }
        
    if (write_to_file) {
        this->writeDensities(true, mom_space, with_cov_densdens, with_cov_densdens_diag);
    }
    
    return STATUS_OKAY;
}

int CheckpointManager::initOffdiag(unsigned int num_spins, const SimulationParameters& simParams, bool mom_space, bool with_cov, bool with_abs2, bool write_to_file) {
    unsigned int rhoLen = _lattice.rhoLen();
    unsigned int num_i_t = simParams.num_tsteps + 1;
    unsigned int num_spin_confs = num_spins*num_spins;
    
    arma::cx_cube& rho_offdiag = mom_space ? this->_rho_offdiag_kl : this->_rho_offdiag_ij;
    arma::umat& n_samples_offdiag = mom_space ? this->_n_samples_offdiag_kl : this->_n_samples_offdiag_ij;
    arma::cube& rho_abs_offdiag = mom_space ? this->_rho_abs_offdiag_kl : this->_rho_abs_offdiag_ij;
    
    rho_offdiag = arma::zeros<arma::cx_cube>(rhoLen, num_spin_confs, num_i_t);
    n_samples_offdiag = arma::zeros<arma::umat>(num_i_t, num_spin_confs);
    if (with_abs2) {
        rho_abs_offdiag = arma::zeros<arma::cube>(rhoLen, num_spin_confs, num_i_t);
    }
        
    if (write_to_file) {
        this->writeOffdiag(true, mom_space, with_cov, with_abs2);
    }
    
    return STATUS_OKAY;
}

int CheckpointManager::initBonds(const SimulationParameters& simParams, bool with_abs, bool with_abs2, bool write_to_file) {
    unsigned int uV = _lattice.numCells();
    unsigned int num_i_t = simParams.num_tsteps + 1;
    
    _rho_bonds = arma::zeros<arma::cx_mat>(4*uV, num_i_t);
    _n_samples_bonds = arma::zeros<arma::uvec>(num_i_t);
    if (with_abs) {
        _rho_abs_bonds = arma::zeros<arma::mat>(4*uV, num_i_t);
    }
    if (with_abs2) {
        _rho_abs2_bonds = arma::zeros<arma::mat>(4*uV, num_i_t);
    }
    
    _rho_bondbond_cell0 = arma::zeros<arma::mat>(4*uV, num_i_t);
        
    if (write_to_file) {
        this->writeBonds(true, with_abs, with_abs2);
    }
    
    return STATUS_OKAY;
}

int CheckpointManager::initFluxes(const SimulationParameters& simParams, bool write_to_file) {
    unsigned int V = _lattice.numSites();
    unsigned int num_i_t = simParams.num_tsteps + 1;
    
    _rho_pi_fluxes_plaq = arma::zeros<arma::cx_mat>(V, num_i_t);
    _rho_abs_fluxes_plaq = arma::zeros<arma::mat>(V, num_i_t);
    _rho_phase_fluxes_plaq = arma::zeros<arma::cx_mat>(V, num_i_t);
    _rho_op_fluxes_plaq = arma::zeros<arma::mat>(V, num_i_t);
    _rho_op_abs_fluxes_plaq = arma::zeros<arma::mat>(V, num_i_t);
    _rho_op_squared_fluxes_plaq = arma::zeros<arma::mat>(V, num_i_t);
    
    _rho_pi_fluxes_vert = arma::zeros<arma::cx_mat>(V, num_i_t);
    _rho_abs_fluxes_vert = arma::zeros<arma::mat>(V, num_i_t);
    _rho_phase_fluxes_vert = arma::zeros<arma::cx_mat>(V, num_i_t);
    _rho_op_fluxes_vert = arma::zeros<arma::mat>(V, num_i_t);
    _rho_op_abs_fluxes_vert = arma::zeros<arma::mat>(V, num_i_t);
    _rho_op_squared_fluxes_vert = arma::zeros<arma::mat>(V, num_i_t);
    
    _n_samples_fluxes = arma::zeros<arma::uvec>(num_i_t);
        
    if (write_to_file) {
        this->writeFluxes(true);
    }
    
    return STATUS_OKAY;
}

int CheckpointManager::initModes(const SimulationParameters& simParams, bool write_to_file) {
    unsigned int uV = _lattice.numCells();
    unsigned int num_i_t = simParams.num_tsteps + 1;
    
    _rho_k_diag_PM = arma::zeros<arma::cx_cube>(uV, 4, num_i_t);
    _n_samples_k_diag_PM = arma::zeros<arma::uvec>(num_i_t);
        
    if (write_to_file) {
        this->writeModes(true);
    }
    
    return STATUS_OKAY;
}

int HDF5CheckpointManager::beginTransaction() {
    return STATUS_OKAY;
}

int HDF5CheckpointManager::endTransaction() {
    return STATUS_OKAY;
}

int CheckpointManager::updateDensities(const arma::cx_vec& vals, unsigned int i_t, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag) {
    arma::cube& rho_dens = mom_space ? this->_rho_dens_k : this->_rho_dens_i;
    arma::cube& rho_cov_densdens = mom_space ? this->_rho_cov_densdens_kl : this->_rho_cov_densdens_ij;
    arma::cube& rho_cov_densdens_diag = mom_space ? this->_rho_cov_densdens_kk : this->_rho_cov_densdens_ii;
    arma::umat& n_samples = mom_space ? this->_n_samples_dens_k : this->_n_samples_dens_i;
    
    unsigned int V = _lattice.numSites();
    
    double n = (double) n_samples(i_t, 0);
    double v, w;
    
    for (unsigned int i = 0; i < V; ++i) {
        v = vals(ftwa_index(i, i)).real() - rho_dens(i, 0, i_t);
        
        if(with_cov_densdens) {
            for (unsigned int j = i; j < V; ++j) {
                w = vals(ftwa_index(j, j)).real() - rho_dens(j, 0, i_t);
                
                rho_cov_densdens(ftwa_index(i, j), 0, i_t) = 1.0/(1.0+n) * (n * rho_cov_densdens(ftwa_index(i, j), 0, i_t) + n/(1.0+n) * v * w);
            }
        }
        
        if(with_cov_densdens_diag) {
            rho_cov_densdens_diag(i, 0, i_t) = 1.0/(1.0+n) * (n * rho_cov_densdens_diag(i, 0, i_t) + n/(1.0+n) * v * v);
        }
        
        rho_dens(i, 0, i_t) += 1.0/(1.0+n) * v;
    }
    n_samples(i_t, 0)++;
            
    return STATUS_OKAY;
}

int CheckpointManager::updateOffdiag(const arma::cx_vec& vals, unsigned int i_t, bool mom_space, bool with_cov, bool with_abs2) {
    arma::cx_cube& rho_offdiag = mom_space ? this->_rho_offdiag_kl : this->_rho_offdiag_ij;
    arma::umat& n_samples = mom_space ? this->_n_samples_offdiag_kl : this->_n_samples_offdiag_ij;
    arma::cube& rho_abs_offdiag = mom_space ? this->_rho_abs_offdiag_kl : this->_rho_abs_offdiag_ij;
    
    unsigned int V = _lattice.numSites();
    
    double n = (double) n_samples(i_t, 0);
    std::complex<double> v;
    
    for (unsigned int i = 0; i < V; ++i) {
        for (unsigned int j = i+1; j < V; ++j) {
            v = vals(ftwa_index(i, j)) - rho_offdiag(ftwa_index(i, j), 0, i_t);
            
            rho_offdiag(ftwa_index(i, j), 0, i_t) += 1.0/(1.0+n) * v;
            
            if (with_abs2) {
                rho_abs_offdiag(ftwa_index(i, j), 0, i_t) += 1.0/(1.0+n) * (std::norm(vals(ftwa_index(i, j))) - rho_abs_offdiag(ftwa_index(i, j), 0, i_t));
            }
        }
    }
    n_samples(i_t, 0)++;
    
    return STATUS_OKAY;
}

int CheckpointManager::updateDensities(const arma::cx_cube& vals, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag) {
    arma::cube& rho_dens = mom_space ? this->_rho_dens_k : this->_rho_dens_i;
    arma::cube& rho_cov_densdens = mom_space ? this->_rho_cov_densdens_kl : this->_rho_cov_densdens_ij;
    // arma::cube& rho_cov_densdens_diag = mom_space ? this->_rho_cov_densdens_kk : this->_rho_cov_densdens_ii;
    arma::umat& n_samples_dens = mom_space ? this->_n_samples_dens_k : this->_n_samples_dens_i;
    
    unsigned int V = _lattice.numSites();
    unsigned int num_spins = n_samples_dens.n_cols;
    
    arma::cube v(1, num_spins, vals.n_slices), w(1, num_spins, vals.n_slices);
    arma::cube n_fact(1, n_samples_dens.n_cols, n_samples_dens.n_rows);
    arma::cube inv_n_fact(1, vals.n_cols, vals.n_slices);
    arma::cube n_inv_n_fact(1, vals.n_cols, vals.n_slices);
    
    n_fact.row(0) = arma::conv_to<arma::mat>::from(n_samples_dens.t());
    inv_n_fact = 1.0/(n_fact + 1.0);
    n_inv_n_fact = n_fact / (n_fact + 1.0);
    
    for (unsigned int i = 0; i < V; ++i) {
        for (unsigned int spin = 0; spin < num_spins; ++spin) {
            v.col(spin) = arma::real(vals.tube(ftwa_index(i, i), (spin+1)*(spin+1)-1)) - rho_dens.tube(i, spin);
        }
        
        if(with_cov_densdens) {
            for (unsigned int j = i; j < V; ++j) {
                for (unsigned int spin = 0; spin < num_spins; ++spin) {
                    w.col(spin) = arma::real(vals.tube(ftwa_index(j, j), (spin+1)*(spin+1)-1)) - rho_dens.tube(j, spin);
                }
                
                for (unsigned int spin2 = 0; spin2 < num_spins; ++spin2) {
                    for (unsigned int spin1 = 0; spin1 < num_spins; ++spin1) {
                        rho_cov_densdens.tube(ftwa_index(i, j), spin2*num_spins + spin1) = inv_n_fact.col(spin1) % (n_fact.col(spin1) % rho_cov_densdens.tube(ftwa_index(i, j), spin2*num_spins + spin1) + n_inv_n_fact.col(spin1) % v.col(spin1) % w.col(spin2));
                    }
                }
            }
        }
        
        if (with_cov_densdens_diag) {
            // TODO
        }
        
        rho_dens.row(i) += (inv_n_fact % v);
    }
    n_samples_dens = n_samples_dens + 1;
            
    return STATUS_OKAY;
}

int CheckpointManager::updateOffdiag(const arma::cx_cube& vals, bool mom_space, bool with_cov, bool with_abs2) {
    arma::cx_cube& rho_offdiag = mom_space ? this->_rho_offdiag_kl : this->_rho_offdiag_ij;
    arma::umat& n_samples = mom_space ? this->_n_samples_offdiag_kl : this->_n_samples_offdiag_ij;
    arma::cube& rho_abs_offdiag = mom_space ? this->_rho_abs_offdiag_kl : this->_rho_abs_offdiag_ij;
    
    unsigned int V = _lattice.numSites();
    
    arma::cube n_fact(1, n_samples.n_cols, n_samples.n_rows);
    arma::cube inv_n_fact(1, vals.n_cols, vals.n_slices);
    
    n_fact.row(0) = arma::conv_to<arma::mat>::from(n_samples.t());
    inv_n_fact = 1.0/(n_fact + 1.0);
    
    for (unsigned int i = 0; i < V; ++i) {
        // if rho_{k up, k down} shall be included: i <= j
        for (unsigned int j = i+1; j < V; ++j) {
            rho_offdiag.row(ftwa_index(i, j)) = rho_offdiag.row(ftwa_index(i, j)) + inv_n_fact % (vals.row(ftwa_index(i, j)) - rho_offdiag.row(ftwa_index(i, j)));
            
            if (with_abs2) {
                rho_abs_offdiag.row(ftwa_index(i, j)) = rho_abs_offdiag.row(ftwa_index(i, j)) + inv_n_fact % (arma::square(arma::abs(vals.row(ftwa_index(i, j)))) - rho_abs_offdiag.row(ftwa_index(i, j)));
            }
        }
    }
    n_samples = n_samples + 1;
    
    return STATUS_OKAY;
}

int CheckpointManager::updateBonds(const arma::cx_vec& vals, unsigned int i_t, bool with_abs, bool with_abs2) {
    unsigned int L = _lattice.getLength(0), uV = _lattice.numCells();
    
    double n = (double) _n_samples_bonds(i_t), r, s;
    unsigned int nns[4], i, i_c, d, d_c, i_p_d;
    std::complex<double> v;
    
    // bonds for bond-bond correlation
    arma::cx_mat bonds = arma::zeros<arma::cx_mat>(uV, 4);
    for (unsigned int iy = 0; iy < L; ++iy) {
        for (unsigned int ix = 0; ix < L; ++ix) {
            i = 2*L*iy + L + ix;
            i_c = L*iy + ix;
            hubhei_sqrt2cell_neighbors(L, i, nns);
            for (unsigned int a = 0; a < 4; ++a) {
                bonds(i_c, a) = (a > 1) ? (
                    (nns[a] > i) ? vals(ftwa_index(i, nns[a])) : std::conj(vals(ftwa_index(nns[a], i)))
                ) : vals(ftwa_index(nns[a], i));
            }
        }
    }
    
    // need to loop only over the B sites
    for (unsigned int iy = 0; iy < L; ++iy) {
        for (unsigned int ix = 0; ix < L; ++ix) {
            i_c = L*iy + ix;
            for (unsigned int a = 0; a < 4; ++a) {
                v = bonds(i_c, a) - _rho_bonds(a*uV + i_c, i_t);
                
                _rho_bonds(a*uV + i_c, i_t) += 1.0/(1.0+n) * v;
                
                if (with_abs) {
                    r = std::abs(bonds(i_c, a)) - _rho_abs_bonds(a*uV + i_c, i_t);
                    _rho_abs_bonds(a*uV + i_c, i_t) += 1.0/(1.0+n) * r;
                }
                if (with_abs2) {
                    r = std::norm(bonds(i_c, a)) - _rho_abs2_bonds(a*uV + i_c, i_t);
                    _rho_abs2_bonds(a*uV + i_c, i_t) += 1.0/(1.0+n) * r;
                }
                
                s = 0.0;
                // correlations 0+d with i+d
                for (unsigned int dy = 0; dy < L; ++dy) {
                    for (unsigned int dx = 0; dx < L; ++dx) {
                        d_c   = L*dy + dx;
                        i_p_d = L*((iy+dy)%L) + ((ix+dx)%L);
                        
                        s += std::imag(bonds(d_c, a))*std::imag(bonds(i_p_d, a));
                    }
                }
                
                // r = _rho_bondbond_cell0(a*uV + i_c, i_t) - 4.0/uV * s;
                r = 4.0/uV * s - _rho_bondbond_cell0(a*uV + i_c, i_t);
                _rho_bondbond_cell0(a*uV + i_c, i_t) += 1.0/(1.0+n) * r;
            }
        }
    }
    _n_samples_bonds(i_t)++;
    
    return STATUS_OKAY;
}

int CheckpointManager::updateFluxes(const arma::cx_vec& vals, unsigned int i_t) {
    unsigned int L = _lattice.getLength(0);
    
    double n = (double) _n_samples_fluxes(i_t), op;
    unsigned int nns_i[4], nns_i_r[4], i, i_r;
    std::complex<double> bond_tl, bond_tr, bond_br, bond_bl, flux, bond0, bond3;
    
    // site coordinates, not unit cells
    for (unsigned int iy = 0; iy < 2*L; ++iy) {
        for (unsigned int ix = 0; ix < L; ++ix) {
            i = iy*L + ix;
            hubhei_sqrt2cell_neighbors(L, i, nns_i);
            
            // flux around the plaquettes
            // to the right of i and to the left of i_r
            
            i_r = iy*L + ((ix+1) % L);      
            hubhei_sqrt2cell_neighbors(L, i_r, nns_i_r);
            
            bond_tl = (nns_i[1] < i) ? vals(ftwa_index(nns_i[1], i)) : std::conj(vals(ftwa_index(i, nns_i[1])));
            bond_tr = (nns_i_r[0] < i_r) ? vals(ftwa_index(nns_i_r[0], i_r)) : std::conj(vals(ftwa_index(i_r, nns_i_r[0])));
            bond_bl = (nns_i[2] > i) ? vals(ftwa_index(i, nns_i[2])) : std::conj(vals(ftwa_index(nns_i[2], i)));
            bond_br = (nns_i_r[3] > i_r) ? vals(ftwa_index(i_r, nns_i_r[3])) : std::conj(vals(ftwa_index(nns_i_r[3], i_r)));
                
            flux = bond_tr * bond_br * std::conj(bond_bl) * std::conj(bond_tl);
            if (iy % 2 == 0) flux = std::conj(flux);
            op = std::arg(flux); // should reside in interval [-pi, pi]
            
            _rho_pi_fluxes_plaq(i, i_t) += 1.0/(1.0+n) * (flux - _rho_pi_fluxes_plaq(i, i_t));
            _rho_abs_fluxes_plaq(i, i_t) += 1.0/(1.0+n) * (std::abs(flux) - _rho_abs_fluxes_plaq(i, i_t));
            _rho_phase_fluxes_plaq(i, i_t) += 1.0/(1.0+n) * (flux/std::abs(flux) - _rho_phase_fluxes_plaq(i, i_t));
            
            _rho_op_fluxes_plaq(i, i_t) += 1.0/(1.0+n) * (op - _rho_op_fluxes_plaq(i, i_t));
            _rho_op_abs_fluxes_plaq(i, i_t) += 1.0/(1.0+n) * (std::abs(op) - _rho_op_abs_fluxes_plaq(i, i_t));
            _rho_op_squared_fluxes_plaq(i, i_t) += 1.0/(1.0+n) * (op*op - _rho_op_squared_fluxes_plaq(i, i_t));
            
            // flux at the vertices/sites, bond_tl, bond_bl can be used again
            bond0 = (nns_i[0] < i) ? vals(ftwa_index(nns_i[0], i)) : std::conj(vals(ftwa_index(i, nns_i[0])));
            bond3 = (nns_i[3] > i) ? vals(ftwa_index(i, nns_i[3])) : std::conj(vals(ftwa_index(nns_i[3], i)));
            
            flux = bond0 * std::conj(bond_tl) * std::conj(bond_bl) * bond3;
            if (iy % 2 == 0) flux = std::conj(flux);
            op = std::arg(flux);
            
            _rho_pi_fluxes_vert(i, i_t) += 1.0/(1.0+n) * (flux - _rho_pi_fluxes_vert(i, i_t));
            _rho_abs_fluxes_vert(i, i_t) += 1.0/(1.0+n) * (std::abs(flux) - _rho_abs_fluxes_vert(i, i_t));
            _rho_phase_fluxes_vert(i, i_t) += 1.0/(1.0+n) * (flux/std::abs(flux) - _rho_phase_fluxes_vert(i, i_t));
            
            _rho_op_fluxes_vert(i, i_t) += 1.0/(1.0+n) * (op - _rho_op_fluxes_vert(i, i_t));
            _rho_op_abs_fluxes_vert(i, i_t) += 1.0/(1.0+n) * (std::abs(op) - _rho_op_abs_fluxes_vert(i, i_t));
            _rho_op_squared_fluxes_vert(i, i_t) += 1.0/(1.0+n) * (op*op - _rho_op_squared_fluxes_vert(i, i_t));
        }
    }
    _n_samples_fluxes(i_t)++;
    
    return STATUS_OKAY;
}

int CheckpointManager::updateModes(const arma::cx_mat& vals, unsigned int i_t) {
    arma::cx_cube& rho_k_diag_PM = this->_rho_k_diag_PM;
    arma::uvec& n_samples        = this->_n_samples_k_diag_PM;
    
    double n = (double) n_samples(i_t);
    
    rho_k_diag_PM.slice(i_t) += 1.0/(1.0+n) * (vals - rho_k_diag_PM.slice(i_t));
    
    n_samples(i_t)++;
            
    return STATUS_OKAY;
}

// in the future: save whole matrices, then load needs no init before
int HDF5CheckpointManager::loadDensities(bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag) {
    arma::cube& rho_dens = mom_space ? this->_rho_dens_k : this->_rho_dens_i;
    arma::cube& rho_cov_densdens = mom_space ? this->_rho_cov_densdens_kl : this->_rho_cov_densdens_ij;
    arma::cube& rho_cov_densdens_diag = mom_space ? this->_rho_cov_densdens_kk : this->_rho_cov_densdens_ii;
    arma::umat& n_samples_dens = mom_space ? this->_n_samples_dens_k : this->_n_samples_dens_i;
    
    std::string rho_dens_table_name = mom_space ? std::string("rho_dens_k") : std::string("rho_dens_i");
    std::string rho_cov_densdens_table_name = mom_space ? std::string("rho_cov_densdens_kl") : std::string("rho_cov_densdens_ij");
    std::string rho_cov_densdens_diag_table_name = mom_space ? std::string("rho_cov_densdens_kk") : std::string("rho_cov_densdens_ii");
    
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    // unsigned int num_i_t = rho_dens.n_slices;
    unsigned int num_spins = rho_dens.n_cols;
    unsigned int num_spin_confs = num_spins * num_spins;
    
    for (unsigned int spin = 0; spin < num_spins; ++spin) {
        n_samples_dens.unsafe_col(spin).load(
            arma::hdf5_name(filename,
                rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                              + sep + std::string("n_samples")));
    }

    arma::mat t;
    for (unsigned int spin = 0; spin < num_spins; ++spin) {
        t.load(arma::hdf5_name(filename,
            rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                + sep + std::string("data")));
        rho_dens.col(spin) = t;
    }
    for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
        if (with_cov_densdens) {
            t.load(
                arma::hdf5_name(filename,
                rho_cov_densdens_table_name + sep + std::to_string(spin_conf)
                    + sep + std::string("data"))
            );
            rho_cov_densdens.col(spin_conf) = t;
        }
        
        if (with_cov_densdens_diag) {
            t.load(
                arma::hdf5_name(filename,
                rho_cov_densdens_diag_table_name + sep + std::to_string(spin_conf)
                    + sep + std::string("data"))
            );
            rho_cov_densdens_diag.col(spin_conf) = t;
        }
    }
    //~ for (unsigned int i_t = 0; i_t < num_i_t; ++i_t) {
        //~ for (unsigned int spin = 0; spin < num_spins; ++spin) {
            //~ rho_dens.slice(i_t).unsafe_col(spin).load(
                //~ arma::hdf5_name(filename,
                //~ rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                              //~ + sep + std::to_string(i_t)));
        //~ }
        
        //~ if (with_cov_densdens) {
            //~ for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
                //~ rho_cov_densdens.slice(i_t).unsafe_col(spin_conf).load(
                    //~ arma::hdf5_name(filename,
                    //~ rho_cov_densdens_table_name + sep + std::to_string(spin_conf)
                                 //~ + sep + std::to_string(i_t)));
            //~ }
        //~ }
        
        //~ if (with_cov_densdens_diag) {
            //~ for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
                //~ rho_cov_densdens_diag.slice(i_t).unsafe_col(spin_conf).load(
                    //~ arma::hdf5_name(filename,
                    //~ rho_cov_densdens_diag_table_name + sep + std::to_string(spin_conf)
                                 //~ + sep + std::to_string(i_t)));
            //~ }
        //~ }
    //~ }
    
    return STATUS_OKAY;
}

// in the future: save whole matrices, then load needs no init before
int HDF5CheckpointManager::loadOffdiag(bool mom_space, bool with_cov, bool with_abs2) {
    arma::cx_cube& rho_offdiag = mom_space ? this->_rho_offdiag_kl : this->_rho_offdiag_ij;
    arma::umat& n_samples_offdiag = mom_space ? this->_n_samples_offdiag_kl : this->_n_samples_offdiag_ij;
    arma::cube& rho_abs_offdiag = mom_space ? this->_rho_abs_offdiag_kl : this->_rho_abs_offdiag_ij;
    
    std::string offdiagTableName = mom_space ? std::string("rho_offdiag_kl") : std::string("rho_offdiag_ij");
    std::string absOffdiagTableName = mom_space ? std::string("rho_abs_offdiag_kl") : std::string("rho_abs_offdiag_ij");
    
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    unsigned int num_i_t = rho_offdiag.n_slices;
    unsigned int num_spins = rho_offdiag.n_cols;
    unsigned int num_spin_confs = num_spins * num_spins;
    
    for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
        n_samples_offdiag.unsafe_col(spin_conf).load(arma::hdf5_name(filename,
            offdiagTableName + sep + std::to_string(spin_conf) + sep + std::string("n_samples")));
    }

    for (unsigned int i_t = 0; i_t < num_i_t; ++i_t) {
        for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
            rho_offdiag.slice(i_t).unsafe_col(spin_conf).load(
                arma::hdf5_name(filename,
                offdiagTableName + sep + std::to_string(spin_conf)
                                 + sep + std::to_string(i_t)));
            if (with_abs2) {
                rho_abs_offdiag.slice(i_t).unsafe_col(spin_conf).load(
                    arma::hdf5_name(filename,
                    absOffdiagTableName + sep + std::to_string(spin_conf)
                                        + sep + std::to_string(i_t)));
            }
        }
    }
    return STATUS_OKAY;
}

int HDF5CheckpointManager::loadBonds(bool with_abs, bool with_abs2) {
    std::string bondsTableName = std::string("rho_bonds");
    
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    _n_samples_bonds.load(arma::hdf5_name(filename,
        bondsTableName + sep + std::string("n_samples")));

    _rho_bonds.load(arma::hdf5_name(filename,
        bondsTableName + sep + std::string("bonds")));
        
    if (with_abs) {
        _rho_abs_bonds.load(
            arma::hdf5_name(filename,
            bondsTableName + sep + std::string("abs_bonds")));
    }
    if (with_abs2) {
        _rho_abs2_bonds.load(
            arma::hdf5_name(filename,
            bondsTableName + sep + std::string("abs2_bonds")));
    }
    
    _rho_bondbond_cell0.load(arma::hdf5_name(filename,
            bondsTableName + sep + std::string("bondbond_cell0")));

    return STATUS_OKAY;
}

int HDF5CheckpointManager::loadFluxes() {
    std::string fluxesTableName = std::string("rho_fluxes");
    
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    _n_samples_fluxes.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("n_samples")));
    
    _rho_pi_fluxes_plaq.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("pi_fluxes_plaq")));
    
    _rho_abs_fluxes_plaq.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("abs_fluxes_plaq")));
        
    _rho_phase_fluxes_plaq.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("phase_fluxes_plaq")));
    
    _rho_op_fluxes_plaq.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("op_fluxes_plaq")));

    _rho_op_abs_fluxes_plaq.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("op_abs_fluxes_plaq")));
    
    _rho_op_squared_fluxes_plaq.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("op_squared_fluxes_plaq")));
    
    // #################################################################
    
    _rho_pi_fluxes_vert.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("pi_fluxes_vert")));
    
    _rho_abs_fluxes_vert.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("abs_fluxes_vert")));
        
    _rho_phase_fluxes_vert.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("phase_fluxes_vert")));
    
    _rho_op_fluxes_vert.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("op_fluxes_vert")));

    _rho_op_abs_fluxes_vert.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("op_abs_fluxes_vert")));
    
    _rho_op_squared_fluxes_vert.load(arma::hdf5_name(filename,
        fluxesTableName + sep + std::string("op_squared_fluxes_vert")));

    return STATUS_OKAY;
}

int HDF5CheckpointManager::loadModes() {
    std::string tableName = std::string("rho_modes");
    
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    _n_samples_k_diag_PM.load(arma::hdf5_name(filename,
        tableName + sep + std::string("n_samples")));
    
    _rho_k_diag_PM.load(arma::hdf5_name(filename,
        tableName + sep + std::string("k_diag_PM")));

    return STATUS_OKAY;
}

int HDF5CheckpointManager::writeDensities(bool initialize, bool mom_space, bool with_cov_densdens, bool with_cov_densdens_diag) {
    std::string sep("/");
    std::string filename = this->_runName + std::string(".h5");
    
    std::string rho_dens_table_name = mom_space ? std::string("rho_dens_k") : std::string("rho_dens_i");
    std::string rho_cov_densdens_table_name = mom_space ? std::string("rho_cov_densdens_kl") : std::string("rho_cov_densdens_ij");
    std::string rho_cov_densdens_diag_table_name = mom_space ? std::string("rho_cov_densdens_kk") : std::string("rho_cov_densdens_ii");
    
    arma::cube& rho_dens = mom_space ? this->_rho_dens_k : this->_rho_dens_i;
    arma::cube& rho_cov_densdens = mom_space ? this->_rho_cov_densdens_kl : this->_rho_cov_densdens_ij;
    arma::cube& rho_cov_densdens_diag = mom_space ? this->_rho_cov_densdens_kk : this->_rho_cov_densdens_ii;
    arma::umat& n_samples_dens = mom_space ? this->_n_samples_dens_k : this->_n_samples_dens_i;
    
    unsigned int num_spins = n_samples_dens.n_cols;
    unsigned int num_spin_confs = num_spins * num_spins;
    
    for (unsigned int spin = 0; spin < num_spins; ++spin) {
        if (initialize) {
            n_samples_dens.unsafe_col(spin).save(
                arma::hdf5_name(filename,
                rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                              + sep + std::string("n_samples"),
                arma::hdf5_opts::append));
        } else {
            n_samples_dens.unsafe_col(spin).save(
                arma::hdf5_name(filename,
                rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                              + sep + std::string("n_samples"),
                arma::hdf5_opts::replace));
        }
        
        arma::mat t = rho_dens.col(spin);
        if (initialize) {
            t.save(arma::hdf5_name(filename,
                rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                              + sep + std::string("data"),
                arma::hdf5_opts::append));
        } else {
            t.save(arma::hdf5_name(filename,
                rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                              + sep + std::string("data"),
                arma::hdf5_opts::replace));
        }
        //~ for (unsigned int i_t = 0; i_t < rho_dens.n_slices; ++i_t) {
            //~ if (initialize) {
                //~ rho_dens.slice(i_t).unsafe_col(spin).save(
                    //~ arma::hdf5_name(filename,
                    //~ rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                                  //~ + sep + std::to_string(i_t),
                    //~ arma::hdf5_opts::append));
            //~ } else {
                //~ rho_dens.slice(i_t).unsafe_col(spin).save(
                    //~ arma::hdf5_name(filename,
                    //~ rho_dens_table_name + sep + std::to_string((spin+1)*(spin+1)-1)
                                  //~ + sep + std::to_string(i_t),
                    //~ arma::hdf5_opts::replace));
            //~ }
        //~ }
    }
    
    if (with_cov_densdens) {
        for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
            arma::mat t = rho_cov_densdens.col(spin_conf);
            if (initialize) {
                t.save(arma::hdf5_name(filename,
                    rho_cov_densdens_table_name + sep + std::to_string(spin_conf) + sep + std::string("data"), arma::hdf5_opts::append));
            } else {
                t.save(arma::hdf5_name(filename,
                    rho_cov_densdens_table_name + sep + std::to_string(spin_conf) + sep + std::string("data"), arma::hdf5_opts::replace));
            }
        }
        //~ for (unsigned int i_t = 0; i_t < rho_cov_densdens.n_slices; ++i_t) {
            //~ for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
                //~ if (initialize) {
                    //~ rho_cov_densdens.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                        //~ rho_cov_densdens_table_name + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::append));
                //~ } else {
                    //~ rho_cov_densdens.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                        //~ rho_cov_densdens_table_name + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::replace));
                //~ }
            //~ }
        //~ }
    }
    
    if (with_cov_densdens_diag) {
        for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
            arma::mat t = rho_cov_densdens_diag.col(spin_conf);
            if (initialize) {
                t.save(arma::hdf5_name(filename,
                    rho_cov_densdens_diag_table_name + sep + std::to_string(spin_conf) + sep + std::string("data"), arma::hdf5_opts::append));
            } else {
                t.save(arma::hdf5_name(filename,
                    rho_cov_densdens_diag_table_name + sep + std::to_string(spin_conf) + sep + std::string("data"), arma::hdf5_opts::replace));
            }
        }
        //~ for (unsigned int i_t = 0; i_t < rho_cov_densdens_diag.n_slices; ++i_t) {
            //~ for (unsigned int spin_conf = 0; spin_conf < num_spin_confs; ++spin_conf) {
                //~ if (initialize) {
                    //~ rho_cov_densdens_diag.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                        //~ rho_cov_densdens_diag_table_name + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::append));
                //~ } else {
                    //~ rho_cov_densdens_diag.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                        //~ rho_cov_densdens_diag_table_name + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::replace));
                //~ }
            //~ }
        //~ }
    }
    
    return STATUS_OKAY;
}

int HDF5CheckpointManager::writeOffdiag(bool initialize, bool mom_space, bool with_cov, bool with_abs2) {
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    std::string offdiagTableName = mom_space ? std::string("rho_offdiag_kl") : std::string("rho_offdiag_ij");
    std::string absOffdiagTableName = mom_space ? std::string("rho_abs_offdiag_kl") : std::string("rho_abs_offdiag_ij");
    
    arma::cx_cube& rho_offdiag = mom_space ? this->_rho_offdiag_kl : this->_rho_offdiag_ij;
    arma::umat& n_samples = mom_space ? this->_n_samples_offdiag_kl : this->_n_samples_offdiag_ij;
    arma::cube& rho_abs_offdiag = mom_space ? this->_rho_abs_offdiag_kl : this->_rho_abs_offdiag_ij;
    
    for (unsigned int spin_conf = 0; spin_conf < rho_offdiag.n_cols; spin_conf++) {
        if (initialize) {
            n_samples.unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                offdiagTableName + sep + std::to_string(spin_conf) + sep + std::string("n_samples"),
                arma::hdf5_opts::append));
        } else {
            n_samples.unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                offdiagTableName + sep + std::to_string(spin_conf) + sep + std::string("n_samples"),
                arma::hdf5_opts::replace));
        }
        for (unsigned int i_t = 0; i_t < rho_offdiag.n_slices; ++i_t) {
            if (initialize) {
                rho_offdiag.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                    offdiagTableName + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::append));
            } else {
                rho_offdiag.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                    offdiagTableName + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::replace));
            }
            if (with_abs2) {
                if (initialize) {
                    rho_abs_offdiag.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                        absOffdiagTableName + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::append));
                } else {
                    rho_abs_offdiag.slice(i_t).unsafe_col(spin_conf).save(arma::hdf5_name(filename,
                        absOffdiagTableName + sep + std::to_string(spin_conf) + sep + std::to_string(i_t), arma::hdf5_opts::replace));
                }
            }
        }
    }
    
    return STATUS_OKAY;
}

int HDF5CheckpointManager::writeBonds(bool initialize, bool with_abs, bool with_abs2) {
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    std::string bondsTableName = std::string("rho_bonds");
    
    if (initialize) {
        _n_samples_bonds.save(arma::hdf5_name(filename,
            bondsTableName + sep + std::string("n_samples"),
            arma::hdf5_opts::append));
    } else {
        _n_samples_bonds.save(arma::hdf5_name(filename,
            bondsTableName + sep + std::string("n_samples"),
            arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_bonds.save(arma::hdf5_name(filename,
            bondsTableName + sep + std::string("bonds"), arma::hdf5_opts::append));
    } else {
        _rho_bonds.save(arma::hdf5_name(filename,
            bondsTableName + sep + std::string("bonds"), arma::hdf5_opts::replace));
    }
    if (with_abs) {
        if (initialize) {
            _rho_abs_bonds.save(arma::hdf5_name(filename,
                bondsTableName + sep + std::string("abs_bonds"), arma::hdf5_opts::append));
        } else {
            _rho_abs_bonds.save(arma::hdf5_name(filename,
                bondsTableName + sep + std::string("abs_bonds"), arma::hdf5_opts::replace));
        }
    }
    if (with_abs2) {
        if (initialize) {
            _rho_abs2_bonds.save(arma::hdf5_name(filename,
                bondsTableName + sep + std::string("abs2_bonds"), arma::hdf5_opts::append));
        } else {
            _rho_abs2_bonds.save(arma::hdf5_name(filename,
                bondsTableName + sep + std::string("abs2_bonds"), arma::hdf5_opts::replace));
        }
    }
    
    if (initialize) {
        _rho_bondbond_cell0.save(arma::hdf5_name(filename,
            bondsTableName + sep + std::string("bondbond_cell0"), arma::hdf5_opts::append));
    } else {
        _rho_bondbond_cell0.save(arma::hdf5_name(filename,
            bondsTableName + sep + std::string("bondbond_cell0"), arma::hdf5_opts::replace));
    }
    
    return STATUS_OKAY;
}

int HDF5CheckpointManager::writeFluxes(bool initialize) {
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    std::string fluxesTableName = std::string("rho_fluxes");
    
    // arma::hdf5_opts::opts mode(arma::hdf5_opts::none);

    
    if (initialize) {
        _n_samples_fluxes.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("n_samples"),
            arma::hdf5_opts::append));
    } else {
        _n_samples_fluxes.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("n_samples"),
            arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_pi_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("pi_fluxes_plaq"), arma::hdf5_opts::append));
    } else {
        _rho_pi_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("pi_fluxes_plaq"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_abs_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("abs_fluxes_plaq"), arma::hdf5_opts::append));
    } else {
        _rho_abs_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("abs_fluxes_plaq"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_phase_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("phase_fluxes_plaq"), arma::hdf5_opts::append));
    } else {
        _rho_phase_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("phase_fluxes_plaq"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_op_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_fluxes_plaq"), arma::hdf5_opts::append));
    } else {
        _rho_op_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_fluxes_plaq"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_op_abs_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_abs_fluxes_plaq"), arma::hdf5_opts::append));
    } else {
        _rho_op_abs_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_abs_fluxes_plaq"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_op_squared_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_squared_fluxes_plaq"), arma::hdf5_opts::append));
    } else {
        _rho_op_squared_fluxes_plaq.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_squared_fluxes_plaq"), arma::hdf5_opts::replace));
    }
    
    // ######################### VERTICES ##############################
    
    if (initialize) {
        _rho_pi_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("pi_fluxes_vert"), arma::hdf5_opts::append));
    } else {
        _rho_pi_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("pi_fluxes_vert"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_abs_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("abs_fluxes_vert"), arma::hdf5_opts::append));
    } else {
        _rho_abs_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("abs_fluxes_vert"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_phase_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("phase_fluxes_vert"), arma::hdf5_opts::append));
    } else {
        _rho_phase_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("phase_fluxes_vert"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_op_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_fluxes_vert"), arma::hdf5_opts::append));
    } else {
        _rho_op_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_fluxes_vert"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_op_abs_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_abs_fluxes_vert"), arma::hdf5_opts::append));
    } else {
        _rho_op_abs_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_abs_fluxes_vert"), arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_op_squared_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_squared_fluxes_vert"), arma::hdf5_opts::append));
    } else {
        _rho_op_squared_fluxes_vert.save(arma::hdf5_name(filename,
            fluxesTableName + sep + std::string("op_squared_fluxes_vert"), arma::hdf5_opts::replace));
    }
    
    return STATUS_OKAY;
}

int HDF5CheckpointManager::writeModes(bool initialize) {
    std::string filename = this->_runName + std::string(".h5");
    std::string sep("/");
    
    std::string tableName = std::string("rho_modes");
    
    if (initialize) {
        _n_samples_k_diag_PM.save(arma::hdf5_name(filename,
            tableName + sep + std::string("n_samples"),
            arma::hdf5_opts::append));
    } else {
        _n_samples_k_diag_PM.save(arma::hdf5_name(filename,
            tableName + sep + std::string("n_samples"),
            arma::hdf5_opts::replace));
    }
    
    if (initialize) {
        _rho_k_diag_PM.save(arma::hdf5_name(filename,
            tableName + sep + std::string("k_diag_PM"),
            arma::hdf5_opts::append));
    } else {
        _rho_k_diag_PM.save(arma::hdf5_name(filename,
            tableName + sep + std::string("k_diag_PM"),
            arma::hdf5_opts::replace));
    }
    
    return STATUS_OKAY;
}
