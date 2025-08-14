// su_n/eom_hubhei.cpp

#include "eom_hubhei.hpp"

namespace ftwa_su_n {

ODEHubHei2dPBC::ODEHubHei2dPBC(
    const Lattice& lattice,
    const HubbardHeisenbergParameters& params,
    const SimulationParameters& simParams,
    const unitCellValues& ucv_tevol,
    CheckpointManager& cm,
    const FTSqrt2CellLattice& ft,
    double symmBreakStrength,
    bool mom_space,
    unsigned int verbosity) : _lattice(lattice), _params(params),
        _simParams(simParams), _ucv_tevol(ucv_tevol), _cm(cm), _ft(ft),
        _symmBreakStrength(symmBreakStrength), _mom_space(mom_space),
        _verbosity(verbosity) {
    
    unsigned int L = _lattice.getLength(0), V = _lattice.numSites(),
            rhoLen = _lattice.rhoLen();
    
#ifdef FTWA_CACHE_CHECKPOINTS
    _checkpoints = arma::zeros<arma::cx_cube>(rhoLen, 1, _simParams.num_tsteps+1);
#endif
    _nns = arma::zeros<arma::umat>(V, 4);
    
    //~ _indices_ai_j = arma::zeros<arma::uvec>(4*rhoLen);
    //~ _indices_aj_i = arma::zeros<arma::uvec>(4*rhoLen);
    //~ _rho_conj_ai_j = arma::zeros<arma::vec>(4*rhoLen);
    //~ _rho_conj_aj_i = arma::zeros<arma::vec>(4*rhoLen);
    //~ _indices_i = arma::zeros<arma::uvec>(4*rhoLen);
    //~ _indices_j = arma::zeros<arma::uvec>(4*rhoLen);
    
    // symmetry breaking for Flux order parameter
    double val = 0.0;
    std::array<double,4> aval = { 0.0, 0.0, 0.0, 0.0 };
    _symmBreakEps = arma::mat(V, 4); // eps_{l, a(l)}
    
    unsigned int neighbors[4], index = 0;
    for (unsigned int j = 0; j < V; ++j) {
        hubhei_sqrt2cell_neighbors(_lattice.getLength(0), j, neighbors);
        
        val = ((j / L) % 2 == 0) ? +_symmBreakStrength : -_symmBreakStrength;
        // This is the orientation of eps
        // where the direction is always "up-down"
        aval = { val, -val, -val, val };
        
        for (unsigned int a = 0; a < 4; ++a) {
            _nns(j, a) = neighbors[a];
            
            if (j <= neighbors[a]) {
                _symmBreakEps(j, a)   = ((j / L == 0) && (neighbors[a] / L == (2*L-1))) ? -aval[a] : +aval[a];
            } else {
                _symmBreakEps(j, a)   = ((j / L == (2*L-1)) && (neighbors[a] / L == 0)) ? +aval[a] : -aval[a];
            }
            
            //~ for (unsigned int i = 0; i <= j; ++i) {
                //~ index = ftwa_index(i, j);
                
                //~ _indices_i(a*rhoLen + index) = a*V + i;
                //~ _indices_j(a*rhoLen + index) = a*V + j;
                
                //~ if (_nns(i, a) > j) {
                    //~ _indices_ai_j(a*rhoLen + index) = ftwa_index(j, _nns(i, a));
                    //~ _rho_conj_ai_j(a*rhoLen + index) = -1.0;
                //~ } else {
                    //~ _indices_ai_j(a*rhoLen + index) = ftwa_index(_nns(i, a), j);
                    //~ _rho_conj_ai_j(a*rhoLen + index) = 1.0;
                //~ }
                
                //~ if (_nns(j, a) > i) {
                    //~ _indices_aj_i(a*rhoLen + index) = ftwa_index(i, _nns(j, a));
                    //~ _rho_conj_aj_i(a*rhoLen + index) = -1.0;
                //~ } else {
                    //~ _indices_aj_i(a*rhoLen + index) = ftwa_index(_nns(j, a), i);
                    //~ _rho_conj_aj_i(a*rhoLen + index) = 1.0;
                //~ }
            //~ }
        }
    }
#ifdef FTWA_OMIT_U
    std::cout << "# REMINDER: U is omitted in dynamics!!" << std::endl;
#endif
}

ODEHubHei2dPBC::~ODEHubHei2dPBC() { }

/**
 * tilted lattice structure: unit cell with double volume
 * L x 2L sites
 **/
void ODEHubHei2dPBC::system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const {
    unsigned int V = _lattice.numSites(), rhoLen = _lattice.rhoLen();
    
    dxdt.zeros(rhoLen);
    
    std::complex<double> hop = _params.t * std::complex<double>(0.0, 1.0);
    std::complex<double> hei = _params.J * std::complex<double>(0.0, -1.0);
    std::complex<double> hub = _params.U * std::complex<double>(0.0, -1.0);
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    
    unsigned int index = 0;
    
    arma::cx_mat pfact = hop + _symmBreakEps;
    
    for (unsigned int i = 0; i < V; ++i) {
        for (arma::uword a = 0; a < 4; ++a) {
            pfact(i, a) += hei * rhoVal(x, i, _nns(i, a));
        }
    }
    
    // #pragma omp parallel for default(none) private(index) shared(V, hei, hub, x, dxdt, pfact)
    for (unsigned int j = 0; j < V; ++j) {
        for (unsigned int i = 0; i <= j; ++i) {
            // index = ftwa_index(i, j);
            
            for (arma::uword a = 0; a < 4; ++a) {
                dxdt(index) += pfact(i, a) * rhoVal(x, _nns(i, a), j)
                   + std::conj(pfact(j, a) * rhoVal(x, _nns(j, a), i));
            }
#ifndef FTWA_OMIT_U
            dxdt(index) += 2.0*hub*(x(ftwa_index(j, j)) - x(ftwa_index(i, i)))*x(index);
#endif
            
            index++;
        }
    }

#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    if (_verbosity >= 100) std::cout << "# TIMER system function: " << time_span.count() << "s" << std::endl;
#endif
}

void ODEHubHei2dPBC::observer(const arma::cx_vec& x, const double t) {
    unsigned int i_t = std::round((t - _simParams.start_time) / _simParams.checkpoint_dt);
    
    if (_verbosity >= 10) std::cout << "# measurement at t = " << t << std::endl;
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
#ifdef FTWA_CACHE_CHECKPOINTS
    _checkpoints.slice(i_t).col(0) = x;
#else
    // _cm.updateDensities(x, i_t, false, true, false);
    _cm.updateDensities(
        x,
        i_t,
        false, // mom_space
        false, // with_cov_densdens
        true   // with_cov_densdens_diag
    );
    // _cm.updateOffdiag(x, i_t, false, false, true);
    _cm.updateBonds(
        x,
        i_t,
        true, // with_abs
        true  // with_abs2
    );
    _cm.updateFluxes(x, i_t);
    
    arma::cx_mat x_AB = _ft.reshapeFlatToAB(x);
    arma::cx_mat k_AB = _ft.transformDiagAB(x_AB);
    arma::cx_mat k_PM = _ft.fromDiagABtoPM(_ucv_tevol, k_AB);
    _cm.updateModes(k_PM, i_t);
    
#endif
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        if (_verbosity >= 100) std::cout << "# TIMER OBSERVER updating observables: " << time_span.count() << "s" << std::endl;
#endif
        
    // last iteration: write checkpoints to database
    if (i_t == _simParams.num_tsteps) { 
        
#ifdef FTWA_WITH_TIMER
        snap0 = sc.now();
#endif
#ifdef FTWA_CACHE_CHECKPOINTS
        _cm.updateDensities(_checkpoints, false, true, false);
        _cm.updateOffdiag(_checkpoints, false, false, true);
#endif
    
    if (_cm.write_flag) {
        if (_verbosity >= 10) std::cout << "# writing to database" << std::endl;
        
        _cm.beginTransaction();
        _cm.writeDensities(
            false, // initialize
            false, // mom_space
            false, // with_cov_densdens
            true   // with_cov_densdens_diag
        );
        // _cm.writeOffdiag(false, false, false, true);
        _cm.writeBonds(false, true, true);
        _cm.writeFluxes(false);
        _cm.writeModes(false);
        _cm.endTransaction();
    }
        
#ifdef FTWA_WITH_TIMER
        snap1 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        if (_verbosity >= 100) {
            std::cout << "# TIMER OBSERVER writing to db: " << time_span.count() << "s" << std::endl;
        }
#endif

    }
}


ODEHubHei2dSwitchJPBC::ODEHubHei2dSwitchJPBC(
    const Lattice& lattice,
    const HubbardHeisenbergParameters& params,
    const SimulationParameters& simParams,
    const unitCellValues& ucv_tevol,
    CheckpointManager& cm,
    const FTSqrt2CellLattice& ft,
    double symmBreakStrength,
    bool mom_space,
    unsigned int verbosity,
    const HubbardHeisenbergParameters& params_fin,
    const unitCellValues& ucv_fin,
    double time_switch_start,
    double time_switch_end,
    int switch_order,
    double switch_symm_break_strength) : ODEHubHei2dPBC(lattice, params, simParams,
        ucv_tevol, cm, ft, symmBreakStrength, mom_space, verbosity),
        _params_fin(params_fin), _ucv_fin(ucv_fin),
        _switch_time_start(time_switch_start),
        _switch_time_end(time_switch_end),
        _switch_order(switch_order),
        _switch_symm_break_strength(switch_symm_break_strength) { }

ODEHubHei2dSwitchJPBC::~ODEHubHei2dSwitchJPBC() { }

void ODEHubHei2dSwitchJPBC::system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const {
    unsigned int L = _lattice.getLength(0), V = _lattice.numSites(), rhoLen = _lattice.rhoLen();
    
    dxdt.zeros(rhoLen);
    
    double Jvalue = _params.J;
    // quench case
    if (_switch_order < 0) {
        if (_switch_time_end <= t) {
            Jvalue = _params_fin.J;
        }
    } else {
        if (_switch_time_start <= t) {
            if (t < _switch_time_end) {
                double tau_switch = (t - _switch_time_start) / (_switch_time_end - _switch_time_start);
                // interpolation functions smooth at boundaries up to m-th derivative
                switch (_switch_order) {
                    case 0:
                        Jvalue = _params_fin.J * tau_switch;
                        break;
                    case 1:
                        Jvalue = _params_fin.J * ( -std::pow(tau_switch, 2) * (2*(tau_switch) - 3) );
                        break;
                    case 2:
                        Jvalue = _params_fin.J * ( tau_switch - std::sin(2*3.141592*tau_switch)/(2*3.141592) );
                        break;
                    case 3:
                        Jvalue = _params_fin.J * ( -std::pow(tau_switch, 4)*( 20*std::pow(tau_switch, 3)
                                                   - 70*std::pow(tau_switch, 2) + 84*tau_switch - 35 ) );
                        break;
                    case 4:
                        Jvalue = _params_fin.J * ( std::pow(tau_switch, 5)*( 70*std::pow(tau_switch, 4)
                                                - 315*std::pow(tau_switch, 3) + 540*std::pow(tau_switch, 2)
                                                - 420*tau_switch + 126 ) );
                        break;
                }
            } else { // _switch_time_end <= t
                Jvalue = _params_fin.J;
            }
        }
    }
    
    std::complex<double> hop = _params.t * std::complex<double>(0.0, 1.0);
    std::complex<double> hei = Jvalue    * std::complex<double>(0.0, -1.0);
    std::complex<double> hub = _params.U * std::complex<double>(0.0, -1.0);
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    
    arma::cx_mat pfact = hop + _symmBreakEps;
    double pei_sb = (t < _switch_time_end ? _switch_symm_break_strength : 0.0);
    
    for (unsigned int i = 0; i < V; ++i) {
        for (arma::uword a = 0; a < 4; ++a) {
            pfact(i, a) += hei * rhoVal(x, i, _nns(i, a));
        }
        // transient symmetry breaking for the switching procedure
        if (_switch_order >= 0) {
            if ((i / L) % 2 == 0) {
                pfact(i, 2) += pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 2));
                
                //~ pfact(i, 0) -= 0.1*pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 0));
                //~ pfact(i, 1) -= 0.1*pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 1));
                //~ pfact(i, 3) -= 0.1*pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 3));
            } else {
                //~ // pfact(i, 0) += (t < Tswitch ? std::exp(-20*(t/Tswitch-0.5)*(t/Tswitch-0.5))*1e-3 : 0.0) * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 0));
                pfact(i, 0) += pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 0));
                
                //~ pfact(i, 1) -= 0.1*pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 1));
                //~ pfact(i, 2) -= 0.1*pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 2));
                //~ pfact(i, 3) -= 0.1*pei_sb * std::complex<double>(0.0, -1.0) * rhoVal(x, i, _nns(i, 3));
            }
        }
    }
    
    unsigned int index = 0;
    // #pragma omp parallel for default(none) private(index) shared(V, hei, hub, x, dxdt, pfact)
    for (unsigned int j = 0; j < V; ++j) {
        for (unsigned int i = 0; i <= j; ++i) {
            // index = ftwa_index(i, j);
            
            for (arma::uword a = 0; a < 4; ++a) {
                dxdt(index) += pfact(i, a) * rhoVal(x, _nns(i, a), j)
                   + std::conj(pfact(j, a) * rhoVal(x, _nns(j, a), i));
            }
#ifndef FTWA_OMIT_U
            dxdt(index) += 2.0*hub*(x(ftwa_index(j, j)) - x(ftwa_index(i, i)))*x(index);
#endif
            
            index++;
        }
    }

#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    if (_verbosity >= 100) std::cout << "# TIMER system function: " << time_span.count() << "s" << std::endl;
#endif
}

ODEHubHei2dPeierlsPBC::ODEHubHei2dPeierlsPBC(
    const Lattice& lattice,
    const HubbardHeisenbergParameters& params,
    const SimulationParameters& simParams,
    const unitCellValues& ucv_tevol,
    CheckpointManager& cm,
    const FTSqrt2CellLattice& ft,
    double symmBreakStrength,
    bool mom_space,
    unsigned int verbosity,
    const PeierlsPulseParameters& pulseParams
) : ODEHubHei2dPBC(lattice, params, simParams, ucv_tevol, cm, ft,
    symmBreakStrength, mom_space, verbosity),
    _pulseParams(pulseParams) {
    // compute all \varphi(i, a(i)) where we already account for the correct
    // direction signs INCLUDING the boundary effect
    // "physical" ordering: \varphi(i, j) positive if i <= j
    unsigned int L = _lattice.getLength(0), V = _lattice.numSites();
    unsigned int neighbors[4];
    double phi = 0.0, spat_fact = 1.0;
    double ix, iy, dist_ix = 0.0, dist_iy = 0.0;
    double ax, ay, dist_ax = 0.0, dist_ay = 0.0;
    unsigned int sigmaSpatSquared = _pulseParams.spat_width*_pulseParams.spat_width;
    _peierlsExpArgs = arma::mat(V, 4);
    
    for (unsigned int i = 0; i < V; ++i) {
        if (!_pulseParams.is_spat_uniform) {
            ix = (((i % (2*L)) > (L-1)) ? 0.5 : 0.0) + (double) (i % L);
            iy = 0.5 * (double) (i / L);
            dist_ix = std::abs(_pulseParams.spat_centr_x - ix);
            dist_ix = std::min(dist_ix, L - dist_ix);
            
            dist_iy = std::abs(_pulseParams.spat_centr_y - iy);
            dist_iy = std::min(dist_iy, L - dist_iy);
        }
        
        hubhei_sqrt2cell_neighbors(L, i, neighbors);
        
        for (unsigned int a = 0; a < 4; ++a) {
            if (!_pulseParams.is_spat_uniform) {
                ax = (((neighbors[a] % (2*L)) > (L-1)) ? 0.5 : 0.0) + (double) (neighbors[a] % L);
                ay = 0.5 * (double) (neighbors[a] / L);
                dist_ax = std::abs(_pulseParams.spat_centr_x - ax);
                dist_ax = std::min(dist_ax, L - dist_ax);

                dist_ay = std::abs(_pulseParams.spat_centr_y - ay);
                dist_ay = std::min(dist_ay, L - dist_ay);
            
                spat_fact = 0.5*(std::exp(-0.5*(dist_ix*dist_ix + dist_iy*dist_iy)/sigmaSpatSquared) + std::exp(-0.5*(dist_ax*dist_ax + dist_ay*dist_ay)/sigmaSpatSquared));
            }
            phi = spat_fact * ( ((a==0) || (a==2)) ? _pulseParams.ampl_y : _pulseParams.ampl_x );
            
            if (i <= neighbors[a]) {
                _peierlsExpArgs(i, a) = ((i / L == 0) && (neighbors[a] / L == 2*L-1)) ? -phi : phi;
            } else {
                _peierlsExpArgs(i, a) = ((i / L == 2*L-1) && (neighbors[a] / L == 0)) ? phi : -phi;
            }
        }
    }
}

ODEHubHei2dPeierlsPBC::~ODEHubHei2dPeierlsPBC() { };

void ODEHubHei2dPeierlsPBC::system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const {
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap_begin = sc.now();
#endif
    unsigned int V = _lattice.numSites(), rhoLen = _lattice.rhoLen();
    unsigned int index = 0;
    
    std::complex<double> hop = _params.t * std::complex<double>(0.0, 1.0);
    std::complex<double> hei = _params.J * std::complex<double>(0.0, -1.0);
    std::complex<double> hub = _params.U * std::complex<double>(0.0, -1.0);
    
    double diff_t = t-_pulseParams.temp_centr;
    double time_shape = std::exp(-0.5*diff_t*diff_t/(_pulseParams.temp_width*_pulseParams.temp_width)) * std::sin(_pulseParams.freq*diff_t + _pulseParams.phase);
    
    arma::cx_mat pfact = hop * arma::exp(std::complex<double>(0.0, -1.0) * time_shape * _peierlsExpArgs) + _symmBreakEps;
    
    for (unsigned int i = 0; i < V; ++i) {
        for (arma::uword a = 0; a < 4; ++a) {
            pfact(i, a) += hei * rhoVal(x, i, _nns(i, a));
        }
    }
    
    //~ ALTERNATIVE to loop below (except for hub)
    //~ arma::cx_vec pfactc = pfact.as_col();   
    //~ dxdt = arma::sum(arma::reshape(
                //~ pfactc(_indices_i) % arma::cx_vec(arma::real(x(_indices_ai_j)),
                                //~ _rho_conj_ai_j % arma::imag(x(_indices_ai_j)) )
  //~ + arma::conj( pfactc(_indices_j) % arma::cx_vec(arma::real(x(_indices_aj_i)),
                                //~ _rho_conj_aj_i % arma::imag(x(_indices_aj_i)) ) ),
                //~ rhoLen, 4), 1);
    
#ifdef FTWA_WITH_TIMER
    auto snap_before_loop = sc.now();
#endif

    dxdt.zeros(rhoLen);

    // #pragma omp parallel for default(none) private(index) shared(V, hei, hub, x, dxdt, pfact)
    for (unsigned int j = 0; j < V; ++j) {
        for (unsigned int i = 0; i <= j; ++i) {
            // index = ftwa_index(i, j);
            
            for (arma::uword a = 0; a < 4; ++a) {
                dxdt(index) += pfact(i, a) * rhoVal(x, _nns(i, a), j)
                   + std::conj(pfact(j, a) * rhoVal(x, _nns(j, a), i));
            }
#ifndef FTWA_OMIT_U
            dxdt(index) += 2.0*hub*(x(ftwa_index(j, j)) - x(ftwa_index(i, i)))*x(index);
#endif
            
            index++;
        }
    }
    
#ifdef FTWA_WITH_TIMER
    auto snap_end = sc.now();
    auto time_span_loop = static_cast<std::chrono::duration<double>>(snap_end - snap_before_loop);
    auto time_span_total = static_cast<std::chrono::duration<double>>(snap_end - snap_begin);
    if (_verbosity >= 1000) std::cout << "## TIMER system function (loop): " << time_span_loop.count() << "s" << std::endl;
    if (_verbosity >= 100)  std::cout << "# TIMER system function: " << time_span_total.count() << "s" << std::endl;
#endif
}


ODEHubHei2dMomPeierlsPBC::ODEHubHei2dMomPeierlsPBC(
    const Lattice& lattice,
    const HubbardHeisenbergParameters& params,
    const SimulationParameters& simParams,
    const unitCellValues& ucv_tevol,
    CheckpointManager& cm,
    const FTSqrt2CellLattice& ft,
    bool mom_space,
    unsigned int verbosity,
    const PeierlsPulseParameters& pulseParams
) : ODEHubHei2dPeierlsPBC(lattice, params, simParams, ucv_tevol, cm, ft,
    0.0, mom_space, verbosity, pulseParams) {
    
    unsigned int Lx = _lattice.getLength(0), Ly = _lattice.getLength(1),
        uV = _lattice.numCells(), kc;
    _bondFourierFactors = arma::zeros<arma::cx_mat>(uV, 4);
    std::complex<double> ukx = std::complex<double>(0, 2*M_PI/(double) Lx);
    std::complex<double> uky = std::complex<double>(0, 2*M_PI/(double) Ly);
    std::complex<double> pref;
    for (unsigned int ky = 0; ky < Ly; ++ky) {
        for (unsigned int kx = 0; kx < Lx; ++kx) {
            kc = Lx*ky+kx;
            pref = std::exp(-ukx*(0.5*kx) - uky*(0.5*ky));
            _bondFourierFactors(kc, 0) = pref * 1.0;
            _bondFourierFactors(kc, 1) = pref * std::exp(ukx*(1.0*kx));
            _bondFourierFactors(kc, 2) = pref * std::exp(ukx*(1.0*kx) + uky*(1.0*ky));
            _bondFourierFactors(kc, 3) = pref * std::exp(uky*(1.0*ky));
        }
    }
}

ODEHubHei2dMomPeierlsPBC::~ODEHubHei2dMomPeierlsPBC() { };

// vector k has the structure ( rho_{kA kA} rho_{kA kB} rho_{kB kA} rho_{kB kB} )^T
void ODEHubHei2dMomPeierlsPBC::system(const arma::cx_vec& k_diag, arma::cx_vec& dkdt_diag, const double t) const {
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    unsigned int Lx = _lattice.getLength(0), Ly = _lattice.getLength(1),
    uV = _lattice.numCells(), kc, kc2;
    
    if (_verbosity >= 1000) std::cout << "# system function at t = " << t << std::endl;

    arma::cx_mat dkdt_diag_AB = arma::zeros<arma::cx_mat>(uV, 4);
    
    arma::cx_mat k_diag_AB = _ft.reshapeDiagFlatToAB(k_diag);
    arma::cx_mat bond_fac = _bondFourierFactors;
    
    // std::cout << k_diag_AB << std::endl;
    
    //~ // symmetry for the Ay driving
    //~ for (unsigned int ky = 0; ky < Ly; ++ky) {
        //~ for (unsigned int kx = 0; kx < ky; ++kx) {
            //~ kc = Lx*ky+kx;
            //~ kc2 = Lx*kx+ky;
            //~ for (unsigned int i = 0; i < 2; ++i) {
                //~ k_diag_AB(kc2, i) = k_diag_AB(kc, i);
            //~ }
        //~ }
    //~ }
    
    bond_fac.each_col() %= k_diag_AB.col(1);
    arma::cx_vec v = arma::strans(arma::sum(bond_fac, 0)) / (1.0*uV);
    
    struct ftwa_su_n::unitCellValues ucv; // need to determine dynamically
    ucv.rhoA = 0.0;
    ucv.rhoB = 0.0;
    ucv.bonds[0] = v(0);
    ucv.bonds[1] = v(1);
    ucv.bonds[2] = std::conj(v(2));
    ucv.bonds[3] = std::conj(v(3));
    
    double diff_t = t-_pulseParams.temp_centr;
    double time_shape = std::exp(-0.5*diff_t*diff_t/(_pulseParams.temp_width*_pulseParams.temp_width)) * std::sin(_pulseParams.freq*diff_t + _pulseParams.phase);
    
    double Ax = _pulseParams.ampl_x * time_shape;
    double Ay = _pulseParams.ampl_y * time_shape;
    arma::cx_vec tkmat = std::complex<double>(0.0, -1.0) * create_tkmat(_lattice, _params, ucv, Ax, Ay);
    
    dkdt_diag_AB.unsafe_col(0).set_real(2.0*arma::real(tkmat % k_diag_AB.col(1)));
    // dkdt_diag_AB.col(0) = tkmat % k_diag_AB.col(1) + arma::conj(tkmat) % k_diag_AB.col(2);
    dkdt_diag_AB.col(1) = -2.0*arma::conj(tkmat) % k_diag_AB.col(0);
    // dkdt_diag_AB.col(1) = arma::conj(tkmat) % (k_diag_AB.col(3) - k_diag_AB.col(0));
    dkdt_diag_AB.col(2) = arma::conj(dkdt_diag_AB.col(1));
    dkdt_diag_AB.col(3) = -dkdt_diag_AB.col(0);
    
    arma::cx_vec temp = _ft.reshapeDiagABToFlat(dkdt_diag_AB);
    
    dkdt_diag = temp;

    //~ for (unsigned int ky = 0; ky < Ly; ++ky) {
        //~ for (unsigned int kx = 0; kx < Lx; ++kx) {
            //~ if (ky > kx) {
                //~ if (_verbosity >= 100) std::cout << "dkdt (" << kx << ", " << ky << "): " << dkdt_diag_AB(Lx*kx+ky, 1) << std::endl;
            //~ }
        //~ }
    //~ }
    
#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    if (_verbosity >= 100) std::cout << "# TIMER system function: " << time_span.count() << "s" << std::endl;
#endif
}


void ODEHubHei2dMomPeierlsPBC::observer(const arma::cx_vec& k, const double t) {
    unsigned int i_t = std::round((t - _simParams.start_time) / _simParams.checkpoint_dt);
    
    if (_verbosity >= 10) std::cout << "# observer function at t = " << t << std::endl;
    
    arma::cx_mat k_diag_AB = _ft.reshapeDiagFlatToAB(k);
    
    unsigned int Lx = _lattice.getLength(0), Ly = _lattice.getLength(1);
    
    // symmetry in the Ax = 0 case
    //~ arma::cx_vec dkdt(k);
    //~ this->system(k, dkdt, t);
    //~ arma::cx_mat dkdt_diag_AB = _ft.reshapeDiagFlatToAB(dkdt);
    for (unsigned int ky = 0; ky < Ly; ++ky) {
        for (unsigned int kx = 0; kx < ky; ++kx) {
            for (unsigned int i = 0; i < 2; ++i) {
                k_diag_AB(Lx*kx+ky, i) = k_diag_AB(Lx*ky+kx, i);
                //~ if (_verbosity >= 100) {
                    //~ std::cout << "     A A (" << kx << ", " << ky << "): " << k_diag_AB(Lx*ky+kx, 0) << std::endl;
                    //~ std::cout << "dkdt A A (" << kx << ", " << ky << "): " << dkdt_diag_AB(Lx*ky+kx, 0) << std::endl;
                    //~ std::cout << "     A B (" << kx << ", " << ky << "): " << k_diag_AB(Lx*ky+kx, 1) << std::endl;
                    //~ std::cout << "dkdt A B (" << kx << ", " << ky << "): " << dkdt_diag_AB(Lx*ky+kx, 1) << std::endl;
                //~ }
            }
        }
    }
    k_diag_AB.col(2) = arma::conj(k_diag_AB.col(1));
    k_diag_AB.col(3) = -k_diag_AB.col(0);
    
    arma::cx_mat x_AB = _ft.itransformDiagAB(k_diag_AB);
    arma::cx_mat x = _ft.reshapeABToFlat(x_AB);
    
    arma::cx_mat k_diag_PM = _ft.fromDiagABtoPM(_ucv_tevol, k_diag_AB);
    _cm.updateModes(k_diag_PM, i_t);
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
#ifdef FTWA_CACHE_CHECKPOINTS
    _checkpoints.slice(i_t).col(0) = x;
#else
    // _cm.updateDensities(x, i_t, false, true, false);
    _cm.updateDensities(
        x,
        i_t,
        false, // mom_space
        false, // with_cov_densdens
        true   // with_cov_densdens_diag
    );
    // _cm.updateOffdiag(x, i_t, false, false, true);
    _cm.updateBonds(
        x,
        i_t,
        true, // with_abs
        true  // with_abs2
    );
    _cm.updateFluxes(x, i_t);
    
#endif
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        if (_verbosity >= 100) std::cout << "# TIMER OBSERVER updating observables: " << time_span.count() << "s" << std::endl;
#endif
        
    // last iteration: write checkpoints to database
    if (i_t == _simParams.num_tsteps) { 
        
#ifdef FTWA_WITH_TIMER
        snap0 = sc.now();
#endif
#ifdef FTWA_CACHE_CHECKPOINTS
        _cm.updateDensities(_checkpoints, false, true, false);
        _cm.updateOffdiag(_checkpoints, false, false, true);
#endif
    
    if (_cm.write_flag) {
        if (_verbosity >= 10) std::cout << "# writing to database" << std::endl;
        
        _cm.beginTransaction();
        _cm.writeDensities(
            false, // initialize
            false, // mom_space
            false, // with_cov_densdens
            true   // with_cov_densdens_diag
        );
        // _cm.writeOffdiag(false, false, false, true);
        _cm.writeBonds(false, true, true);
        _cm.writeFluxes(false);
        _cm.writeModes(false);
        _cm.endTransaction();
    }
        
#ifdef FTWA_WITH_TIMER
        snap1 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        if (_verbosity >= 100) {
            std::cout << "# TIMER OBSERVER writing to db: " << time_span.count() << "s" << std::endl;
        }
#endif

    }
}


}
