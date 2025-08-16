// su_n/eom_hub.cpp

#include "eom_hub.hpp"

namespace ftwa_su_n {

ODEHub2dPBC::ODEHub2dPBC(
    const Lattice& lattice, const HubbardParameters& params,
    const SimulationParameters& simParams, CheckpointManager& cm,
    const ftwa_su_n::FourierTransformer2dPBC& ft, unsigned int verbosity)
    : _lattice(lattice), _params(params), _simParams(simParams),
      _cm(cm), _ft(ft), _fourierOutput(true), _verbosity(verbosity)  {
#ifdef FTWA_CACHE_CHECKPOINTS
    _checkpoints = arma::zeros<arma::cx_cube>(_lattice.rhoLen(), 1, _simParams.num_tsteps+1);
#endif
    // std::cout << "(" << omp_get_thread_num() << ") Hubbard2dPeriodicODE: I'm being constructed! (" << this << ")" << std::endl;
}

ODEHub2dPBC::~ODEHub2dPBC() {
    // std::cout << "(" << omp_get_thread_num() << ") Hubbard2dPeriodicODE: I'm being destroyed! (" << this << ")" << std::endl;
}

void ODEHub2dPBC::system2(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const {
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    
    if (_verbosity >= 1000) {
        std::cout << "# system called!" << std::endl;
    }
    
    unsigned int Lx = _lattice.getLength(0);
    unsigned int V = _lattice.numSites();
    unsigned int rhoLen =_lattice.rhoLen();
    
    dxdt.zeros(rhoLen);
    
    std::complex<double> hop = _params.t * std::complex<double>(0.0, -1.0);
    std::complex<double> hub = _params.U * std::complex<double>(0.0, -1.0);

    // specialization for i == j: no U-term
    unsigned int ctr = 0, c;
    for (unsigned int i = 0; i < V; ++i) {
        ctr = ftwa_index(i, i);
        // left
        if (i % Lx > 0) {
            c = ftwa_index(i-1, i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        } else {
            c = ftwa_index(i, i+Lx-1);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        }
        // above
        if (i > Lx-1) {
            c = ftwa_index(i-Lx, i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        } else {
            c = ftwa_index(i, i+V-Lx);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        }
        // right
        if (i % Lx < Lx-1) {
            c = ftwa_index(i, i+1);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        } else {
            c = ftwa_index(i-Lx+1, i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        }
        // below
        if (i < V-Lx) {
            c = ftwa_index(i, i+Lx);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        } else {
            c = ftwa_index(i-(V-Lx), i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        }
    }

    for (unsigned int j = 1; j < V; ++j) {
        for (unsigned int i = 0; i < j; ++i) {
            ctr = ftwa_index(i, j);
            // j left
            if (j % Lx > 0) {
                c = ftwa_index(i, j-1);
                dxdt(ctr) -= hop * x(c);
            } else {
                c = ftwa_index(i, j+Lx-1);
                dxdt(ctr) -= hop * x(c);
            }
            // j above
            if (j > Lx-1) {
                if (j-Lx < i) {
                    c = ftwa_index(j-Lx, i);
                    dxdt(ctr) -= hop * std::conj(x(c));
                }
                else {
                    c = ftwa_index(i, j-Lx);
                    dxdt(ctr) -= hop * x(c);
                }
            } else {
                c = ftwa_index(i, j+V-Lx);
                dxdt(ctr) -= hop * x(c);
            }
            // j right
            if (j % Lx < Lx-1) {
                c = ftwa_index(i, j+1);
                dxdt(ctr) -= hop * x(c);
            } else {
                if (j-Lx+1 < i) {
                    c = ftwa_index(j-Lx+1, i);
                    dxdt(ctr) -= hop * std::conj(x(c));
                } else {
                    c = ftwa_index(i, j-Lx+1);
                    dxdt(ctr) -= hop * x(c);
                }
            }
            // j below
            if (j < V-Lx) {
                c = ftwa_index(i, j+Lx);
                dxdt(ctr) -= hop * x(c);
            } else {
                if (j-(V-Lx) < i) {
                    c = ftwa_index(j-(V-Lx), i);
                    dxdt(ctr) -= hop * std::conj(x(c));
                } else {
                    c = ftwa_index(i, j-(V-Lx));
                    dxdt(ctr) -= hop * x(c);
                }
            }
            
            // i left
            if (i % Lx > 0) {
                c = ftwa_index(i-1, j);
                dxdt(ctr) += hop * x(c);
            } else {
                if (i+Lx-1 > j) {
                    c = ftwa_index(j, i+Lx-1);
                    dxdt(ctr) += hop * std::conj(x(c));
                } else {
                    c = ftwa_index(i+Lx-1, j);
                    dxdt(ctr) += hop * x(c);
                }
            }
            // i above
            if (i > Lx-1) {
                c = ftwa_index(i-Lx, j);
                dxdt(ctr) += hop * x(c);
            } else {
                if (i+V-Lx > j) {
                    c = ftwa_index(j, i+V-Lx);
                    dxdt(ctr) += hop * std::conj(x(c));
                } else {
                    c = ftwa_index(i+V-Lx, j);
                    dxdt(ctr) += hop * x(c);
                }
            }
            // i right
            if (i % Lx < Lx-1) {
                c = ftwa_index(i+1, j);
                dxdt(ctr) += hop * x(c);
            } else {
                c = ftwa_index(i-Lx+1, j);
                dxdt(ctr) += hop * x(c);
            }
            // i below
            if (i < V-Lx) {
                if (i+Lx > j) {
                    c = ftwa_index(j, i+Lx);
                    dxdt(ctr) += hop * std::conj(x(c));
                }
                else {
                    c = ftwa_index(i+Lx, j);
                    dxdt(ctr) += hop * x(c);
                }
            } else {
                c = ftwa_index(i-(V-Lx), j);
                dxdt(ctr) += hop * x(c);
            }
            
            // Hubbard interaction
            dxdt(ctr) += 2.0 * hub * (x(ftwa_index(j, j)) - x(ftwa_index(i, i))) * x(ctr);
        }
    }

#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    if (_verbosity >= 100) std::cout << "# TIMER system function: " << time_span.count() << "s" << std::endl;
#endif
}

void ODEHub2dPBC::system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const {
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    
    if (_verbosity >= 1000) {
        std::cout << "# system called!" << std::endl;
    }
    
    unsigned int Lx = _lattice.getLength(0), Ly = _lattice.getLength(1);
    unsigned int V = _lattice.numSites();
    unsigned int rhoLen =_lattice.rhoLen();
    
    dxdt.zeros(rhoLen);
    
    std::complex<double> hop = _params.t * std::complex<double>(0.0, -1.0);
    std::complex<double> hub = _params.U * std::complex<double>(0.0, -1.0);

    // specialization for i == j: no U-term
    unsigned int ctr = 0, c;
    for (unsigned int i = 0; i < V; ++i) {
        ctr = ftwa_index(i, i);
        // left
        if (i % Lx > 0) {
            c = ftwa_index(i-1, i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        } else {
            c = ftwa_index(i, i+Lx-1);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        }
        // above
        if (i > Lx-1) {
            c = ftwa_index(i-Lx, i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        } else {
            c = ftwa_index(i, i+V-Lx);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        }
        // right
        if (i % Lx < Lx-1) {
            c = ftwa_index(i, i+1);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        } else {
            c = ftwa_index(i-Lx+1, i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        }
        // below
        if (i < V-Lx) {
            c = ftwa_index(i, i+Lx);
            dxdt(ctr) += hop * (std::conj(x(c)) - x(c));
        } else {
            c = ftwa_index(i-(V-Lx), i);
            dxdt(ctr) += hop * (x(c) - std::conj(x(c)));
        }
    }
    
    unsigned int i, j;
    for (unsigned int jy = 0; jy < Ly; ++jy) {
        for (unsigned int jx = 0; jx < Lx; ++jx) {
            j = jy*Lx + jx;
            for (unsigned int iy = 0; iy <= jy; ++iy) {
                for (unsigned int ix = 0; ix < Lx; ++ix) {
                    i = iy*Lx + ix;
                    if (i >= j) break;
                    
                    ctr = ftwa_index(i, j);
                    // j left
                    if (jx > 0) {
                        c = ftwa_index(i, j-1);
                        dxdt(ctr) -= hop * x(c);
                    } else {
                        c = ftwa_index(i, j+Lx-1);
                        dxdt(ctr) -= hop * x(c);
                    }
                    // j above
                    if (jy > 0) {
                        if (j-Lx < i) {
                            c = ftwa_index(j-Lx, i);
                            dxdt(ctr) -= hop * std::conj(x(c));
                        }
                        else {
                            c = ftwa_index(i, j-Lx);
                            dxdt(ctr) -= hop * x(c);
                        }
                    } else {
                        c = ftwa_index(i, j+V-Lx);
                        dxdt(ctr) -= hop * x(c);
                    }
                    // j right
                    if (jx < Lx-1) {
                        c = ftwa_index(i, j+1);
                        dxdt(ctr) -= hop * x(c);
                    } else {
                        if (j-Lx+1 < i) {
                            c = ftwa_index(j-Lx+1, i);
                            dxdt(ctr) -= hop * std::conj(x(c));
                        } else {
                            c = ftwa_index(i, j-Lx+1);
                            dxdt(ctr) -= hop * x(c);
                        }
                    }
                    // j below
                    if (j < V-Lx) {
                        c = ftwa_index(i, j+Lx);
                        dxdt(ctr) -= hop * x(c);
                    } else {
                        if (j-(V-Lx) < i) {
                            c = ftwa_index(j-(V-Lx), i);
                            dxdt(ctr) -= hop * std::conj(x(c));
                        } else {
                            c = ftwa_index(i, j-(V-Lx));
                            dxdt(ctr) -= hop * x(c);
                        }
                    }
                    
                    // i left
                    if (ix > 0) {
                        c = ftwa_index(i-1, j);
                        dxdt(ctr) += hop * x(c);
                    } else {
                        if (i+Lx-1 > j) {
                            c = ftwa_index(j, i+Lx-1);
                            dxdt(ctr) += hop * std::conj(x(c));
                        } else {
                            c = ftwa_index(i+Lx-1, j);
                            dxdt(ctr) += hop * x(c);
                        }
                    }
                    // i above
                    if (i > Lx-1) {
                        c = ftwa_index(i-Lx, j);
                        dxdt(ctr) += hop * x(c);
                    } else {
                        if (i+V-Lx > j) {
                            c = ftwa_index(j, i+V-Lx);
                            dxdt(ctr) += hop * std::conj(x(c));
                        } else {
                            c = ftwa_index(i+V-Lx, j);
                            dxdt(ctr) += hop * x(c);
                        }
                    }
                    // i right
                    if (ix < Lx-1) {
                        c = ftwa_index(i+1, j);
                        dxdt(ctr) += hop * x(c);
                    } else {
                        c = ftwa_index(i-Lx+1, j);
                        dxdt(ctr) += hop * x(c);
                    }
                    // i below
                    if (i < V-Lx) {
                        if (i+Lx > j) {
                            c = ftwa_index(j, i+Lx);
                            dxdt(ctr) += hop * std::conj(x(c));
                        }
                        else {
                            c = ftwa_index(i+Lx, j);
                            dxdt(ctr) += hop * x(c);
                        }
                    } else {
                        c = ftwa_index(i-(V-Lx), j);
                        dxdt(ctr) += hop * x(c);
                    }
                    
                    // Hubbard interaction
                    dxdt(ctr) += 2.0 * hub * (x(ftwa_index(j, j)) - x(ftwa_index(i, i))) * x(ctr);
                }
            }
        }
    }
#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    if (_verbosity >= 100) std::cout << "# TIMER system function: " << time_span.count() << "s" << std::endl;
#endif
}

void ODEHub2dPBC::observer(const arma::cx_vec& x, const double t) {
    // unsigned int V = _lattice.numSites();
    unsigned int i_t = std::round((t - _simParams.start_time) / _simParams.checkpoint_dt);
    
    if (_verbosity >= 1000) {
        std::cout << "# observer called at " << i_t << " !" << std::endl;
    }
#ifdef FTWA_WITH_TIMER
        std::chrono::steady_clock sc;
        // std::chrono::duration<double> time_span;
        auto snap0 = sc.now();
#endif
    
#ifdef FTWA_CACHE_CHECKPOINTS
    _checkpoints.slice(i_t).col(0) = x;
#else
    _cm.updateDensities(x, i_t,
            false, // mom_space
            false, // with_cov_densdens
            true   // with_cov_densdens_diag
    );
    // _cm.updateOffdiag(x, i_t,
    //         false, // mom_space
    //         false, // with_cov
    //         false  // with_abs2
    // );
    
    arma::cx_vec rho_k = _ft.transform(x);
    _cm.updateDensities(rho_k, i_t,
            true,  // mom_space
            false, // with_cov_densdens
            true   // with_cov_densdens_diag
    );
    // _cm.updateOffdiag(rho_k, i_t,
    //         true,  // mom_space
    //         false, // with_cov
    //         false  // with_abs2
    // );

#endif
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        if (_verbosity >= 100) std::cout << "# TIMER OBSERVER updating observables: " << time_span.count() << "s" << std::endl;
#endif
    
    // last iteration: write checkpoints to database
    if (i_t == _simParams.num_tsteps) {
#ifdef FTWA_CACHE_CHECKPOINTS
        _cm.updateDensities(_checkpoints, false, true, false);
        _cm.updateOffdiag(_checkpoints, false, false, false);
#ifdef FTWA_WITH_TIMER
        auto snap2 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap2 - snap1);
        std::cout << "# TIMER OBSERVER update with pos checkpoint cube: " << time_span.count() << "s" << std::endl;
#endif
        arma::cx_cube checkpoints_mom(_checkpoints.n_rows, _checkpoints.n_cols, _checkpoints.n_slices);
        arma::cx_vec temp(_checkpoints.n_rows);
        for (unsigned int i_t = 0; i_t < _checkpoints.n_slices; ++i_t) {
            _ft.transform(_checkpoints.slice(i_t).unsafe_col(0), temp);
            checkpoints_mom.slice(i_t).col(0) = temp;
        }
#ifdef FTWA_WITH_TIMER
        auto snap3 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap3 - snap2);
        if (_verbosity >= 10) std::cout << "# TIMER OBSERVER FT checkpoints: " << time_span.count() << "s" << std::endl;
#endif
        _cm.updateDensities(checkpoints_mom, true, true, false);
        _cm.updateOffdiag(checkpoints_mom, true, false, false);
#ifdef FTWA_WITH_TIMER
        auto snap4 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap3 - snap2);
        std::cout << "# TIMER OBSERVER update with mom checkpoint cube: " << time_span.count() << "s" << std::endl;
#endif
#endif

#ifdef FTWA_WITH_TIMER
        auto snap_before_write = sc.now();
#endif
        if (_cm.write_flag) {
            _cm.beginTransaction();
            _cm.writeDensities(
                    false, // initialize
                    false, // mom_space
                    false, // with_cov_densdens
                    true   // with_cov_densdens_diag
            );
            _cm.writeDensities(
                    false, // initialize
                    true,  // mom_space
                    false, // with_cov_densdens
                    true   // with_cov_densdens_diag
            );
            // _cm.writeOffdiag(
            //         false, // initialize
            //         true,  // mom_space
            //         false, // with_cov
            //         false  // with_abs2
            // );
            // _cm.writeOffdiag(
            //         false, // initialize
            //         false, // mom_space
            //         false, // with_cov
            //         false  // with_abs2
            // );
            _cm.endTransaction();
        }
#ifdef FTWA_WITH_TIMER
        auto snap_after_write = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap_after_write - snap_before_write);
        std::cout << "# TIMER OBSERVER writing to disk: " << time_span.count() << "s" << std::endl;
#endif
    }
}

ODEHub2dMomPBC::ODEHub2dMomPBC(
    const Lattice& lattice,
    const HubbardParameters& params,
    const SimulationParameters& simParams,
    CheckpointManager& cm,
    const ftwa_su_n::FourierTransformer2dPBC& ft,
    unsigned int verbosity,
    double en_cutoff,
    bool use_cutoff
    ) : _lattice(lattice), _params(params), _simParams(simParams),
      _cm(cm), _ft(ft), _fourierOutput(true), _verbosity(verbosity),
      _en_cutoff(en_cutoff), _use_cutoff(use_cutoff) {
#ifdef FTWA_CACHE_CHECKPOINTS
    _checkpoints = arma::zeros<arma::cx_cube>(_lattice.rhoLen(), 1, _simParams.num_tsteps+1);
#endif
    
    _en_diffs = arma::vec(_lattice.rhoLen());
    
    unsigned int k, l, Lx = _lattice.getLength(0), Ly = _lattice.getLength(1);
    for (unsigned int ly = 0; ly < Ly; ++ly) {
        for (unsigned int lx = 0; lx < Lx; ++lx) {
            l = ly*Lx + lx;
            for (unsigned int ky = 0; ky <= ly; ++ky) {
                for (unsigned int kx = 0; kx < Lx; ++kx) {
                    if (ky == ly && kx == lx+1) break;
                    k = ky*Lx + kx;
                    _en_diffs(ftwa_index(k, l)) = dispTightBinding2d(_lattice.getLength(0), kx, ky) - dispTightBinding2d(_lattice.getLength(0), lx, ly);
                }
            }
        }
    }
}

ODEHub2dMomPBC::~ODEHub2dMomPBC() { }

void ODEHub2dMomPBC::system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const {  
    unsigned int Lx = _lattice.getLength(0);
    unsigned int Ly = _lattice.getLength(1);
    unsigned int V = _lattice.numSites();
    unsigned int rhoLen =_lattice.rhoLen();
    
    if (_verbosity >= 1000) {
        std::cout << "# system called at t = " << t << "!" << std::endl;
    }
    
    dxdt.zeros(rhoLen);
    
    std::complex<double> hub = 2.0*_params.U / V * std::complex<double>(0.0, -1.0);
    
    // arma::cx_vec en_exp = arma::exp(std::complex<double>(0, 1.0) * t * arma::conv_to<arma::cx_vec>::from(_en_diffs));
    arma::cx_vec en_exp = arma::exp(std::complex<double>(0, 1.0) * t * _en_diffs);
    
    unsigned int k, l, p, s, ctr, i, ix, iy;
    double del_en = 0.0;
    
    for (unsigned int ly = 0; ly < Ly; ++ly) {
        for (unsigned int lx = 0; lx < Lx; ++lx) {
            l = ly*Lx + lx;
            
            for (unsigned int ky = 0; ky <= ly; ++ky) {
                for (unsigned int kx = 0; kx < Lx; ++kx) {
                    if (ky == ly && kx == lx+1) break;
                    k = ky*Lx + kx;
            
                    ctr = ftwa_index(k, l);
                    
                    for (unsigned int sx = 0; sx < Lx; ++sx) {
                        for (unsigned int sy = 0; sy < Ly; ++sy) {
                            s = sy*Lx + sx;
                            
                            for (unsigned int px = 0; px < Lx; ++px) {
                                for (unsigned int py = 0; py < Ly; ++py) {
                                    p = py*Lx + px;
                                    
                                    ix = (px + sx + (Lx-lx)) % Lx;
                                    iy = (py + sy + (Ly-ly)) % Ly;
                                    i = iy*Lx + ix;
                                    
                                    del_en = std::abs(((i <= p) ? _en_diffs(ftwa_index(i, p)) : -_en_diffs(ftwa_index(p, i))) + ((l <= s) ? _en_diffs(ftwa_index(l, s)) : -_en_diffs(ftwa_index(s, l))));
                                    if (!_use_cutoff || del_en < _en_cutoff) {
                                        dxdt(ctr) += hub * ((i <= p) ? en_exp(ftwa_index(i, p))*x(ftwa_index(i, p)) : std::conj(en_exp(ftwa_index(p, i))*x(ftwa_index(p, i))))
                                        * ((l <= s) ? en_exp(ftwa_index(l, s)) : std::conj(en_exp(ftwa_index(s, l))))
                                        * ((k <= s) ? x(ftwa_index(k, s)) : std::conj(x(ftwa_index(s, k))));
                                    }
                                    
                                    ix = (px + kx + (Lx-sx)) % Lx;
                                    iy = (py + ky + (Ly-sy)) % Ly;
                                    i = iy*Lx + ix;
                                    
                                    del_en = std::abs(((i <= p) ? _en_diffs(ftwa_index(i, p)) : -_en_diffs(ftwa_index(p, i))) + ((s <= k) ? _en_diffs(ftwa_index(s, k)) : -_en_diffs(ftwa_index(k, s))));
                                    if (!_use_cutoff || del_en < 1.0/10.0) {
                                        dxdt(ctr) -= hub * ((i <= p) ? en_exp(ftwa_index(i, p))*x(ftwa_index(i, p)) : std::conj(en_exp(ftwa_index(p, i))*x(ftwa_index(p, i))))
                                        * ((s <= k) ? en_exp(ftwa_index(s, k)) : std::conj(en_exp(ftwa_index(k, s))))
                                        * ((s <= l) ? x(ftwa_index(s, l)) : std::conj(x(ftwa_index(l, s))));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ODEHub2dMomPBC::observer(const arma::cx_vec& x, const double t) {
    // unsigned int V = _lattice.numSites();
    unsigned int i_t = std::round((t - _simParams.start_time) / _simParams.checkpoint_dt);

#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    std::chrono::duration<double> time_span;
#endif
    
    if (_verbosity >= 1000) {
        std::cout << "# observer called at " << i_t << " !" << std::endl;
    }
    
    arma::cx_vec en_exp = arma::exp(std::complex<double>(0, 1.0) * t * _en_diffs);
    
#ifdef FTWA_CACHE_CHECKPOINTS
    _checkpoints.slice(i_t).col(0) = en_exp % x;
#else
    arma::cx_vec y = en_exp % x;
    _cm.updateDensities(y, i_t, true, true, false);
    _cm.updateOffdiag(y, i_t, true, false, false);
    
    arma::cx_vec temp;
    _ft.itransform(y, temp);
    
    _cm.updateDensities(y, i_t, false, true, false);
    _cm.updateOffdiag(y, i_t, false, false, false);
#endif
    
    // last iteration: write checkpoints to database
    if (i_t == _simParams.num_tsteps) {
#ifdef FTWA_CACHE_CHECKPOINTS
#ifdef FTWA_WITH_TIMER
        auto snap0 = sc.now();
#endif
        _cm.updateDensities(_checkpoints, true, true, false);
        _cm.updateOffdiag(_checkpoints, true, false, false);
        
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        std::cout << "# TIMER OBSERVER update with mom checkpoint cube: " << time_span.count() << "s" << std::endl;
#endif
        
        arma::cx_cube checkpoints_pos(_checkpoints.n_rows, _checkpoints.n_cols, _checkpoints.n_slices);
        arma::cx_vec temp(_checkpoints.n_rows);
        for (unsigned int i_t = 0; i_t < _checkpoints.n_slices; ++i_t) {
            _ft.itransform(_checkpoints.slice(i_t).unsafe_col(0), temp);
            checkpoints_pos.slice(i_t).col(0) = temp;
        }

#ifdef FTWA_WITH_TIMER
        auto snap2 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap2 - snap1);
        std::cout << "# TIMER OBSERVER FT checkpoints: " << time_span.count() << "s" << std::endl;
#endif
        
        _cm.beginTransaction();
        _cm.updateDensities(checkpoints_pos, false, true, false);
        _cm.updateOffdiag(checkpoints_pos, false, false, false);
        _cm.endTransaction();
#ifdef FTWA_WITH_TIMER
        auto snap3 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap3 - snap2);
        std::cout << "# TIMER OBSERVER update with pos checkpoint cube: " << time_span.count() << "s" << std::endl;
#endif
#endif

#ifdef FTWA_WITH_TIMER
        auto snap_before_write = sc.now();
#endif
        if (_cm.write_flag) {
            _cm.beginTransaction();
            _cm.writeDensities(false, false, true, false);
            _cm.writeDensities(false, true, true, false);
            _cm.writeOffdiag(false, false, false, false);
            _cm.writeOffdiag(false, true, false, false);
            _cm.endTransaction();
        }
#ifdef FTWA_WITH_TIMER
        auto snap_after_write = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap_after_write - snap_before_write);
        std::cout << "# TIMER OBSERVER writing to disk: " << time_span.count() << "s" << std::endl;
#endif
    }
}

ODEHub2dHiePBC::ODEHub2dHiePBC(
    const SquareLattice& lattice, const HubbardParameters& params,
    const SimulationParameters& simParams, CheckpointManager& cm,
    const ftwa_su_n::FourierTransformer2dPBC& ft, unsigned int verbosity)
    : _lattice(lattice), _params(params), _simParams(simParams),
      _cm(cm), _ft(ft), _fourierOutput(true), _verbosity(verbosity)  { }

ODEHub2dHiePBC::~ODEHub2dHiePBC() { }

void ODEHub2dHiePBC::system(const arma::cx_vec& x, arma::cx_vec& dxdt, const double t) const {
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    
    if (_verbosity >= 1000) {
        std::cout << "# system called!" << std::endl;
    }
    
    unsigned int Lx = _lattice.getLength(0), Ly = _lattice.getLength(1);
    unsigned int V = _lattice.numSites();
    unsigned int rhoLen = _lattice.rhoLen();
    
    dxdt.set_size(rhoLen + rhoLen*V*V);
    
    arma::cx_vec rho  = x.head(rhoLen);
    arma::cx_mat Dmat = arma::reshape(x.tail(rhoLen*V*V), rhoLen, V*V);
    // Dmat layout
    // (i <= j), (m n)
    
    arma::cx_vec drhodt  = arma::zeros<arma::cx_vec>(rhoLen);
    arma::cx_mat dDmatdt = arma::zeros<arma::cx_mat>(rhoLen, V*V);
    
    std::complex<double> hop = _params.t * std::complex<double>(0.0, -1.0);
    std::complex<double> hub = _params.U * std::complex<double>(0.0, 1.0);
    
    unsigned int ctr = 0;
    std::vector<unsigned int> neighbors_i, neighbors_j, neighbors_m, neighbors_n;
    for (unsigned int j = 0; j < V; ++j) {
        neighbors_j = _lattice.getNeighbors(j);
        for (unsigned int i = 0; i <= j; ++i) {
            neighbors_i = _lattice.getNeighbors(i);
            
            for (unsigned int a = 0; a < 4; ++a) {
                drhodt(ctr) += hop * rhoVal(rho, neighbors_i[a], j);
                drhodt(ctr) -= hop * rhoVal(rho, i, neighbors_j[a]);
            }
            
            drhodt(ctr) += 2.0*hub*( DmatVal(V, Dmat, i, i, i, j) - DmatVal(V, Dmat, j, j, i, j) );
            //~ drhodt(ctr) += 2.0*hub*( DmatValFact(V, rho, i, i, i, j) - DmatValFact(V, rho, j, j, i, j) );
            
            for (unsigned int n = 0; n < V; ++n) {
                neighbors_n = _lattice.getNeighbors(n);
                for (unsigned int m = 0; m < V; ++m) {
                    neighbors_m = _lattice.getNeighbors(m);
                    
                    for (unsigned int a = 0; a < 4; ++a) {
                        dDmatdt(ctr, n*V+m) += hop*DmatVal(V, Dmat, neighbors_i[a], j, m, n);
                        dDmatdt(ctr, n*V+m) -= hop*DmatVal(V, Dmat, i, neighbors_j[a], m, n);
                        dDmatdt(ctr, n*V+m) += hop*DmatVal(V, Dmat, i, j, neighbors_m[a], n);
                        dDmatdt(ctr, n*V+m) -= hop*DmatVal(V, Dmat, i, j, m, neighbors_n[a]);
                        //~ dDmatdt(ctr, n*V+m) += hop*DmatValFact(V, rho, neighbors_i[a], j, m, n);
                        //~ dDmatdt(ctr, n*V+m) -= hop*DmatValFact(V, rho, i, neighbors_j[a], m, n);
                        //~ dDmatdt(ctr, n*V+m) += hop*DmatValFact(V, rho, i, j, neighbors_m[a], n);
                        //~ dDmatdt(ctr, n*V+m) -= hop*DmatValFact(V, rho, i, j, m, neighbors_n[a]);
                    }
                    
                    dDmatdt(ctr, n*V+m) += 2.0*hub*TmatDecouplVal(V, rho, Dmat, i, i, i, j, m, n);
                    dDmatdt(ctr, n*V+m) -= 2.0*hub*TmatDecouplVal(V, rho, Dmat, j, j, i, j, m, n);
                    dDmatdt(ctr, n*V+m) += 2.0*hub*TmatDecouplVal(V, rho, Dmat, i, j, m, n, m, m);
                    dDmatdt(ctr, n*V+m) -= 2.0*hub*TmatDecouplVal(V, rho, Dmat, i, j, m, n, n, n);
                }
            }
            
            ctr++;
        }
    }
    
    dxdt.head(rhoLen)     = drhodt;
    dxdt.tail(rhoLen*V*V) = dDmatdt.as_col();
    
#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    if (_verbosity >= 100) std::cout << "# TIMER system function: " << time_span.count() << "s" << std::endl;
#endif
}

void ODEHub2dHiePBC::observer(const arma::cx_vec& x, const double t) {
    // unsigned int V = _lattice.numSites();
    unsigned int i_t = std::round((t - _simParams.start_time) / _simParams.checkpoint_dt);
    
    if (_verbosity >= 1000) {
        std::cout << "# observer called at " << i_t << " !" << std::endl;
    }
#ifdef FTWA_WITH_TIMER
        std::chrono::steady_clock sc;
        // std::chrono::duration<double> time_span;
        auto snap0 = sc.now();
#endif

    arma::cx_vec rho  = x.head(_lattice.rhoLen());
    
    _cm.updateDensities(rho, i_t,
            false, // mom_space
            false, // with_cov_densdens
            true   // with_cov_densdens_diag
    );
    // _cm.updateOffdiag(x, i_t,
    //         false, // mom_space
    //         false, // with_cov
    //         false  // with_abs2
    // );
    
    arma::cx_vec rho_k = _ft.transform(rho);
    _cm.updateDensities(rho_k, i_t,
            true,  // mom_space
            false, // with_cov_densdens
            true   // with_cov_densdens_diag
    );
    // _cm.updateOffdiag(rho_k, i_t,
    //         true,  // mom_space
    //         false, // with_cov
    //         false  // with_abs2
    // );

#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        if (_verbosity >= 100) std::cout << "# TIMER OBSERVER updating observables: " << time_span.count() << "s" << std::endl;
#endif
    
    // last iteration: write checkpoints to database
    if (i_t == _simParams.num_tsteps) {
#ifdef FTWA_WITH_TIMER
        auto snap_before_write = sc.now();
#endif
        if (_cm.write_flag) {
            _cm.beginTransaction();
            _cm.writeDensities(
                    false, // initialize
                    false, // mom_space
                    false, // with_cov_densdens
                    true   // with_cov_densdens_diag
            );
            _cm.writeDensities(
                    false, // initialize
                    true,  // mom_space
                    false, // with_cov_densdens
                    true   // with_cov_densdens_diag
            );
            // _cm.writeOffdiag(
            //         false, // initialize
            //         true,  // mom_space
            //         false, // with_cov
            //         false  // with_abs2
            // );
            // _cm.writeOffdiag(
            //         false, // initialize
            //         false, // mom_space
            //         false, // with_cov
            //         false  // with_abs2
            // );
            _cm.endTransaction();
        }
#ifdef FTWA_WITH_TIMER
        auto snap_after_write = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap_after_write - snap_before_write);
        std::cout << "# TIMER OBSERVER writing to disk: " << time_span.count() << "s" << std::endl;
#endif
    }
}


}
