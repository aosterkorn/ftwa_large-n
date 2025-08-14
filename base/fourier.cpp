// fourier.cpp

#include "fourier.hpp"

namespace ftwa_su_n {

FourierTransformer2dPBC::FourierTransformer2dPBC(const Lattice& lattice)
    : _lattice(lattice),
      _upper_indices(arma::trimatu_ind(arma::size(_lattice.numCells(), _lattice.numCells()))),
      _ftMat(_lattice.numCells(), _lattice.numCells()),
      _ftMatConj(_lattice.numCells(),
      _lattice.numCells()) {
        unsigned int Lx = _lattice.getLength(0);
        std::complex<double> omega = std::exp(std::complex<double>(0, -2*M_PI/(double) Lx));
        std::vector<std::complex<double>> omega_powers;
        for (int i = 0; i < (int) Lx; ++i) {
            omega_powers.push_back(std::pow(omega, i));
        }
        
        arma::cx_mat block(Lx, Lx);
        for (unsigned int mx = 0; mx < Lx; ++mx) {
            for (unsigned int ix = 0; ix < Lx; ++ix) {
                block(mx, ix) = omega_powers[(mx*ix) % Lx];
            }
        }
        
        //~ arma::cx_mat ftMatCtrl(Lx*Lx, Lx*Lx);
        //~ for (unsigned int my = 0; my < Lx; ++my) {
            //~ for (unsigned int mx = 0; mx < Lx; ++mx) {
                //~ for (unsigned int iy = 0; iy < Lx; ++iy) {
                    //~ for (unsigned int ix = 0; ix < Lx; ++ix) {
                        //~ ftMatCtrl(my*Lx+mx, iy*Lx+ix) = std::exp(std::complex<double>(0, -2.0*M_PI/((double) Lx) * (mx*ix+my*iy)));
                    //~ }
                //~ }
            //~ }
        //~ }
        //~ std::cout << "ftMat check: " << arma::norm(ftMatCtrl - _ftMat) << std::endl;
        //~ _ftMat = ftMatCtrl;
        
        _ftMat = arma::kron(block, block);
        _ftMat /= std::sqrt(_lattice.numCells());
        _ftMatConj = arma::conj(_ftMat);
        // _ftMatConj = _ftMat.t();
}

// Hubbard-Heisenberg model

void readUCVFromFile(const std::string& filename, struct unitCellValues& ucv) {
    std::string line, part;
    std::vector<std::string> parts;
    std::ifstream file(filename);
    if (file.is_open()) {
        while(std::getline(file, line)) {
            if (line.rfind("#", 0) == 0) continue;
            std::istringstream iss(line);
            while(std::getline(iss, part, '\t')) {
                parts.push_back(part);
            }
            ucv.bonds[std::stoi(parts[0])] = std::complex<double>(std::stod(parts[1]), std::stod(parts[2]));
            parts.clear();
        }
    }
}

// take full fTWA vector and make numCells x 4 matrix
// with entries sorted according to the unit cell indices
// 0 AA
// 1 AB
// 2 BA
// 3 BB
arma::cx_mat FTSqrt2CellLattice::reshapeFlatToAB(const arma::cx_vec& rho_flat) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        rhoLenCells = _lattice.rhoLenCells(),
        ic, jc, i_A, i_B, j_A, j_B;
        
    arma::cx_mat rho_AB(rhoLenCells, 4);
    
    for (unsigned int jy = 0; jy < Ly; ++jy) {
        for (unsigned int jx = 0; jx < Lx; ++jx) {
            
            jc = Lx*jy+jx;
            
            j_A = (2*Lx)*jy + jx;
            j_B = (2*Lx)*jy + Lx+jx;
            
            rho_AB(ftwa_index(jc, jc), 0) = rho_flat(ftwa_index(j_A, j_A));
            rho_AB(ftwa_index(jc, jc), 1) = rho_flat(ftwa_index(j_A, j_B));
            rho_AB(ftwa_index(jc, jc), 2) = std::conj(rho_flat(ftwa_index(j_A, j_B)));
            rho_AB(ftwa_index(jc, jc), 3) = rho_flat(ftwa_index(j_B, j_B));
            
            for (unsigned int iy = 0; iy <= jy; ++iy) {
                for (unsigned int ix = 0; ix < Lx; ++ix) {
                    ic = Lx*iy+ix;
                    
                    i_A = (2*Lx)*iy + ix;
                    i_B = (2*Lx)*iy + Lx+ix;
                    
                    // AA
                    rho_AB(ftwa_index(ic, jc), 0) = rho_flat(ftwa_index(i_A, j_A));
                    // AB
                    rho_AB(ftwa_index(ic, jc), 1) = rho_flat(ftwa_index(i_A, j_B));
                    // BA (if same row: B > A)
                    if (iy == jy) {
                        rho_AB(ftwa_index(ic, jc), 2) = std::conj(rho_flat(ftwa_index(j_A, i_B)));
                    } else {
                        rho_AB(ftwa_index(ic, jc), 2) = rho_flat(ftwa_index(i_B, j_A));
                    }
                    // BB
                    rho_AB(ftwa_index(ic, jc), 3) = rho_flat(ftwa_index(i_B, j_B));
                    
                    if (iy == jy && ix == jx) break;
                }
            }
        }
    }
    return rho_AB;
}

arma::cx_vec FTSqrt2CellLattice::reshapeABToFlat(const arma::cx_mat& rho_AB) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        rhoLen = _lattice.rhoLen(),
        ic, jc, i_A, i_B, j_A, j_B;
        
    arma::cx_vec rho_flat(rhoLen);
    
    for (unsigned int jy = 0; jy < Ly; ++jy) {
        for (unsigned int jx = 0; jx < Lx; ++jx) {
            jc = Lx*jy+jx;
            
            j_A = (2*Lx)*jy + jx;
            j_B = (2*Lx)*jy + Lx+jx;
            
            // special treatment for i == j:
            // BA is the conjugate to AB -> store only AB
            rho_flat(ftwa_index(j_A, j_A)) = rho_AB(ftwa_index(jc, jc), 0);
            rho_flat(ftwa_index(j_A, j_B)) = rho_AB(ftwa_index(jc, jc), 1);
            rho_flat(ftwa_index(j_B, j_B)) = rho_AB(ftwa_index(jc, jc), 3);
            
            for (unsigned int iy = 0; iy <= jy; ++iy) {
                for (unsigned int ix = 0; ix < Lx; ++ix) {
                    ic = Lx*iy+ix;
                    
                    i_A = (2*Lx)*iy + ix;
                    i_B = (2*Lx)*iy + Lx+ix;
                    
                    // AA
                    rho_flat(ftwa_index(i_A, j_A)) = rho_AB(ftwa_index(ic, jc), 0);
                    // AB
                    rho_flat(ftwa_index(i_A, j_B)) = rho_AB(ftwa_index(ic, jc), 1);
                    // BA (if same row: B > A)
                    if (iy == jy) {
                        rho_flat(ftwa_index(j_A, i_B)) = std::conj(rho_AB(ftwa_index(ic, jc), 2));
                    } else {
                        rho_flat(ftwa_index(i_B, j_A)) = rho_AB(ftwa_index(ic, jc), 2);
                    }
                    // BB
                    rho_flat(ftwa_index(i_B, j_B)) = rho_AB(ftwa_index(ic, jc), 3);
                    
                    if (iy == jy && ix == jx) break;
                }
            }
        }
    }
    return rho_flat;
}

arma::cx_mat FTSqrt2CellLattice::transformAB(const arma::cx_mat& rho_i_AB) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        rhoLenCells = _lattice.rhoLenCells(),
        kc, lc;
    
    arma::cx_mat rho_k_AB(rhoLenCells, 4);
    
    rho_k_AB.col(0)    = _ft.transform(rho_i_AB.unsafe_col(0));
    rho_k_AB.col(3)    = _ft.transform(rho_i_AB.unsafe_col(3));
    rho_k_AB.cols(1,2) = _ft.transform(rho_i_AB.unsafe_col(1), rho_i_AB.unsafe_col(2));
    
    std::complex<double> ux = std::complex<double>(0, 2*M_PI/(double) Lx);
    std::complex<double> uy = std::complex<double>(0, 2*M_PI/(double) Ly);  
    for (unsigned int ly = 0; ly < Ly; ++ly) {
        for (unsigned int lx = 0; lx < Lx; ++lx) {
            lc = Lx*ly + lx;
            
            for (unsigned int ky = 0; ky <= ly; ++ky) {
                for (unsigned int kx = 0; kx < Lx; ++kx) {
                    kc = Lx*ky + kx;
                    
                    rho_k_AB(ftwa_index(kc, lc), 1) *= std::exp( 0.5*(ux*(1.0*lx) + uy*(1.0*ly)));
                    rho_k_AB(ftwa_index(kc, lc), 2) *= std::exp(-0.5*(ux*(1.0*kx) + uy*(1.0*ky)));
                    rho_k_AB(ftwa_index(kc, lc), 3) *= std::exp( 0.5*(ux*(1.0*lx-1.0*kx) + uy*(1.0*ly-1.0*ky)));
                    
                    if (ky == ly && kx == lx) break;
                }
            }
        }
    }
    
    return rho_k_AB;

}

arma::cx_mat FTSqrt2CellLattice::itransformAB(const arma::cx_mat& rho_k_AB) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        rhoLenCells = _lattice.rhoLenCells(),
        kc, lc;
    
    arma::cx_mat rho_k_AB_cp = rho_k_AB;
    arma::cx_mat rho_i_AB(rhoLenCells, 4);
    
    //~ // remove unit cell phase factors
    //~ // description: see above
    std::complex<double> ux = std::complex<double>(0, 2*M_PI/(double) Lx);
    std::complex<double> uy = std::complex<double>(0, 2*M_PI/(double) Ly);  
    for (unsigned int ly = 0; ly < Ly; ++ly) {
        for (unsigned int lx = 0; lx < Lx; ++lx) {
            lc = Lx*ly + lx;
            
            for (unsigned int ky = 0; ky <= ly; ++ky) {
                for (unsigned int kx = 0; kx < Lx; ++kx) {
                    kc = Lx*ky + kx;
                    
                    rho_k_AB_cp(ftwa_index(kc, lc), 1) *= std::exp(-0.5*(ux*(1.0*lx) + uy*(1.0*ly)));
                    rho_k_AB_cp(ftwa_index(kc, lc), 2) *= std::exp( 0.5*(ux*(1.0*kx) + uy*(1.0*ky)));
                    rho_k_AB_cp(ftwa_index(kc, lc), 3) *= std::exp(-0.5*(ux*(1.0*lx-1.0*kx) + uy*(1.0*ly-1.0*ky)));
                    
                    if (ky == ly && kx == lx) break;
                }
            }
        }
    }
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    rho_i_AB.col(0)    = _ft.itransform(rho_k_AB_cp.unsafe_col(0));
    rho_i_AB.col(3)    = _ft.itransform(rho_k_AB_cp.unsafe_col(3));
    rho_i_AB.cols(1,2) = _ft.itransform(rho_k_AB_cp.unsafe_col(1), rho_k_AB_cp.unsafe_col(2));
#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    std::cout << "# TIMER Fourier transform: " << time_span.count() << "s" << std::endl;
#endif

    return rho_i_AB;
}

// cos( k_x' - A_x ) = cos( 1/sqrt(2)*(k_x - k_y) - A_x)    
arma::cx_vec create_tkmat(const Lattice& lattice,
    const HubbardHeisenbergParameters& params,
    const unitCellValues& ucv,
    double Ax, double Ay) {
    
    unsigned int Lx = lattice.getLength(0), Ly = lattice.getLength(1);

    arma::cx_mat tkmat = arma::zeros<arma::cx_mat>(lattice.numCells());
    
    std::complex<double> eikx, eiky;
    std::complex<double> ux = std::complex<double>(0, 2.0*M_PI/Lx);
    std::complex<double> uy = std::complex<double>(0, 2.0*M_PI/Ly);
    
    unsigned int n;
    for (unsigned int ny = 0; ny < Ly; ++ny) {
        for (unsigned int nx = 0; nx < Lx; ++nx) {
            n = nx + Lx * ny;
            
            eikx = std::exp(0.5*nx*ux);
            eiky = std::exp(0.5*ny*uy);
            // std::exp(std::complex<double>(0.0, -Ay)) *
            tkmat(n) = 2.0*params.t * (
                  std::cos(2.0*M_PI*(nx*0.5/Lx - ny*0.5/Ly) - Ax)
                + std::cos(2.0*M_PI*(nx*0.5/Lx + ny*0.5/Ly) - Ay)
                ) - params.J * (
                      std::conj(ucv.bonds[0] * eikx * eiky)
                    + std::conj(ucv.bonds[1] * eiky) * eikx
                    + ucv.bonds[2] * eikx * eiky
                    + ucv.bonds[3] * std::conj(eikx) * eiky
                );
        }
    }
    return tkmat;
}

arma::cx_mat FTSqrt2CellLattice::fromABtoPM(const unitCellValues& ucv, const arma::cx_mat& rho_k_AB) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        m, n, ind;
    // momentum dependencies are of the form 0.5*sqrt(2)*(kx - ky),
    // k is measured in units of 2pi/(L*sqrt(2)).
    // a = sqrt(2) is included explicitly
    // here we directly work with 2pi/L for simplicity
    std::complex<double> ux = std::complex<double>(0, 2.0*M_PI/Lx);
    std::complex<double> uy = std::complex<double>(0, 2.0*M_PI/Ly);
    std::complex<double> tm, tn, eikx, eiky, eiphim, eiphin;

    arma::cx_mat rho_k_PM(arma::size(rho_k_AB));
    
    for (unsigned int ny = 0; ny < Ly; ++ny) {
        for (unsigned int nx = 0; nx < Lx; ++nx) {
            n = nx + Lx * ny;
            
            eikx = std::exp(0.5*nx*ux);
            eiky = std::exp(0.5*ny*uy);
            tn = 2.0*_params.t * (
                  std::cos(2.0*M_PI*(nx*0.5/Lx - ny*0.5/Ly))
                + std::cos(2.0*M_PI*(nx*0.5/Lx + ny*0.5/Ly))
                ) -_params.J * (
                      std::conj(ucv.bonds[0] * eikx * eiky)
                    + std::conj(ucv.bonds[1] * eiky) * eikx
                    + ucv.bonds[2] * eikx * eiky
                    + ucv.bonds[3] * std::conj(eikx) * eiky
                );
            eiphin = tn/std::abs(tn);
        
            for (unsigned int my = 0; my < Ly; ++my) {
                for (unsigned int mx = 0; mx < Lx; ++mx) {
                    m = mx + Lx * my;
                    
                    eikx = std::exp(0.5*mx*ux);
                    eiky = std::exp(0.5*my*uy);
                    tm = 2.0*_params.t * (
                          std::cos(2.0*M_PI*(mx*0.5/Lx - my*0.5/Ly))
                        + std::cos(2.0*M_PI*(mx*0.5/Lx + my*0.5/Ly))
                        ) -_params.J * (
                              std::conj(ucv.bonds[0] * eikx * eiky)
                            + std::conj(ucv.bonds[1] * eiky) * eikx
                            + ucv.bonds[2] * eikx * eiky
                            + ucv.bonds[3] * std::conj(eikx) * eiky
                        );
                    eiphim = std::conj(tm)/std::abs(tm);

                    ind = ftwa_index(m, n);
                    rho_k_PM(ind, 0) = 0.5*(rho_k_AB(ind, 0) + eiphin*rho_k_AB(ind, 1) + eiphim*rho_k_AB(ind, 2) + eiphim*eiphin*rho_k_AB(ind, 3));
                    rho_k_PM(ind, 1) = 0.5*(rho_k_AB(ind, 0) - eiphin*rho_k_AB(ind, 1) + eiphim*rho_k_AB(ind, 2) - eiphim*eiphim*rho_k_AB(ind, 3));
                    rho_k_PM(ind, 2) = 0.5*(rho_k_AB(ind, 0) + eiphin*rho_k_AB(ind, 1) - eiphim*rho_k_AB(ind, 2) - eiphim*eiphin*rho_k_AB(ind, 3));
                    rho_k_PM(ind, 3) = 0.5*(rho_k_AB(ind, 0) - eiphin*rho_k_AB(ind, 1) - eiphim*rho_k_AB(ind, 2) + eiphim*eiphin*rho_k_AB(ind, 3));
                }
            }
        }
    }
    return rho_k_PM;
}
    
arma::cx_mat FTSqrt2CellLattice::fromPMtoAB(const unitCellValues& ucv, const arma::cx_mat& rho_k_PM) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        m, n, ind;
    // momentum dependencies are of the form 0.5*sqrt(2)*(kx - ky)*a,
    // k is measured in units of 2pi/(L*sqrt(2)) and a = sqrt(2)
    // here we directly consider 2pi/L for simplicity
    std::complex<double> ux = std::complex<double>(0, 2.0*M_PI/Lx);
    std::complex<double> uy = std::complex<double>(0, 2.0*M_PI/Ly);
    std::complex<double> tm, tn, eikx, eiky, eiphim, eiphin;

    arma::cx_mat rho_k_AB(arma::size(rho_k_PM));
    
    for (unsigned int ny = 0; ny < Ly; ++ny) {
        for (unsigned int nx = 0; nx < Lx; ++nx) {
            n = nx + Lx * ny;
            
            eikx = std::exp(0.5*nx*ux);
            eiky = std::exp(0.5*ny*uy);
            tn = 2.0*_params.t * (
                  std::cos(2.0*M_PI*(nx*0.5/Lx - ny*0.5/Ly))
                + std::cos(2.0*M_PI*(nx*0.5/Lx + ny*0.5/Ly))
                ) -_params.J * (
                      std::conj(ucv.bonds[0] * eikx * eiky)
                    + std::conj(ucv.bonds[1] * eiky) * eikx
                    + ucv.bonds[2] * eikx * eiky
                    + ucv.bonds[3] * std::conj(eikx) * eiky
                );
            eiphin = std::conj(tn)/std::abs(tn);
        
            for (unsigned int my = 0; my < Ly; ++my) {
                for (unsigned int mx = 0; mx < Lx; ++mx) {
                    m = mx + Lx * my;
                    
                    eikx = std::exp(0.5*mx*ux);
                    eiky = std::exp(0.5*my*uy);
                    tm = 2.0*_params.t * (
                          std::cos(2.0*M_PI*(mx*0.5/Lx - my*0.5/Ly))
                        + std::cos(2.0*M_PI*(mx*0.5/Lx + my*0.5/Ly))
                        ) -_params.J * (
                              std::conj(ucv.bonds[0] * eikx * eiky)
                            + std::conj(ucv.bonds[1] * eiky) * eikx
                            + ucv.bonds[2] * eikx * eiky
                            + ucv.bonds[3] * std::conj(eikx) * eiky
                        );
                    eiphim = tm/std::abs(tm);
                    
                    ind = ftwa_index(m, n);
                    rho_k_AB(ind, 0) = 0.5*(rho_k_PM(ind, 0) + rho_k_PM(ind, 1) + rho_k_PM(ind, 2) + rho_k_PM(ind, 3));
                    rho_k_AB(ind, 1) = 0.5*eiphin*(rho_k_PM(ind, 0) - rho_k_PM(ind, 1) + rho_k_PM(ind, 2) - rho_k_PM(ind, 3));
                    rho_k_AB(ind, 2) = 0.5*eiphim*(rho_k_PM(ind, 0) + rho_k_PM(ind, 1) - rho_k_PM(ind, 2) - rho_k_PM(ind, 3));
                    rho_k_AB(ind, 3) = 0.5*eiphim*eiphin*(rho_k_PM(ind, 0) - rho_k_PM(ind, 1) - rho_k_PM(ind, 2) + rho_k_PM(ind, 3));
                }
            }
        }
    }
    return rho_k_AB;
}


// Diag functions


arma::cx_mat FTSqrt2CellLattice::reshapeDiagFlatToAB(const arma::cx_vec& rho_diag_flat) const {
    unsigned int numCells = _lattice.numCells();
    
    arma::cx_mat rho_diag_AB = arma::zeros<arma::cx_mat>(numCells, 4);
    
    rho_diag_AB.col(0) = rho_diag_flat.subvec(0, numCells-1);
    rho_diag_AB.col(1) = rho_diag_flat.subvec(numCells, 2*numCells-1);
    rho_diag_AB.col(2) = rho_diag_flat.subvec(2*numCells, 3*numCells-1);
    rho_diag_AB.col(3) = rho_diag_flat.subvec(3*numCells, 4*numCells-1);
    
    return rho_diag_AB;
}
    
arma::cx_vec FTSqrt2CellLattice::reshapeDiagABToFlat(const arma::cx_mat& rho_diag_AB) const {
    unsigned int numCells = _lattice.numCells();
    
    arma::cx_vec rho_diag_flat = arma::zeros<arma::cx_vec>(4*numCells);
    
    rho_diag_flat.subvec(0, numCells-1) = rho_diag_AB.unsafe_col(0);
    rho_diag_flat.subvec(numCells, 2*numCells-1) = rho_diag_AB.unsafe_col(1);
    rho_diag_flat.subvec(2*numCells, 3*numCells-1) = rho_diag_AB.unsafe_col(2);
    rho_diag_flat.subvec(3*numCells, 4*numCells-1) = rho_diag_AB.unsafe_col(3);
    
    return rho_diag_flat;
}

arma::cx_mat FTSqrt2CellLattice::transformDiagAB(const arma::cx_mat& rho_i_AB) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        numCells = _lattice.numCells(),
        kc;
    
    arma::cx_mat rho_k_AB_diag(numCells, 4);
    
    // for (unsigned int i = 0; i < 4; ++i) {
    //  rho_k_AB_diag.col(i) = _ft.transform(rho_i_AB.unsafe_col(i))(_cell_diag_indices);
    // }
    rho_k_AB_diag.col(0)    = _ft.transform(rho_i_AB.unsafe_col(0))(_cell_diag_indices);
    rho_k_AB_diag.col(3)    = _ft.transform(rho_i_AB.unsafe_col(3))(_cell_diag_indices);
    rho_k_AB_diag.cols(1,2) = _ft.transform(rho_i_AB.unsafe_col(1), rho_i_AB.unsafe_col(2)).rows(_cell_diag_indices);
    
    // std::cout << arma::abs(rho_k_AB_diag.col(1) - arma::conj(rho_k_AB_diag.col(2))) << std::endl;
    // std::cout << "checking the FT: " << arma::norm(_ft.transform(_ft.itransform(rho_i_AB.unsafe_col(1))) - _ft.itransform(_ft.transform(rho_i_AB.unsafe_col(1)))) << std::endl;
    
    // add unit cell phase factors
    // they are exp(i sqrt(2)/2 k) since a = sqrt(2)
    // and the B sites are shifted by a/2
    // however, k is measured in units 2pi/(L*sqrt(2))
    // and hence the factors are exp(i 2pi/(2*L) n_k)
    // = exp(i 0.5*uk n_k)
    std::complex<double> ukx = std::complex<double>(0, 2*M_PI/(double) Lx);
    std::complex<double> uky = std::complex<double>(0, 2*M_PI/(double) Ly);
    for (unsigned int ky = 0; ky < Ly; ++ky) {
        for (unsigned int kx = 0; kx < Lx; ++kx) {
            kc = Lx*ky+kx;
            rho_k_AB_diag(kc, 1) *= std::exp(0.5*(ukx*(1.0*kx) + uky*(1.0*ky)));
            rho_k_AB_diag(kc, 2) *= std::exp(0.5*(-ukx*(1.0*kx) - uky*(1.0*ky)));
        }
    }
    
    return rho_k_AB_diag;
}

arma::cx_mat FTSqrt2CellLattice::itransformDiagAB(const arma::cx_mat& rho_k_diag_AB) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        // numCells = _lattice.numCells(),
        rhoLenCells = _lattice.rhoLenCells(),
        kc;
    
    arma::cx_mat rho_k_diag_AB_cp = rho_k_diag_AB;
    arma::cx_mat rho_k_AB = arma::zeros<arma::cx_mat>(rhoLenCells, 4);
    arma::cx_mat rho_i_AB(rhoLenCells, 4);
    
    //~ // remove unit cell phase factors
    //~ // description: see above
    std::complex<double> ukx = std::complex<double>(0, 2*M_PI/(double) Lx);
    std::complex<double> uky = std::complex<double>(0, 2*M_PI/(double) Ly);
    for (unsigned int ky = 0; ky < Ly; ++ky) {
        for (unsigned int kx = 0; kx < Lx; ++kx) {
            kc = Lx*ky+kx;
            rho_k_diag_AB_cp(kc, 1) *= std::exp(-0.5*(ukx*(1.0*kx) + uky*(1.0*ky)));
            rho_k_diag_AB_cp(kc, 2) *= std::exp(-0.5*(-ukx*(1.0*kx) - uky*(1.0*ky)));
        }
    }
    
    rho_k_AB.rows(_cell_diag_indices) = rho_k_diag_AB_cp;
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    rho_i_AB.col(0)    = _ft.itransform(rho_k_AB.unsafe_col(0));
    rho_i_AB.col(3)    = _ft.itransform(rho_k_AB.unsafe_col(3));
    rho_i_AB.cols(1,2) = _ft.itransform(rho_k_AB.unsafe_col(1), rho_k_AB.unsafe_col(2));
#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    std::cout << "# TIMER Fourier transform: " << time_span.count() << "s" << std::endl;
#endif
    
    // arma::cx_mat rho_k_AB_check(rhoLenCells, 4);
    // rho_k_AB_check.col(0)    = _ft.transform(rho_i_AB.unsafe_col(0));
    // rho_k_AB_check.col(3)    = _ft.transform(rho_i_AB.unsafe_col(3));
    // rho_k_AB_check.cols(1,2) = _ft.transform(rho_i_AB.unsafe_col(1), rho_i_AB.unsafe_col(2));
    // std::cout << "rho_k_AB check:" << std::endl << arma::norm(rho_k_AB - rho_k_AB_check) << std::endl;
    
    return rho_i_AB;
}

// transform rho_{kk} from the AB basis to the PM basis
// 0 ++
// 1 +-
// 2 -+
// 3 --
arma::cx_mat FTSqrt2CellLattice::fromDiagABtoPM(const unitCellValues& ucv, const arma::cx_mat& rho_k_diag_AB) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        m;
    // momentum dependencies are of the form 0.5*sqrt(2)*(kx - ky)*a,
    // k is measured in units of 2pi/(L*sqrt(2)) and a = sqrt(2)
    // here we directly consider 2pi/L for simplicity
    std::complex<double> ukx = std::complex<double>(0, 2.0*M_PI/Lx);
    std::complex<double> uky = std::complex<double>(0, 2.0*M_PI/Ly);
    std::complex<double> eikx, eiky, tk, eiphi;
    double re, im;

    arma::cx_mat rho_k_diag_PM(arma::size(rho_k_diag_AB));
        
    for (unsigned int my = 0; my < Ly; ++my) {
        for (unsigned int mx = 0; mx < Lx; ++mx) {
            m = mx + Lx * my;
            
            eikx = std::exp(0.5*mx*ukx);
            eiky = std::exp(0.5*my*uky);
            
            tk = 2.0*_params.t * (
                  std::cos(2.0*M_PI*(mx*0.5/Lx - my*0.5/Ly))
                + std::cos(2.0*M_PI*(mx*0.5/Lx + my*0.5/Ly))
                ) -_params.J * (
                      std::conj(ucv.bonds[0] * eikx * eiky)
                    + std::conj(ucv.bonds[1] * eiky) * eikx
                    + ucv.bonds[2] * eikx * eiky
                    + ucv.bonds[3] * std::conj(eikx) * eiky
                );
                    
            eiphi = tk/std::abs(tk);
            re = (eiphi*rho_k_diag_AB(m, 1)).real();
            im = (eiphi*rho_k_diag_AB(m, 1)).imag();

            rho_k_diag_PM(m, 0) = 0.5*(rho_k_diag_AB(m, 0) + rho_k_diag_AB(m, 3) + 2.0*re);
            
            rho_k_diag_PM(m, 1) = 0.5*(rho_k_diag_AB(m, 0) - rho_k_diag_AB(m, 3) - std::complex<double>(0.0, 2.0*im));
            
            rho_k_diag_PM(m, 2) = std::conj(rho_k_diag_PM(m, 1));
            
            rho_k_diag_PM(m, 3) = 0.5*(rho_k_diag_AB(m, 0) + rho_k_diag_AB(m, 3) - 2.0*re);
        }
    }
    return rho_k_diag_PM;
}

// transform rho_{kk} from the PM basis to the AB basis
arma::cx_mat FTSqrt2CellLattice::fromDiagPMtoAB(const unitCellValues& ucv, const arma::cx_mat& rho_k_diag_PM) const {
    unsigned int Lx = _lattice.getLength(0),
        Ly = _lattice.getLength(1),
        m;
    // momentum dependencies are of the form 0.5*sqrt(2)*(kx - ky)*a,
    // k is measured in units of 2pi/(L*sqrt(2)) and a = sqrt(2)
    // here we directly consider 2pi/L for simplicity
    std::complex<double> ukx = std::complex<double>(0, 2.0*M_PI/Lx);
    std::complex<double> uky = std::complex<double>(0, 2.0*M_PI/Ly);
    std::complex<double> eikx, eiky, tk, eiphi;
    double re, im;

    arma::cx_mat rho_k_diag_AB(arma::size(rho_k_diag_PM));
        
    for (unsigned int my = 0; my < Ly; ++my) {
        for (unsigned int mx = 0; mx < Lx; ++mx) {
            m = mx + Lx * my;
            
            eikx = std::exp(0.5*mx*ukx);
            eiky = std::exp(0.5*my*uky);
            
            tk = 2.0*_params.t * (
                  std::cos(2.0*M_PI*(mx*0.5/Lx - my*0.5/Ly))
                + std::cos(2.0*M_PI*(mx*0.5/Lx + my*0.5/Ly))
                ) -_params.J * (
                      std::conj(ucv.bonds[0] * eikx * eiky)
                    + std::conj(ucv.bonds[1] * eiky) * eikx
                    + ucv.bonds[2] * eikx * eiky
                    + ucv.bonds[3] * std::conj(eikx) * eiky
                );
                    
            eiphi = std::conj(tk)/std::abs(tk);
            re = rho_k_diag_PM(m, 1).real();
            im = rho_k_diag_PM(m, 1).imag();

            rho_k_diag_AB(m, 0) = 0.5*(rho_k_diag_PM(m, 0) + rho_k_diag_PM(m, 3) + 2.0*re);
            
            rho_k_diag_AB(m, 1) = 0.5*eiphi*(rho_k_diag_PM(m, 0) - rho_k_diag_PM(m, 3) - std::complex<double>(0.0, 2.0*im));
            
            rho_k_diag_AB(m, 2) = std::conj(rho_k_diag_AB(m, 1));
            
            rho_k_diag_AB(m, 3) = 0.5*(rho_k_diag_PM(m, 0) + rho_k_diag_PM(m, 3) - 2.0*re);
        }
    }
    return rho_k_diag_AB;
}

}
