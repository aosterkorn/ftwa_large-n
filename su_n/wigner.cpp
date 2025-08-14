// su_n/wigner.cpp

#include "wigner.hpp"

namespace ftwa_su_n {

WignerFuncProduct::WignerFuncProduct(
    const Lattice& lattice
) : _lattice(lattice), _nf(2), _wignerFuncModel(WignerFuncModel::Gaussian), _occupations() { }

WignerFuncProduct::WignerFuncProduct(const Lattice& lattice, unsigned int nf,
    WignerFuncModel wignerFuncModel
) : _lattice(lattice), _nf(nf), _wignerFuncModel(wignerFuncModel) { }

WignerFuncProduct::WignerFuncProduct(const Lattice& lattice, unsigned int nf,
    WignerFuncModel wignerFuncModel, const arma::vec& occupations
) : _lattice(lattice), _nf(nf), _wignerFuncModel(wignerFuncModel), _occupations(occupations) { }

// TODO(include WignerFuncModel)
void WignerFuncProduct::generate(std::mt19937& generator, arma::cx_vec& res) const {
    unsigned int V = _lattice.numSites();
    unsigned int rhoLen = _lattice.rhoLen();
    res.set_size(rhoLen);
    
    for (unsigned int n = 0; n < V; ++n) {
        // m == n
        std::normal_distribution<double> dist(
            _occupations[n] - 0.5,
            std::sqrt(1.0/_nf * _occupations[n]*(1.0-_occupations[n]))
        );
        res(ftwa_index(n, n)) = dist(generator);
        for (unsigned int m = 0; m < n; ++m) {
            std::normal_distribution<double> dist(
                0.0,
                std::sqrt(0.25/_nf * (_occupations[m] + _occupations[n] - 2.0*_occupations[m]*_occupations[n]))
            );
            res(ftwa_index(m, n)) = dist(generator) + std::complex<double>(0.0, 1.0) * dist(generator);
        }
    }
}

WignerFuncFermiSeaGS::WignerFuncFermiSeaGS(
    const Lattice& lattice,
    unsigned int nf,
    WignerFuncModel wignerFuncModel,
    unsigned int num_particles
    ) : WignerFuncProduct(lattice, nf, wignerFuncModel) {
        fermiSurface2D(lattice, num_particles, _occupations);
        // plotFermiSurface2D(lattice, kvals);
    }

WignerFuncFermiSeaGS::WignerFuncFermiSeaGS(
    const Lattice& lattice,
    unsigned int nf,
    WignerFuncModel wignerFuncModel,
    const arma::vec& occupations
    ) : WignerFuncProduct(lattice, nf, wignerFuncModel, occupations) { }

void WignerFuncFermiSeaGS::generate(std::mt19937& generator, arma::cx_vec& res) const {
    unsigned int V = _lattice.numSites();
    unsigned int rhoLen = _lattice.rhoLen();
    res.set_size(rhoLen);
    
    double eps = 1e-12;
    double sig = std::sqrt(0.25/_nf);
    std::normal_distribution<double> dist_gaussian(0.0, sig);
    std::bernoulli_distribution      dist_twopoint(0.5);
    
    for (unsigned int n = 0; n < V; ++n) {
        for (unsigned int m = 0; m <= n; ++m) {
            if (_occupations[m] < eps && _occupations[n] < eps) {
                res(ftwa_index(m, n)) = ((m == n) ? -0.5 : 0.0);
            } else if ((_occupations[m] < eps && (_occupations[n]-1.0) < eps) || ((_occupations[m]-1.0) < eps && _occupations[n] < eps)) {
                switch(_wignerFuncModel) {
                    case WignerFuncModel::Gaussian:
                        res(ftwa_index(m, n)) = dist_gaussian(generator)
                                              + dist_gaussian(generator) * std::complex<double>(0.0, 1.0);
                        break;
                    case WignerFuncModel::TwoPoint:
                        res(ftwa_index(m, n)) = (dist_twopoint(generator) ? 1.0 : -1.0) * sig
                                              + (dist_twopoint(generator) ? 1.0 : -1.0) * sig * std::complex<double>(0.0, 1.0);
                        break;
                }
            } else if ((_occupations[m]-1.0) < eps && (_occupations[n]-1.0) < eps) {
                res(ftwa_index(m, n)) = ((m == n) ? 0.5 : 0.0);
            } else {
                std::cerr << "# ERROR: unexpected occupation number" << std::endl;
            }
        }
    }
}

double WignerFuncFermiSeaTemp::_particleNumberFromChemPot(double chemPot) {
    unsigned int V = _lattice.numSites();
    double numParticles = 0.0;
    for (unsigned int m = 0; m < V; ++m) {
        numParticles += fermiDirac(_energies(m), chemPot, _temp);
    }
    return (numParticles - _numParticles);
}

WignerFuncFermiSeaTemp::WignerFuncFermiSeaTemp(
    const Lattice& lattice,
    unsigned int nf,
    WignerFuncModel wignerFuncModel,
    unsigned int numParticles,
    double temp
    ) : WignerFuncProduct(lattice, nf, wignerFuncModel),
        _numParticles(numParticles), _temp(temp) {
    unsigned int L = _lattice.getLength(0);
    unsigned int V = _lattice.numSites();
    
    _occupations.set_size(V);
    _energies.set_size(V);
    
    double pNum = 0.0;
    
    std::vector<unsigned int> ks = {};
    for (unsigned int m = 0; m < V; ++m) {
        ks.push_back(m);
    }
    std::sort(ks.begin(), ks.end(), std::bind(sortFermi2D, L, std::placeholders::_1, std::placeholders::_2));
    for (unsigned int m = 0; m < V; ++m) {
        _energies(m) = dispTightBinding2d(L, ks[m] % L, ks[m] / L);
    }
    
    auto func = std::bind(
        &WignerFuncFermiSeaTemp::_particleNumberFromChemPot,
        this,
        std::placeholders::_1
    );
    std::pair<double, double> res = boost::math::tools::bisect(
        func, -4.0, 4.0,
        boost::math::tools::eps_tolerance<double>(24)
    );
    _chemPot = 0.5*(res.first + res.second);
    
    for (unsigned int m = 0; m < V; ++m) {
        _occupations[ks[m]] = fermiDirac(_energies(m), _chemPot, _temp);
        pNum += _occupations[ks[m]];
    }
    std::cout << "# particle number: " << pNum << std::endl;
    std::cout << "# chemical potential: " << _chemPot << std::endl;
}

HubHieGaussWigner::HubHieGaussWigner(const Lattice& lattice, unsigned int nf,
    WignerFuncModel wignerFuncModel, unsigned int num_particles, const FourierTransformer2dPBC& ft)
    : WignerFuncFermiSeaGS(lattice, nf, wignerFuncModel, num_particles), _ft(ft) { }
    
void HubHieGaussWigner::generate(std::mt19937& generator, arma::cx_vec& res) const {
    unsigned int V = _lattice.numSites();
    unsigned int rhoLen = _lattice.rhoLen();
    
    arma::cx_vec rho          = arma::zeros<arma::cx_vec>(rhoLen);
    arma::cx_vec rho_nonfluct = arma::zeros<arma::cx_vec>(rhoLen);
    arma::cx_mat Dmat         = arma::zeros<arma::cx_mat>(rhoLen, V*V); 
    
    std::normal_distribution<double> dist(0.0, std::sqrt(0.25/_nf));
    for (unsigned int n = 0; n < V; ++n) {
        for (unsigned int m = 0; m <= n; ++m) {
            if (_occupations[m] == 0.0 && _occupations[n] == 0.0) {
                rho(ftwa_index(m, n))          = (m == n) ? -0.5 : 0.0;
                rho_nonfluct(ftwa_index(m, n)) = (m == n) ? -0.5 : 0.0;
            } else if ((_occupations[m] == 0.0 && _occupations[n] == 1.0) || (_occupations[m] == 1.0 && _occupations[n] == 0.0)) {
                rho(ftwa_index(m, n)) = dist(generator) + std::complex<double>(0.0, 1.0) * dist(generator);
            } else if (_occupations[m] == 1.0 && _occupations[n] == 1.0) {
                rho(ftwa_index(m, n))          = (m == n) ? 0.5 : 0.0;
                rho_nonfluct(ftwa_index(m, n)) = (m == n) ? 0.5 : 0.0;
            } else {
                std::cerr << "# ERROR: unexpected occupation number" << std::endl;
            }
        }
    }
    
    arma::cx_vec rho_pos, rho_nonfluct_pos;
    _ft.itransform(rho, rho_pos);
    _ft.itransform(rho_nonfluct, rho_nonfluct_pos);
    
    unsigned int ctr = 0;
    for (unsigned int j = 0; j < V; ++j) {
        for (unsigned int i = 0; i <= j; ++i) {
            for (unsigned int n = 0; n < V; ++n) {
                for (unsigned int m = 0; m < V; ++m) {
                    Dmat(ctr, n*V+m) = rhoVal(rho_pos, i, j)*rhoVal(rho_pos, m, n);
                }
            }
            ctr++;
        }
    }
    
    res.set_size(rhoLen + rhoLen*V*V);
    res.head(rhoLen)     = rho_pos;
    res.tail(rhoLen*V*V) = Dmat.as_col();
}

////////////////////////////////////////////////////////////////////////
////////////////////// Hubbard-Heisenberg model ////////////////////////
////////////////////////////////////////////////////////////////////////

void HubHeiGaussWignerInfty::generate(arma::cx_vec& res) const {
    unsigned int numCells = _lattice.numCells();
    
    arma::cx_mat rho_k_diag_PM(numCells, 4), rho_k_diag_AB;
    
    // half filling assumed at the moment !
    rho_k_diag_PM.col(0).fill(-0.5);
    rho_k_diag_PM.col(1).fill(0.0);
    rho_k_diag_PM.col(2).fill(0.0);
    rho_k_diag_PM.col(3).fill(0.5);
    
    rho_k_diag_AB = _ft.fromDiagPMtoAB(_ucv, rho_k_diag_PM);
    rho_k_diag_AB.clean(1e-12);
    arma::cx_mat rho_i_AB = _ft.itransformDiagAB(rho_k_diag_AB);
    rho_i_AB.clean(1e-12);
    // std::cout << rho_i_AB.col(1) + rho_i_AB.col(2) << std::endl;
    
    res = _ft.reshapeABToFlat(rho_i_AB);
}

void HubHeiGaussWignerInfty::generateMom(arma::cx_vec& res) const {
    unsigned int numCells = _lattice.numCells();
    
    arma::cx_mat rho_k_diag_PM(numCells, 4), rho_k_diag_AB;
    
    // half filling assumed at the moment !
    rho_k_diag_PM.col(0).fill(-0.5);
    rho_k_diag_PM.col(1).fill(0.0);
    rho_k_diag_PM.col(2).fill(0.0);
    rho_k_diag_PM.col(3).fill(0.5);
    
    rho_k_diag_AB = _ft.fromDiagPMtoAB(_ucv, rho_k_diag_PM);
    rho_k_diag_AB.clean(1e-12);
    
    res = _ft.reshapeDiagABToFlat(rho_k_diag_AB);
}

void HubHeiGaussWignerFiniteN::generate(arma::cx_vec& res) const {
    unsigned int rhoLenCells = _lattice.rhoLenCells();
    
    arma::uvec cell_diag_indices = _ft.getCellDiagIndices();
    
    arma::cx_mat rho_k_PM = arma::zeros<arma::cx_mat>(rhoLenCells, 4);
    std::normal_distribution<double> dist(0.0, std::sqrt(0.25/_n));
    
    // half filling assumed at the moment !
    rho_k_PM.submat(cell_diag_indices, arma::uvec{0}).fill(-0.5);
    rho_k_PM.col(1).imbue( [&]() { return std::complex<double>(dist(_generator), dist(_generator)); } );
    rho_k_PM.col(2).imbue( [&]() { return std::complex<double>(dist(_generator), dist(_generator)); } );
    rho_k_PM.submat(cell_diag_indices, arma::uvec{2}) = arma::conj(rho_k_PM.submat(cell_diag_indices, arma::uvec{1}));
    rho_k_PM.submat(cell_diag_indices, arma::uvec{3}).fill(+0.5);
    
    // std::cout << "rho_k_PM = " << std::endl << rho_k_PM << std::endl;
    arma::cx_mat rho_k_AB = _ft.fromPMtoAB(_ucv, rho_k_PM);
    // std::cout << "rho_k_AB = " << std::endl << rho_k_AB << std::endl;
    arma::cx_mat rho_i_AB = _ft.itransformAB(rho_k_AB);
    res = _ft.reshapeABToFlat(rho_i_AB);
}

void HubHeiGaussWignerFiniteNDiagFluct::generate(arma::cx_vec& res) const {
    unsigned int numCells = _lattice.numCells();
    
    arma::cx_mat rho_k_diag_PM(numCells, 4), rho_k_diag_AB;
    std::normal_distribution<double> dist(0.0, std::sqrt(0.25/_n));
    
    // half filling assumed at the moment !
    rho_k_diag_PM.col(0).fill(-0.5);
    rho_k_diag_PM.col(1).imbue( [&]() { return std::complex<double>(dist(_generator), dist(_generator)); } );
    rho_k_diag_PM.col(2) = arma::conj(rho_k_diag_PM.col(1));
    rho_k_diag_PM.col(3).fill(0.5);
    
    rho_k_diag_AB = _ft.fromDiagPMtoAB(_ucv, rho_k_diag_PM);
    arma::cx_mat rho_i_AB = _ft.itransformDiagAB(rho_k_diag_AB);
    res = _ft.reshapeABToFlat(rho_i_AB);
}

//~ void HubHeiGaussWignerInfty::generate(arma::cx_vec& res) const {
    //~ // half filling assumed at the moment
    
    //~ unsigned int Lx = _lattice.getLength(0), Ly = _lattice.getLength(1),
        //~ rhoLenCells = _lattice.rhoLenCells(), m;
    //~ std::complex<double> tk, eikx, eiky,
        //~ omega_x = std::exp(std::complex<double>(0, 2*M_PI/(double) Lx)),
        //~ omega_y = std::exp(std::complex<double>(0, 2*M_PI/(double) Ly));
    //~ arma::cx_vec rho_k_AA = arma::zeros<arma::cx_vec>(rhoLenCells),
        //~ rho_k_AB = arma::zeros<arma::cx_vec>(rhoLenCells),
        //~ rho_k_BA,
        //~ rho_AA, rho_AB, rho_BA;
        
    //~ for (unsigned int my = 0; my < Ly; ++my) {
        //~ for (unsigned int mx = 0; mx < Lx; ++mx) {
            //~ eikx = std::pow(omega_x, mx);
            //~ eiky = std::pow(omega_y, my);
            //~ tk = _params.t * (1.0 + eikx + eiky + eikx*eiky)
                //~ -_params.J * (std::conj(_ucv.bonds[0]) + std::conj(_ucv.bonds[1])*eikx
                    //~ + _ucv.bonds[2]*eikx*eiky + _ucv.bonds[3]*eiky);
            //~ // tk *= std::pow(eikx*eiky, 0.5);
            //~ m = mx + Lx * my;
            //~ rho_k_AA(ftwa_index(m, m)) = 0.5*(-0.5 + 0.5);
            //~ rho_k_AB(ftwa_index(m, m)) = 0.5*std::conj(tk)/std::abs(tk)*(-0.5 - 0.5);
        //~ }
    //~ }
    //~ rho_k_BA = arma::conj(rho_k_AB);
    
    //~ // Fourier transform from k to ij space.
    //~ _ft.itransform(rho_k_AA, rho_AA);
    //~ _ft.itransform(rho_k_AB, rho_AB);
    //~ _ft.itransform(rho_k_BA, rho_BA);
    
    //~ // iterate over all pairs of unit cells and assign the bond values to
    //~ // the sites in tilted lattice enumeration
    //~ // boundary conditions should be respected by the Fourier transform
    //~ res.set_size(_lattice.rhoLen());
    
    //~ for (unsigned int jy = 0; jy < Ly; ++jy) {
        //~ for (unsigned int jx = 0; jx < Lx; ++jx) {
            //~ // std::cout << "unit cell j = " << Lx*jy + jx << " (jx, jy = " << jx << ", " << jy << ")" << std::endl;
            
            //~ // special treatment for i == j
            //~ // A - A
            //~ res(ftwa_index((2*Lx)*jy + jx, (2*Lx)*jy + jx)) = rho_AA(ftwa_index(Lx*jy+jx, Lx*jy+jx));
            //~ // B - B
            //~ res(ftwa_index((2*Lx)*jy + Lx+jx, (2*Lx)*jy + Lx+jx)) = rho_AA(ftwa_index(Lx*jy+jx, Lx*jy+jx));
            //~ // A - B
            //~ res(ftwa_index((2*Lx)*jy + jx, (2*Lx)*jy + Lx+jx)) = rho_AB(ftwa_index(Lx*jy+jx, Lx*jy+jx));
            
            //~ for (unsigned int iy = 0; iy <= jy; ++iy) {
                //~ for (unsigned int ix = 0; ix < Lx; ++ix) {
                    //~ if (iy == jy && ix == jx) break;
                    
                    //~ // A - A
                    //~ res(ftwa_index((2*Lx)*iy + ix, (2*Lx)*jy + jx)) = rho_AA(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ // B - B
                    //~ res(ftwa_index((2*Lx)*iy + Lx+ix, (2*Lx)*jy + Lx+jx)) = rho_AA(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ // A - B
                    //~ res(ftwa_index((2*Lx)*iy + ix, (2*Lx)*jy + Lx+jx)) = rho_AB(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ // B - A
                    //~ if (iy == jy) {
                        //~ res(ftwa_index((2*Lx)*jy + jx, (2*Lx)*iy + Lx+ix)) = std::conj(rho_BA(ftwa_index(Lx*iy+ix, Lx*jy+jx)));
                    //~ } else {
                        //~ res(ftwa_index((2*Lx)*iy + Lx+ix, (2*Lx)*jy + jx)) = rho_BA(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ }
                //~ }
            //~ }
        //~ }
    //~ }
//~ }

//~ void HubHeiGaussWignerFiniteN::generate(arma::cx_vec& res) const {
    //~ // half filling assumed at the moment
    //~ unsigned int Lx = _lattice.getLength(0), Ly = _lattice.getLength(1),
        //~ rhoLenCells = _lattice.rhoLenCells(), m;
    //~ double re, im;
    //~ std::complex<double> tk, eikx, eiky,
        //~ omega_x = std::exp(std::complex<double>(0, 2*M_PI/(double) Lx)),
        //~ omega_y = std::exp(std::complex<double>(0, 2*M_PI/(double) Ly));
    //~ arma::cx_vec rho_k_AA = arma::zeros<arma::cx_vec>(rhoLenCells),
        //~ rho_k_BB = arma::zeros<arma::cx_vec>(rhoLenCells),
        //~ rho_k_AB = arma::zeros<arma::cx_vec>(rhoLenCells), rho_k_BA,
        //~ rho_AA, rho_BB, rho_AB, rho_BA;
    //~ std::normal_distribution<double> dist(0.0, std::sqrt(0.25/_n));
        
    //~ for (unsigned int my = 0; my < Ly; ++my) {
        //~ for (unsigned int mx = 0; mx < Lx; ++mx) {
            //~ eikx = std::pow(omega_x, mx);
            //~ eiky = std::pow(omega_y, my);
            //~ tk = _params.t * (1.0 + eikx + eiky + eikx*eiky)
                //~ -_params.J * (std::conj(_ucv.bonds[0]) + std::conj(_ucv.bonds[1])*eikx
                    //~ + _ucv.bonds[2]*eikx*eiky + _ucv.bonds[3]*eiky);
            //~ // tk *= std::pow(eikx*eiky, 0.5);
            //~ m = mx + Lx * my;
            //~ re = dist(_generator);
            //~ im = dist(_generator);
            //~ rho_k_AA(ftwa_index(m, m)) = 0.5*(-0.5 + 2.0*re + 0.5);
            //~ rho_k_BB(ftwa_index(m, m)) = 0.5*(-0.5 - 2.0*re + 0.5);
            //~ rho_k_AB(ftwa_index(m, m)) = 0.5*std::conj(tk)/std::abs(tk)*(-0.5 -2.0*im - 0.5);
        //~ }
    //~ }
    //~ rho_k_BA = arma::conj(rho_k_AB);
    
    //~ // Fourier transform from k to ij space.
    //~ _ft.itransform(rho_k_AA, rho_AA);
    //~ _ft.itransform(rho_k_BB, rho_BB);
    //~ _ft.itransform(rho_k_AB, rho_AB);
    //~ _ft.itransform(rho_k_BA, rho_BA);
    
    //~ // iterate over all pairs of unit cells and assign the bond values to
    //~ // the sites in tilted lattice enumeration
    //~ // boundary conditions should be respected by the Fourier transform
    //~ res.set_size(_lattice.rhoLen());
    
    //~ for (unsigned int jy = 0; jy < Ly; ++jy) {
        //~ for (unsigned int jx = 0; jx < Lx; ++jx) {
            //~ // std::cout << "unit cell j = " << Lx*jy + jx << " (jx, jy = " << jx << ", " << jy << ")" << std::endl;
            
            //~ // special treatment for i == j
            //~ // A - A
            //~ res(ftwa_index((2*Lx)*jy + jx, (2*Lx)*jy + jx)) = rho_AA(ftwa_index(Lx*jy+jx, Lx*jy+jx));
            //~ // B - B
            //~ res(ftwa_index((2*Lx)*jy + Lx+jx, (2*Lx)*jy + Lx+jx)) = rho_BB(ftwa_index(Lx*jy+jx, Lx*jy+jx));
            //~ // A - B
            //~ res(ftwa_index((2*Lx)*jy + jx, (2*Lx)*jy + Lx+jx)) = rho_AB(ftwa_index(Lx*jy+jx, Lx*jy+jx));
            
            //~ for (unsigned int iy = 0; iy <= jy; ++iy) {
                //~ for (unsigned int ix = 0; ix < Lx; ++ix) {
                    //~ if (iy == jy && ix == jx) break;
                    //~ // A - A
                    //~ res(ftwa_index((2*Lx)*iy + ix, (2*Lx)*jy + jx)) = rho_AA(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ // B - B
                    //~ res(ftwa_index((2*Lx)*iy + Lx+ix, (2*Lx)*jy + Lx+jx)) = rho_BB(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ // A - B
                    //~ res(ftwa_index((2*Lx)*iy + ix, (2*Lx)*jy + Lx+jx)) = rho_AB(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ // B - A
                    //~ if (iy == jy) {
                        //~ res(ftwa_index((2*Lx)*jy + jx, (2*Lx)*iy + Lx+ix)) = std::conj(rho_BA(ftwa_index(Lx*iy+ix, Lx*jy+jx)));
                    //~ } else {
                        //~ res(ftwa_index((2*Lx)*iy + Lx+ix, (2*Lx)*jy + jx)) = rho_BA(ftwa_index(Lx*iy+ix, Lx*jy+jx));
                    //~ }
                //~ }
            //~ }
        //~ }
    //~ }
//~ }

}
