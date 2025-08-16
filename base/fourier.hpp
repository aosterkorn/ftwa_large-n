#ifndef FOURIER_HPP
#define FOURIER_HPP

#ifdef FTWA_WITH_TIMER
#include <chrono>
#endif

#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include <random>

#include <armadillo>

#include "basic_defs.hpp"
#include "lattice.hpp"

namespace ftwa_su_n {

/**
 * Fourier transform of a fTWA rho vector in two dimensions with periodic boundary conditions.
 * Momentum entries are labelled according to m = 0, ..., L-1 with k = 2pi/L * m
 */
class FourierTransformer2dPBC {
  public:
    FourierTransformer2dPBC(const Lattice& lattice);
    
    /**
     * Transform a rho vector from position to momentum space.
     */
    template<typename T>
    void transform(const arma::Col<T>& rho, arma::Col<T>& res) const {
        unsigned int V = _lattice.numCells();
        arma::Mat<T> rhoMat(V, V);
        res.set_size(_lattice.rhoLenCells());
        
        rhoMat(_upper_indices) = rho;
        rhoMat = _ftMat * arma::symmatu(rhoMat) * _ftMatConj;
        
        res = rhoMat(_upper_indices);
    }
    
    template<typename T>
    arma::Col<T> transform(const arma::Col<T>& rho) const {
        unsigned int V = _lattice.numCells();
        arma::Mat<T> rhoMat(V, V);
        
        rhoMat(_upper_indices) = rho;
        
        rhoMat = _ftMat * arma::symmatu(rhoMat) * _ftMatConj;
        
        return rhoMat(_upper_indices).as_col();
    }
    
    template<typename T>
    arma::Mat<T> transform(const arma::Col<T>& rho, const arma::Col<T>& rho_conj) const {
        unsigned int V = _lattice.numCells(), rhoLenCells = _lattice.rhoLenCells();
        arma::Mat<T> rhoMat(V, V);
        arma::Mat<T> res(rhoLenCells, 2);
        
        rhoMat(_upper_indices) = rho_conj;
        arma::inplace_trans(rhoMat);
        rhoMat(_upper_indices) = rho;
        
        rhoMat = _ftMat * rhoMat * _ftMatConj;
        
        res.col(0) = rhoMat(_upper_indices);
        arma::inplace_trans(rhoMat);
        res.col(1) = rhoMat(_upper_indices);
        
        return res;
    }
    
    /**
     * Transform a rho vector from momentum to position space.
     */
    template<typename T>
    void itransform(const arma::Col<T>& rho, arma::Col<T>& res) const {
        unsigned int V = _lattice.numCells();
        arma::cx_mat rhoMat(V, V);
        res.set_size(_lattice.rhoLenCells());
        
        rhoMat(_upper_indices) = rho;
        rhoMat = _ftMatConj * arma::symmatu(rhoMat) * _ftMat;
        
        res = rhoMat(_upper_indices);
    }
    
    template<typename T>
    arma::Col<T> itransform(const arma::Col<T>& rho) const {
        unsigned int V = _lattice.numCells();
        arma::cx_mat rhoMat(V, V);

#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
        rhoMat(_upper_indices) = rho;

#ifdef FTWA_WITH_TIMER
    auto snap1 = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
    std::cout << "# TIMER reshaping: " << time_span.count() << "s" << std::endl;
#endif
        rhoMat = _ftMatConj * arma::symmatu(rhoMat) * _ftMat;
#ifdef FTWA_WITH_TIMER
    auto snap2 = sc.now();
    time_span = static_cast<std::chrono::duration<double>>(snap2 - snap1);
    std::cout << "# TIMER mat-mat mult: " << time_span.count() << "s" << std::endl;
#endif
        
        return rhoMat(_upper_indices).as_col();
    }
    
    template<typename T>
    arma::Mat<T> itransform(const arma::Col<T>& rho, const arma::Col<T>& rho_conj) const {
        unsigned int V = _lattice.numCells(), rhoLenCells = _lattice.rhoLenCells();
        arma::Mat<T> rhoMat(V, V);
        arma::Mat<T> res(rhoLenCells, 2);
        
        rhoMat(_upper_indices) = rho_conj;
        arma::inplace_trans(rhoMat);
        rhoMat(_upper_indices) = rho;
        
        rhoMat = _ftMatConj * rhoMat * _ftMat;
        
        res.col(0) = rhoMat(_upper_indices);
        arma::inplace_trans(rhoMat);
        res.col(1) = rhoMat(_upper_indices);
        
        return res;
    }
    
  private:
    const Lattice& _lattice;
    arma::uvec _upper_indices;
    arma::cx_mat _ftMat;
    arma::cx_mat _ftMatConj;
};

// Hubbard-Heisenberg model

struct unitCellValues {
    double rhoA, rhoB;
    std::complex<double> bonds[4];
};

/**
 * Read the "unit cell values" of an equilibrium state from a file.
 */
void readUCVFromFile(const std::string& filename, struct unitCellValues& ucv);

arma::cx_vec create_tkmat(const Lattice& lattice,
    const HubbardHeisenbergParameters& params,
    const unitCellValues& ucv,
    double Ax, double Ay);


/**
 * Fourier transform for the Hubbard-Heisenberg model on a tilted square lattice (two sites per unit cell).
 * This class is used to transform the rho vector from position to momentum space and vice versa.
 */
class FTSqrt2CellLattice {
  public:   
    FTSqrt2CellLattice(
        const TiltedSquareLattice& lattice,
        const HubbardHeisenbergParameters& params,
        const FourierTransformer2dPBC& ft
    ) : _lattice(lattice), _params(params), _ft(ft) { 
        _cell_diag_indices = arma::uvec(_lattice.numCells());
        for (arma::uword i = 0; i < _lattice.numCells(); ++i) {
            _cell_diag_indices(i) = ftwa_index(i, i);
        }
    };
    
    // ~FTSqrt2CellLattice() { };
    
    arma::uvec getCellDiagIndices() const { return _cell_diag_indices; }
    
    arma::cx_mat reshapeFlatToAB(const arma::cx_vec& rho_flat) const;
    
    arma::cx_vec reshapeABToFlat(const arma::cx_mat& rho_AB) const;
    
    arma::cx_mat transformAB(const arma::cx_mat& rho_i_AB) const;
    
    arma::cx_mat itransformAB(const arma::cx_mat& rho_k_AB) const;
    
    arma::cx_mat fromABtoPM(const unitCellValues& ucv, const arma::cx_mat& rho_k_AB) const;
    
    arma::cx_mat fromPMtoAB(const unitCellValues& ucv, const arma::cx_mat& rho_k_PM) const;
    
    // Diag functions
    
    arma::cx_mat reshapeDiagFlatToAB(const arma::cx_vec& rho_flat) const;
    
    arma::cx_vec reshapeDiagABToFlat(const arma::cx_mat& rho_AB) const;
    
    arma::cx_mat transformDiagAB(const arma::cx_mat& rho_i_AB) const;
    
    arma::cx_mat itransformDiagAB(const arma::cx_mat& rho_k_diag_AB) const;
    
    arma::cx_mat fromDiagABtoPM(const unitCellValues& ucv, const arma::cx_mat& rho_k_diag_AB) const;
    
    arma::cx_mat fromDiagPMtoAB(const unitCellValues& ucv, const arma::cx_mat& rho_k_diag_PM) const;

  private:  
    const Lattice& _lattice;
    const HubbardHeisenbergParameters& _params;
    const ftwa_su_n::FourierTransformer2dPBC& _ft;
    
    arma::uvec _cell_diag_indices;
};

}

#endif // FOURIER_HPP
