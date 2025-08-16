#ifndef LATTICE_HPP
#define LATTICE_HPP

#include "basic_defs.hpp"

/**
 * Class describing size and geometry of a lattice.
 * It is used to compute the number of sites, unit cells, and the length of the rho vector.
 */
class Lattice {
  public:
    Lattice(unsigned int dim, unsigned int L) : _dim(dim), _L(nullptr) {
        _L = new unsigned int[_dim];
        for (unsigned int d = 0; d < _dim; ++d) {
            _L[d] = L;
        }
    }
    
    inline unsigned int getLength(unsigned int dimIndex) const {
        return _L[dimIndex];
    }
    
    virtual unsigned int numCells() const {
        unsigned int n = 1;
        for (unsigned int d = 0; d < _dim; ++d) {
            n *= _L[d];
        }
        return n;
    }
    
    virtual unsigned int numSites() const {
        return numCells();
    }
    
    inline unsigned int rhoLenCells() const {
        return numCells()*(numCells()+1)/2;
    }
    
    inline unsigned int rhoLen() const {
        return numSites()*(numSites()+1)/2;
    }
    
    // virtual const std::vector<unsigned int>& getNeighbors(unsigned int i) const = 0;
    
    virtual ~Lattice() {
        delete[] _L;
    }
  protected:
    unsigned int _dim;
    unsigned int* _L;
};

class ChainLattice : public Lattice {
  public:
    ChainLattice(unsigned int L) : Lattice(1, L) { }
};

class SquareLattice : public Lattice {
  public:
    SquareLattice(unsigned int L) : Lattice(2, L) {
        unsigned int Lx = this->getLength(0), Ly = this->getLength(1);
        for (unsigned int iy = 0; iy < Ly; ++iy) {
            for (unsigned int ix = 0; ix < Lx; ++ix) {
                std::vector<unsigned int> nns = {
                    ((iy+Ly-1)%Ly)*Lx + ix,
                    iy*Lx + ((ix+1)%Lx),
                    ((iy+1)%Ly)*Lx + ix,
                    iy*Lx + ((ix+Lx-1)%Lx)
                };
                _nearestNeighbors.push_back(nns);
            }
        }
    }
    
    const std::vector<unsigned int>& getNeighbors(unsigned int i) const {
        return _nearestNeighbors[i];
    }
    
  private:
    std::vector< std::vector<unsigned int> > _nearestNeighbors;
};

class TiltedSquareLattice : public Lattice {
  public:
    // L is the number of unit cells!
    TiltedSquareLattice(unsigned int L) : Lattice(2, L) { }
    
    unsigned int numSites() const {
        return 2 * _L[0] * _L[1];
    }
};

inline void hubhei_sqrt2cell_neighbors(unsigned int L, unsigned int i, unsigned int* neighbors) {
    unsigned int iy = i / (2*L);
    unsigned int ix = i % (2*L);
    
    if (ix < L) {
        // EVEN rows
        
        // above left
        neighbors[0] = 2*L*((iy+L-1)%L) + (ix+L-1)%L+L;
        
        // above right
        neighbors[1] = 2*L*((iy+L-1)%L) + (ix+L);
        
        // below right
        neighbors[2] = 2*L*iy + (ix+L);
        
        // below left
        neighbors[3] = 2*L*iy + (ix+L-1)%L+L;
                
    } else {
        // ODD rows
        
        // above left
        neighbors[0] = 2*L*iy + (ix-L);
        
        // above right
        neighbors[1] = 2*L*iy + (ix-L+1)%L;
        
        // below right
        neighbors[2] = 2*L*((iy+1)%L) + (ix-L+1)%L;
        
        // below left
        neighbors[3] = 2*L*((iy+1)%L) + (ix-L);
        
    }
}

#endif // LATTICE_HPP
