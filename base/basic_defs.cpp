// basic_defs.cpp

#include "basic_defs.hpp"

void halfFilling2D(const Lattice& lattice, arma::vec& kvals) {
    unsigned int L = lattice.getLength(0);
    unsigned int V = lattice.numSites();
    kvals.zeros(V);
    
    double n = 0.0;
    // at uneven L: not exactly half filling
    for (unsigned int mx = 0; mx < L; ++mx) {
        for (unsigned int my = 0; my < L; ++my) {
            if (dispTightBinding2d(L, mx, my) < 1e-9) {
                kvals(mx + L*my) = 1.0;
                n += 1.0;
            }
            //~ } else if (dispTightBinding2d(L, mx, my) < 1e-9) {
                //~ if (L % 2 == 1) {
                    //~ kvals(mx + L*my) = 1.0;
                    //~ n += 1.0;
                //~ } else {
                    //~ kvals(mx + L*my) = 0.5;
                    //~ n += 0.5;
                //~ }
            //~ }
        }
    }
    
    std::cout << "# filling factor " << n / (L*L) << std::endl;
}

void fermiSurface2D(const Lattice& lattice, unsigned int N, arma::vec& kvals) {
    if (N == 0) {
        halfFilling2D(lattice, kvals);
        return;
    }
    unsigned int L = lattice.getLength(0);
    unsigned int V = lattice.numSites();
    kvals.zeros(V);
    
    std::vector<unsigned int> ks = {};
    
    for (unsigned int m = 0; m < V; ++m) {
        ks.push_back(m);
    }
    std::sort(ks.begin(), ks.end(), std::bind(sortFermi2D, L, std::placeholders::_1, std::placeholders::_2));
    for (unsigned int i = 0; i < N; ++i) {
        kvals(ks[i]) = 1.0;
    }
    
    std::cout << "# filling factor: " << (double) N / (double) (L*L) << std::endl;
}

void plotFermiSurface2D(const Lattice& lattice, const arma::vec& kvals) {
    unsigned int L = lattice.getLength(0);
    unsigned int V = lattice.numSites();
    for (unsigned int m = 0; m < V; ++m) {
        if (m % L == 0 && m != 0) std::cout << std::endl;
        std::cout << " " << (unsigned int) kvals(m);
    }
    std::cout << std::endl;
}

inline double dispTightBinding1dLR(unsigned int L, unsigned int mx) {
    if (mx <= L/2 - 1) {
        return 2*M_PI/(double) L * (mx + 0.5);
    }
    else {
        return 2*M_PI/(double) L * ((int) mx - (int) L + 0.5);
    }
}

void halfFilling1dLR(const Lattice& lattice, arma::vec& kvals) {
    unsigned int L = lattice.getLength(0);
    kvals.zeros(L);
    
    double n = 0.0;
    // at uneven L: not exactly half filling
    for (unsigned int mx = 0; mx < L; ++mx) {
        if (dispTightBinding1dLR(L, mx) < 1e-9) {
            kvals(mx) = 1.0;
            n += 1.0;
        }
    }
    std::cout << "# filling factor " << n / L << std::endl;
}
