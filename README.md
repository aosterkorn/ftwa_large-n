# Large-flavor fermionic truncated Wigner approximation (ftwa_large-n)

This respository provides an implementation of the fermionic truncated Wigner approximation method (`https://doi.org/10.1016/j.aop.2017.07.003`) for interacting fermionic quantum many-body systems in a large-flavor formulation.
It was used in the papers `https://doi.org/10.1103/PhysRevB.106.214318` and `https://doi.org/10.1088/1751-8121/ad6f7a`.

The purpose is to calculate the non-equilibrium dynamics of the SU(N)-Hubbard-Heisenberg Hamiltonian
```math
\hat H = -t_\mathrm{h} \sum_{\alpha = 1}^N \sum_{\langle i,j \rangle} \Big( c_{i\alpha}^\dagger c_{j\alpha} + \mathrm{H.c.} \Big) - \frac{J}{N} \sum_{\langle i,j\rangle} \Big| \sum_{\alpha = 1}^N c_{i\alpha}^\dagger c_{j\alpha} \Big|^2 + \frac{U}{N} \sum_i \Big( \sum_{\alpha=1}^N c_{i\alpha}^\dagger c_{i\alpha} - \frac{N}{2} \Big)^2
```
within the fermionic truncated Wigner approximation

Different protocols are implemented:
* Hubbard model interaction quench ($J = 0$): `su_n_fermi_quench.cpp`
* J-quench in the Hubbard-Heisenberg model: `su_n_hubhei_switchJ.cpp`
* electric field pulse in the Hubbard-Heisenberg model: `su_n_hubhei_peierls.cpp`

A legacy repository containing this code is available at
`https://gitlab.gwdg.de/stefan-kehrein-condensed-matter-theory/alexander-osterkorn/ftwa_code/`

## Compiling

I use the following libraries:
* `armadillo` (header only), needs (open)blas and lapack
* `hdf5`
* `boost_program_options`

I have not (yet) set up more sophisticated build tools, so you might need to adjust your `$CPLUS_INCLUDE_PATH` and `$(LD_)LIBRARY_PATH`.
Afterwards, just run
```
make
```

Useful armadillo compile flags:
* `-DARMA_DONT_USE_WRAPPER`: header only
* `-DARMA_ALLOW_FAKE_GCC`: e.g. for compiling with Intel MKL
* `-DARMA_NO_DEBUG`

Additional compile flags for the code:
* `-DFTWA_WITH_TIMER`: show time measurements of operations
* `-DFTWA_CACHE_CHECKPOINTS`


## Running
A description of all changeable parameters is obtained upon
```
./program_name --help
```

