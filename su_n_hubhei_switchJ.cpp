#include <complex>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <armadillo>

#include <stdlib.h>
#include <stdint.h>

#include <boost/program_options.hpp>

#include "base/basic_defs.hpp"
#include "base/lattice.hpp"
#include "base/fourier.hpp"
#include "base/checkpoint_manager.hpp"
#include "su_n/wigner.hpp"
#include "su_n/eom_hubhei.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "print this help message")
        ("verbosity,v", po::value<unsigned int>()->default_value(10), "verbosity")
        ("name", po::value<std::string>(), "run name")
        ("ifile1", po::value<std::string>(), "file with the inital bond values")
        ("ifile2", po::value<std::string>(), "file with the time-evol eq. bond values")
        ("init,I", po::bool_switch()->default_value(false), "initialize database")
        ("lattice_dim,d", po::value<unsigned int>()->default_value(2), "spatial dimension of the lattice")
        ("lattice_length,L", po::value<unsigned int>()->default_value(11), "(cubic) side length of the lattice (in unit cells)")
        ("doping,d", po::value<double>()->default_value(0.0), "doping of the system")
        ("n_flavor,n", po::value<unsigned int>()->default_value(0), "degeneracy parameter")
        ("t_ini,h", po::value<double>()->default_value(1.0), "hopping parameter")
        ("J_ini,J", po::value<double>()->default_value(15.0), "Heisenberg interaction parameter")
        ("U_ini,U", po::value<double>()->default_value(0.0), "Hubbard interaction strength")
        ("t_fin", po::value<double>()->default_value(1.0), "hopping parameter for time-evol")
        ("J_fin", po::value<double>()->default_value(20.0), "Heisenberg interaction parameter for time-evol")
        ("U_fin", po::value<double>()->default_value(0.0), "Hubbard interaction strength for time-evol")
        ("switch_time_start", po::value<double>()->default_value(0.0), "time when the switching procedure starts")
        ("switch_time_end", po::value<double>()->default_value(10.0), "time when the switching procedure ends")
        ("switch_order,m", po::value<int>()->default_value(-1), "order of the switching function (-1 for quench)")
        ("switch_symm_break_strength", po::value<double>()->default_value(1e-3), "strength of transient Peierls symmetry breaking during switching")
        ("symm_break_strength", po::value<double>()->default_value(0.0), "strength of the symmetry breaking field")
        ("T_start,t", po::value<double>()->default_value(0.0), "starting point time-evolution")
        ("T_end,T", po::value<double>()->default_value(20.0), "end point time-evolution")
        ("dt,D", po::value<double>()->default_value(0.01), "time step")
        ("rep,r", po::value<unsigned int>()->default_value(1), "number of repetitions")
        ("write_interval,w", po::value<unsigned int>()->default_value(1), "after how many iterations program should write to disk")
    ;
    po::positional_options_description p;
    p.add("name", 1).add("ifile1", 1).add("ifile2", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm); // .allow_unregistered()
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return EXIT_SUCCESS;
    }
    else if (!vm.count("name")) {
        std::cout << "Need to provide a run name! Exiting." << std::endl;
        return EXIT_FAILURE;
    }
    else if (!vm.count("ifile1") || !vm.count("ifile2")) {
        std::cout << "Need to provide file paths! Exiting." << std::endl;
        return EXIT_FAILURE;
    }
    
    std::string runName(vm["name"].as<std::string>());
    
    std::cout.setf(std::ios::scientific);
    std::cout << std::setprecision(6);
    arma::arma_rng::set_seed_random();
    
    unsigned int verbosity = vm["verbosity"].as<unsigned int>();
    
    // what if d != 2?
    TiltedSquareLattice lattice(vm["lattice_length"].as<unsigned int>());
    
    HubbardHeisenbergParameters params_ini;
    params_ini.t = vm["t_ini"].as<double>();
    params_ini.J = vm["J_ini"].as<double>();
    params_ini.U = vm["U_ini"].as<double>();
    HubbardHeisenbergParameters params_fin;
    params_fin.t = vm["t_fin"].as<double>();
    params_fin.J = vm["J_fin"].as<double>();
    params_fin.U = vm["U_fin"].as<double>();
    
    SimulationParameters simParams;
    simParams.start_time    = vm["T_start"].as<double>();
    simParams.end_time      = vm["T_end"].as<double>();
    simParams.checkpoint_dt = vm["dt"].as<double>();
    simParams.num_tsteps    = std::round((simParams.end_time - simParams.start_time) / simParams.checkpoint_dt);
    
    HDF5CheckpointManager cm(lattice, runName);
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    if (vm["init"].as<bool>()) {
        cm.initDensities(
            1,
            simParams,
            false, // mom_space
            false, // with_cov_densdens
            true,  // with_cov_densdens_diag
            true   // write_to_file
        );
        
        // no offdiag
        
        cm.initBonds(
            simParams,
            true, // with_abs
            true, // with_abs2
            true  // write_to_file
        );
        
        cm.initFluxes(
            simParams,
            true  // write_to_file
        );
        
        cm.initModes(
            simParams,
            true // write_to_file
        );
        
        if (verbosity >= 1) std::cout << "# Initialized!" << std::endl;
        
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        if (verbosity >= 1) std::cout << "# TIMER DB init + write: " << time_span.count() << "s" << std::endl;
#endif
    } else {
        cm.initDensities(
            1,
            simParams,
            false, // mom_space
            false, // with_cov_densdens
            true,  // with_cov_densdens_diag
            false  // write_to_file
        );
        
        cm.initBonds(
            simParams,
            true, // with_abs
            true, // with_abs2
            false // write_to_file
        );
        
        cm.initFluxes(
            simParams,
            false // write_to_file
        );
        
        cm.initModes(
            simParams,
            false // write_to_file
        );
        
#ifdef FTWA_WITH_TIMER
        auto snap1_i = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1_i - snap0);
        if (verbosity >= 1) std::cout << "# TIMER DB init: " << time_span.count() << "s" << std::endl;
#endif  
        cm.loadDensities(
            false, // mom_space
            false, // with_cov_densdens
            true   // with_cov_densdens_diag
        );
        
        // no offdiag
        
        cm.loadBonds(
            true, // with_abs
            true  // with_abs2
        );
        
        cm.loadFluxes();
        
        cm.loadModes();
        
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap1 - snap1_i);
        if (verbosity >= 1) std::cout << "# TIMER DB load: " << time_span.count() << "s" << std::endl;
#endif
    }
    
    ftwa_su_n::FourierTransformer2dPBC ft_cells(lattice);
    ftwa_su_n::FTSqrt2CellLattice ft(lattice, params_ini, ft_cells);
    
    struct ftwa_su_n::unitCellValues ucv_ini, ucv_fin;
    ucv_ini.rhoA = -0.5*vm["doping"].as<double>();
    ucv_ini.rhoB = -0.5*vm["doping"].as<double>();
    
    ucv_fin.rhoA = -0.5*vm["doping"].as<double>();
    ucv_fin.rhoB = -0.5*vm["doping"].as<double>();
    
    readUCVFromFile(vm["ifile1"].as<std::string>(), ucv_ini);
    readUCVFromFile(vm["ifile2"].as<std::string>(), ucv_fin);
    
    ftwa_su_n::ODEHubHei2dSwitchJPBC hub2dODE(
        lattice,
        params_ini,
        simParams,
        ucv_ini,
        cm,
        ft,
        vm["symm_break_strength"].as<double>(),
        false,
        verbosity,
        params_fin,
        ucv_fin,
        vm["switch_time_start"].as<double>(),
        vm["switch_time_end"].as<double>(),
        vm["switch_order"].as<int>(),
        vm["switch_symm_break_strength"].as<double>()
    );
    auto system_func = std::bind(&ftwa_su_n::ODEHubHei2dSwitchJPBC::system, std::ref(hub2dODE), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    auto observer_func = std::bind(&ftwa_su_n::ODEHubHei2dSwitchJPBC::observer, std::ref(hub2dODE), std::placeholders::_1, std::placeholders::_2);
    
    unsigned int n_flavor = vm["n_flavor"].as<unsigned int>();
    std::mt19937 generator(std::random_device{}());
    
    std::unique_ptr<ftwa_su_n::HubHeiGaussWignerInfty> p_gw;
    (n_flavor == 0) ?
        p_gw.reset(new ftwa_su_n::HubHeiGaussWignerInfty(
            lattice,
            params_ini,
            ucv_ini,
            ft
        )) :
        p_gw.reset(new ftwa_su_n::HubHeiGaussWignerFiniteN(
            lattice,
            params_ini,
            ucv_ini,
            ft,
            n_flavor,
            generator
        ));
    
    unsigned int rep = (n_flavor == 0) ? 1 : vm["rep"].as<unsigned int>();
    unsigned int write_interval = (n_flavor == 0) ? 1 : vm["write_interval"].as<unsigned int>();
    
    double eps_abs = 1.0e-10, eps_rel = 1.0e-8;
    
    auto stepper = boost::numeric::odeint::make_controlled<boost::numeric::odeint::runge_kutta_cash_karp54<arma::cx_vec>>(eps_abs, eps_rel);
    
    arma::cx_vec icoords;
    
    for (unsigned int m = 0; m < rep; ++m) {
        if (verbosity >= 1) std::cout << "# rep = " << m << std::endl;
        
        if (m % write_interval == (write_interval - 1)) {
            cm.write_flag = true;
        }
        
#ifdef FTWA_WITH_TIMER
        auto snap2 = sc.now();
#endif
        p_gw->generate(icoords);
        
#ifdef FTWA_WITH_TIMER
        auto snap3 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap3 - snap2);
        if (verbosity >= 1) std::cout << "# TIMER initial state generation: " << time_span.count() << "s" << std::endl;
#endif
        
        boost::numeric::odeint::integrate_n_steps(stepper, std::ref(system_func), icoords, simParams.start_time, simParams.checkpoint_dt, simParams.num_tsteps, std::ref(observer_func));
        
        if (m % write_interval == (write_interval - 1)) {
            cm.write_flag = false;
        }
        
#ifdef FTWA_WITH_TIMER 
        auto snap4 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap4 - snap3);
        if (verbosity >= 1) std::cout << "# TIMER solving the e.o.m.: " << time_span.count() << "s" << std::endl;
#endif
    }
    
    return EXIT_SUCCESS;
}
