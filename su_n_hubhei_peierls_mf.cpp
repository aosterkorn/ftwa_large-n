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
        ("ifile", po::value<std::string>(), "file with the inital bond values")
        ("init,I", po::bool_switch()->default_value(false), "initialize database")
        ("lattice_dim,d", po::value<unsigned int>()->default_value(2), "spatial dimension of the lattice")
        ("lattice_length,L", po::value<unsigned int>()->default_value(11), "(cubic) side length of the lattice (in unit cells)")
        ("doping,d", po::value<double>()->default_value(0.0), "doping of the system")
        ("n_flavor,n", po::value<unsigned int>()->default_value(0), "degeneracy parameter")
        ("t_hop,t", po::value<double>()->default_value(1.0), "hopping parameter")
        ("J_hei,J", po::value<double>()->default_value(15.0), "Heisenberg interaction parameter")
        ("U_hub,U", po::value<double>()->default_value(0.0), "Hubbard interaction strength")
        ("p_ampl_x", po::value<double>()->default_value(0.0), "vector potential amplitude in x direction")
        ("p_ampl_y", po::value<double>()->default_value(1.0), "vector potential amplitude in y direction")
        ("p_freq", po::value<double>()->default_value(2.0), "frequency of the Peierls pulse")
        ("p_phase", po::value<double>()->default_value(0.0), "phase shift of the Peierls pulse")
        ("p_centr", po::value<double>()->default_value(15.0), "center of the Peierls pulse")
        ("p_width", po::value<double>()->default_value(4.0), "width of the Peierls pulse")
        ("p_is_spat_inhom", po::bool_switch()->default_value(false), "Peierls pulse is spatially inhomogeneous")
        ("p_spat_centr_x", po::value<double>()->default_value(0.0), "spatial center x coord of the Peierls pulse")
        ("p_spat_centr_y", po::value<double>()->default_value(0.0), "spatial center y coord of the Peierls pulse")
        ("p_spat_width", po::value<double>()->default_value(4.0), "spatial width of the Peierls pulse")
        ("T_start", po::value<double>()->default_value(0.0), "starting point time-evolution")
        ("T_end", po::value<double>()->default_value(30.0), "end point time-evolution")
        ("dt,D", po::value<double>()->default_value(0.1), "time step")
        ("rep,r", po::value<unsigned int>()->default_value(1), "number of repetitions")
        ("write_interval,w", po::value<unsigned int>()->default_value(1), "after how many iterations program should write to disk")
        ("eps_abs", po::value<double>()->default_value(1e-14), "absolute error tolerance")
        ("eps_rel", po::value<double>()->default_value(1e-12), "relative error tolerance")
    ;
    po::positional_options_description p;
    p.add("name", 1).add("ifile", 1);

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
    else if (!vm.count("ifile")) {
        std::cout << "Need to provide inital values! Exiting." << std::endl;
        return EXIT_FAILURE;
    }
    
    std::string runName(vm["name"].as<std::string>());
    
    std::cout.setf(std::ios::scientific);
    std::cout << std::setprecision(6);
    arma::arma_rng::set_seed_random();
    
    unsigned int verbosity = vm["verbosity"].as<unsigned int>();
    
    // what if d != 2?
    TiltedSquareLattice lattice(vm["lattice_length"].as<unsigned int>());
    
    HubbardHeisenbergParameters params;
    params.t = vm["t_hop"].as<double>();
    params.J = vm["J_hei"].as<double>();
    params.U = vm["U_hub"].as<double>();
    
    SimulationParameters simParams;
    simParams.start_time = vm["T_start"].as<double>();
    simParams.end_time = vm["T_end"].as<double>();
    simParams.checkpoint_dt = vm["dt"].as<double>();
    simParams.num_tsteps = std::round((simParams.end_time - simParams.start_time) / simParams.checkpoint_dt);
    
    HDF5CheckpointManager cm(lattice, runName);
    cm.write_flag = true;
    
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
            true // write_to_file
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
    } else{
        cm.initDensities(
            1,
            simParams,
            false, // mom_space
            false, // with_cov_densdens
            true,  // with_cov_densdens_diag
            false  // write_to_file
        );
        
        // no offdiag
        
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
    
    //~ double p_ampl_x = vm["p_ampl_x"].as<double>(),
        //~ p_ampl_y = vm["p_ampl_y"].as<double>(), 
        //~ p_freq = vm["p_freq"].as<double>(),
        //~ p_phase = vm["p_phase"].as<double>(),
        //~ p_centr = vm["p_centr"].as<double>(),
        //~ p_width = vm["p_width"].as<double>(),
        //~ p_spat_centr_x = vm["p_spat_centr_x"].as<double>(),
        //~ p_spat_centr_y = vm["p_spat_centr_y"].as<double>(),
        //~ p_spat_width = vm["p_spat_width"].as<double>();
    
    PeierlsPulseParameters pulseParams = {
        vm["p_ampl_x"].as<double>(),
        vm["p_ampl_y"].as<double>(), 
        vm["p_freq"].as<double>(),
        vm["p_phase"].as<double>(),
        vm["p_centr"].as<double>(),
        vm["p_width"].as<double>(),
        !vm["p_is_spat_inhom"].as<bool>(),
        vm["p_spat_centr_x"].as<double>(),
        vm["p_spat_centr_y"].as<double>(),
        vm["p_spat_width"].as<double>()
    };
    
    ftwa_su_n::FourierTransformer2dPBC ft_cells(lattice);
    
    ftwa_su_n::FTSqrt2CellLattice ft(lattice, params, ft_cells);
    
    struct ftwa_su_n::unitCellValues ucv;
    ucv.rhoA = -0.5*vm["doping"].as<double>();
    ucv.rhoB = -0.5*vm["doping"].as<double>();
        
    ftwa_su_n::ODEHubHei2dMomPeierlsPBC hub2dODE(
        lattice,
        params,
        simParams,
        ucv,
        cm,
        ft,
        false,
        verbosity,
        pulseParams
    );
    
    ftwa_su_n::readUCVFromFile(vm["ifile"].as<std::string>(), ucv);
    unsigned int n_flavor = vm["n_flavor"].as<unsigned int>();
    std::mt19937 generator(std::random_device{}());
    
    ftwa_su_n::HubHeiGaussWignerInfty gw(lattice, params, ucv, ft);
    
    unsigned int rep = (n_flavor == 0) ? 1 : vm["rep"].as<unsigned int>();
    unsigned int write_interval = (n_flavor == 0) ? 1 : vm["write_interval"].as<unsigned int>();
    
    double eps_abs = vm["eps_abs"].as<double>(), eps_rel = vm["eps_rel"].as<double>();
    
    // boost::numeric::odeint::runge_kutta4<arma::cx_vec> stepper;
    // boost::numeric::odeint::bulirsch_stoer<arma::cx_vec> stepper(eps_abs, eps_rel);
    // auto stepper = boost::numeric::odeint::make_controlled<boost::numeric::odeint::runge_kutta_cash_karp54<arma::cx_vec>>(eps_abs, eps_rel);
    
    auto stepper = boost::numeric::odeint::make_controlled<boost::numeric::odeint::runge_kutta_fehlberg78<arma::cx_vec>>(eps_abs, eps_rel, boost::numeric::odeint::runge_kutta_fehlberg78<arma::cx_vec>());
    
    auto system_func = std::bind(&ftwa_su_n::ODEHubHei2dMomPeierlsPBC::system, std::ref(hub2dODE), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    auto observer_func = std::bind(&ftwa_su_n::ODEHubHei2dMomPeierlsPBC::observer, std::ref(hub2dODE), std::placeholders::_1, std::placeholders::_2);
    
    arma::cx_vec icoords;
    
    for (unsigned int m = 0; m < rep; ++m) {
        if (verbosity >= 1) std::cout << "# rep = " << m << std::endl;
    
        if (m % write_interval == (write_interval - 1)) {
            cm.write_flag = true;
        }
        
#ifdef FTWA_WITH_TIMER
        auto snap2 = sc.now();
#endif

        gw.generateMom(icoords);
        icoords.clean(1e-12);
        
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
