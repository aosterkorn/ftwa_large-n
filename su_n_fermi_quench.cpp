#include <complex>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>

#include <armadillo>

#include <stdlib.h>
#include <stdint.h>

#include <boost/program_options.hpp>

#include "base/basic_defs.hpp"
#include "base/lattice.hpp"
#include "base/fourier.hpp"
#include "base/checkpoint_manager.hpp"
#include "su_n/wigner.hpp"
#include "su_n/eom_hub.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "print this help message")
        ("verbosity,v", po::value<unsigned int>()->default_value(10), "verbosity")
        ("name", po::value<std::string>(), "run name")
        ("init,I", po::bool_switch()->default_value(false), "initialize database")
        ("lattice_dim,d", po::value<unsigned int>()->default_value(2), "spatial dimension of the lattice")
        ("lattice_length,L", po::value<unsigned int>()->default_value(10), "(cubic) side length of the lattice")
        ("particles,p", po::value<unsigned int>()->default_value(25), "specify number of particles (0 = half filling)")
        ("temperature,K", po::value<double>(), "initial temperature")
        ("n_flavor,n", po::value<unsigned int>()->default_value(2), "specify number of flavors")
        ("t_hop,t", po::value<double>()->default_value(1.0), "hopping parameter")
        ("U_int,U", po::value<double>()->default_value(0.5), "interaction strength")
        ("T_start", po::value<double>()->default_value(0.0), "starting point time-evolution")
        ("T_end", po::value<double>()->default_value(20.0), "end point time-evolution")
        ("dt,D", po::value<double>()->default_value(0.05), "time step")
        ("wigner_func_model,m", po::value<unsigned int>()->default_value(1), "Wigner function model: Gaussian (1), Two point (2)")
        ("rep,r", po::value<unsigned int>()->default_value(1), "number of repetitions")
        ("write_interval,w", po::value<unsigned int>()->default_value(1), "after how many iterations program should write to disk")
        ("eps_abs", po::value<double>()->default_value(1e-10), "absolute error tolerance")
        ("eps_rel", po::value<double>()->default_value(1e-8), "relative error tolerance")
    ;
    po::positional_options_description p;
    p.add("name", 1);

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
    
    std::string runName(vm["name"].as<std::string>());
    unsigned int rep = vm["rep"].as<unsigned int>();
    unsigned int write_interval = vm["write_interval"].as<unsigned int>();
    
    std::cout.setf(std::ios::scientific);
    std::cout << std::setprecision(6);
    arma::arma_rng::set_seed_random();
    
    unsigned int verbosity = vm["verbosity"].as<unsigned int>();
    
    SquareLattice lattice(vm["lattice_length"].as<unsigned int>());
    if (vm["lattice_dim"].as<unsigned int>() != 2) {
        std::cerr << "different lattice dimension chosen!" << std::endl;
    }
    
    HubbardParameters params;
    params.t = vm["t_hop"].as<double>();
    params.U = vm["U_int"].as<double>();
    
    ftwa_su_n::FourierTransformer2dPBC ft(lattice);
    
    SimulationParameters simParams;
    simParams.start_time = vm["T_start"].as<double>();
    simParams.end_time = vm["T_end"].as<double>();
    simParams.checkpoint_dt = vm["dt"].as<double>();
    simParams.num_tsteps = std::round((simParams.end_time - simParams.start_time) / simParams.checkpoint_dt);
    
    std::unique_ptr<ftwa_su_n::WignerFuncProduct> p_gw;
    vm.count("temperature") ?
        p_gw.reset(new ftwa_su_n::WignerFuncFermiSeaTemp(
            lattice,
            vm["n_flavor"].as<unsigned int>(),
            static_cast<ftwa_su_n::WignerFuncModel>(vm["wigner_func_model"].as<unsigned int>()),
            vm["particles"].as<unsigned int>(),
            vm["temperature"].as<double>()
        )) :
        p_gw.reset(new ftwa_su_n::WignerFuncFermiSeaGS(
            lattice,
            vm["n_flavor"].as<unsigned int>(),
            static_cast<ftwa_su_n::WignerFuncModel>(vm["wigner_func_model"].as<unsigned int>()),
            vm["particles"].as<unsigned int>()
        ));

    // SQLiteCheckpointManager cm(lattice, runName);
    HDF5CheckpointManager cm(lattice, runName);
    ftwa_su_n::ODEHub2dPBC hub2dODE(lattice, params, simParams, cm, ft, verbosity);
    
#ifdef FTWA_WITH_TIMER
    std::chrono::steady_clock sc;
    auto snap0 = sc.now();
#endif
    
if (vm["init"].as<bool>()) {
        // Init empty database in position space
        // and write it to the disk.
        cm.initDensities(1, simParams,
                false, // mom_space
                false, // with_cov_densdens
                true,  // with_cov_densdens_diag
                true   // write_to_file
        );
        // cm.initOffdiag(1, simParams,
        //         false, // mom_space
        //         false, // with_cov
        //         false, // with_abs2
        //         true   // write_to_file
        // );

        // Init empty database in momentum space
        // and write it to the disk.
        cm.initDensities(1, simParams,
                true,  // mom_space
                false, // with_cov_densdens
                true,  // with_cov_densdens_diag
                true   // write_to_file
        );
        // cm.initOffdiag(1, simParams,
        //         true,  // mom_space
        //         false, // with_cov
        //         false, // with_abs2
        //         true   // write_to_file
        // );
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        std::cout << "# TIMER DB init: " << time_span.count() << "s" << std::endl;
#endif
    } else {
        std::cout << "NO db initialization, assuming that " << vm["name"].as<std::string>() << ".h5 exists!" << std::endl;
        // Init empty database in position space and
        // load the data from existing db on the disk.
        cm.initDensities(1, simParams,
                false, // mom_space
                false, // with_cov
                true,  // with_abs2
                false  // write_to_file
        );
        // cm.initOffdiag(1, simParams,
        //         false, // mom_space
        //         false, // with_cov
        //         false, // with_abs2
        //         false  // write_to_file
        // );
        cm.loadDensities(
                false, // mom_space
                false, // with_cov_densdens
                true   // with_cov_densdens_diag
        );
        // cm.loadOffdiag(
        //         false, // mom_space
        //         false, // with_cov
        //         false  // with_abs2
        // );
        
        // Init empty database in momentum space and
        // load the data from existing db on the disk.
        cm.initDensities(1, simParams,
                true,  // mom_space
                false, // with_cov
                true,  // with_abs2
                false  // write_to_file
        );
        // cm.initOffdiag(1, simParams,
        //         true,  // mom_space
        //         false, // with_cov
        //         false, // with_abs2
        //         false  // write_to_file
        // );
        cm.loadDensities(
                true,  // mom_space
                false, // with_cov_densdens
                true   // with_cov_densdens_diag
        );
        // cm.loadOffdiag(
        //         true,  // mom_space
        //         false, // with_cov
        //         false  // with_abs2
        // );
#ifdef FTWA_WITH_TIMER
        auto snap1 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap1 - snap0);
        std::cout << "# TIMER DB load: " << time_span.count() << "s" << std::endl;
#endif
    }
    
    double eps_abs = vm["eps_abs"].as<double>(), eps_rel = vm["eps_rel"].as<double>();
    
    auto stepper = boost::numeric::odeint::make_controlled<boost::numeric::odeint::runge_kutta_cash_karp54<arma::cx_vec>>(eps_abs, eps_rel);
    auto system_func = std::bind(&ftwa_su_n::ODEHub2dPBC::system, std::ref(hub2dODE), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    auto observer_func = std::bind(&ftwa_su_n::ODEHub2dPBC::observer, std::ref(hub2dODE), std::placeholders::_1, std::placeholders::_2);
    
    std::mt19937 generator(std::random_device{}());
    arma::cx_vec icoords, coords;
    
    for (unsigned int m = 0; m < rep; ++m) {
        std::cout << "# m = " << m << std::endl;
        
        if (m % write_interval == (write_interval - 1)) {
            cm.write_flag = true;
        }
#ifdef FTWA_WITH_TIMER
        auto snap2 = sc.now();
#endif
        p_gw->generate(generator, icoords);
        
#ifdef FTWA_WITH_TIMER
        auto snap3 = sc.now();
        auto time_span = static_cast<std::chrono::duration<double>>(snap3 - snap2);
        std::cout << "# TIMER initial state generation: " << time_span.count() << "s" << std::endl;
#endif
        ft.itransform(icoords, coords);
        
#ifdef FTWA_WITH_TIMER
        auto snap4 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap4 - snap3);
        std::cout << "# TIMER FT state to real space: " << time_span.count() << "s" << std::endl;
#endif
        boost::numeric::odeint::integrate_n_steps(stepper, std::ref(system_func), coords, simParams.start_time, simParams.checkpoint_dt, simParams.num_tsteps, std::ref(observer_func));
        
        if (m % write_interval == (write_interval - 1)) {
            cm.write_flag = false;
        }
       
#ifdef FTWA_WITH_TIMER 
        auto snap5 = sc.now();
        time_span = static_cast<std::chrono::duration<double>>(snap5 - snap4);
        std::cout << "# TIMER solving the e.o.m.: " << time_span.count() << "s" << std::endl;
#endif

    }
    return EXIT_SUCCESS;
}
