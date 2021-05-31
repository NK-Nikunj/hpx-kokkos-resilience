#include <hpx/kokkos.hpp>

#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hkr/kokkos-executor.hpp>

#include <boost/program_options.hpp>

#include <random>

struct validate
{
    HPX_HOST_DEVICE bool operator()(hpx::tuple<double, double> result) const
    {
        return true;
    }
};

HPX_HOST_DEVICE double stencil(double left, double center, double right)
{
    return 0.5 * (0.75) * left + (0.75) * center - 0.5 * (0.25) * right;
}

HPX_HOST_DEVICE double left_flux(double left, double center)
{
    return (0.625) * left - (0.125) * center;
}

HPX_HOST_DEVICE double right_flux(double center, double right)
{
    return 0.5 * (0.75) * center + (1.125) * right;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        namespace bpo = boost::program_options;
        bpo::options_description desc("Options");

        desc.add_options()("n-value",
            bpo::value<std::uint64_t>()->default_value(3),
            "Number of repeat launches for async replay.");
        desc.add_options()("error-rate",
            bpo::value<std::uint64_t>()->default_value(1),
            "Error rate for injecting errors.");
        desc.add_options()("subdomain-width",
            bpo::value<std::uint64_t>()->default_value(128),
            "Local x dimension (of each partition).");
        desc.add_options()("iterations",
            bpo::value<std::uint64_t>()->default_value(10),
            "Number of time steps.");
        desc.add_options()("steps-per-iteration",
            bpo::value<std::uint64_t>()->default_value(16),
            "Number of time steps per iterations.");
        desc.add_options()("subdomains",
            bpo::value<std::uint64_t>()->default_value(10),
            "Number of partitions.");

        bpo::variables_map vm;

        // Setup commandline arguments
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);

        // Start application work
        std::uint64_t n = vm["n-value"].as<std::uint64_t>();
        std::uint64_t error = vm["error-rate"].as<std::uint64_t>();
        std::uint64_t subdomain_width =
            vm["subdomain-width"].as<std::uint64_t>();
        std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
        std::uint64_t subdomains = vm["subdomains"].as<std::uint64_t>();
        std::uint64_t sti = vm["steps-per-iteration"].as<std::uint64_t>();

        {
            Kokkos::Cuda cuda_inst{};
            Kokkos::Experimental::HPX hpx_inst{};

            Kokkos::View<double** [2], Kokkos::CudaSpace> data_device(
                "data_device", subdomains, subdomain_width + 1);
            Kokkos::View<double** [2], Kokkos::Experimental::HPX> data_host(
                "data_host", subdomains, subdomain_width + 1);

            Kokkos::View<double*, Kokkos::CudaSpace> checksum_device(
                "checksum_device", subdomains);
            Kokkos::View<double*, Kokkos::Experimental::HPX> checksum_host(
                "checksum_host", subdomains);

            Kokkos::View<double*, Kokkos::CudaSpace> test_value_device(
                "test_device", subdomains);
            Kokkos::View<double*, Kokkos::Experimental::HPX> test_value_host(
                "test_host", subdomains);

            const double pi = std::acos(-1.0);

            // Initialize
            Kokkos::parallel_for(
                "initialize_data",
                Kokkos::RangePolicy<Kokkos::Experimental::HPX>(
                    hpx_inst, 0, subdomains),
                KOKKOS_LAMBDA(std::uint64_t i) {
                    checksum_host[i] = 0.0;
                    test_value_host[i] = 0.0;

                    for (std::uint64_t j = 0; j != subdomain_width + 1; ++j)
                    {
                        double value =
                            static_cast<double>(subdomain_width * i + j) /
                            static_cast<double>(subdomain_width * subdomains);
                        data_host(i, j, 0) = std::sin(2 * pi * value);

                        checksum_host[i] += data_host(i, j, 0);
                    }
                });
            Kokkos::fence();

            Kokkos::deep_copy(data_host, data_device);
            Kokkos::deep_copy(checksum_host, checksum_device);
            Kokkos::deep_copy(test_value_host, test_value_device);

            std::vector<hpx::future<hpx::tuple<double, double>>> tasks;
            auto exec =
                hpx::kokkos::experimental::resiliency::make_replay_executor(
                    cuda_inst, 3, validate{});
            for (std::uint64_t timestep = 0; timestep != iterations; ++timestep)
            {
                for (std::uint64_t subdomain_index = 0;
                     subdomain_index != subdomains; ++subdomain_index)
                {
                    std::uint64_t size = subdomains + 2 * sti + 1;
                    Kokkos::View<double*, Kokkos::CudaSpace> workspace(
                        "temp", size);
                    hpx::future<hpx::tuple<double, double>> fut = hpx::async(
                        exec, KOKKOS_LAMBDA() {
                            std::uint64_t curr = timestep % 2;
                            std::uint64_t next = (timestep + 1) % 2;

                            for (std::uint64_t i = sti, j = 0; i != 0; --i, ++j)
                                workspace[j] = data_device(
                                    (subdomain_index + subdomains - 1) %
                                        subdomains,
                                    subdomain_width - sti, curr);
                            for (std::uint64_t i = 0; i != subdomain_width; ++i)
                                workspace[sti + i] =
                                    data_device(subdomain_index, i, curr);
                            for (std::uint64_t i = 0; i != sti; ++i)
                                workspace[sti + subdomain_width + i] =
                                    data_device(
                                        (subdomain_index + 1) % subdomains, i,
                                        curr);

                            double checksum = 0.0;
                            for (std::uint64_t i = 0; i != size; ++i)
                                checksum += workspace[i];

                            for (std::size_t t = 0; t != sti; ++t)
                            {
                                checksum -=
                                    left_flux(workspace[0], workspace[1]);
                                checksum -= right_flux(workspace[size - 2 * t],
                                    workspace[size - 2 * t + 1]);
                                for (std::size_t k = 0; k != size - 2 * t; ++k)
                                    workspace[k] = stencil(workspace[k],
                                        workspace[k + 1], workspace[k + 2]);
                            }

                            double test_value = 0.0;
                            for (std::uint64_t i = 0; i < subdomain_width + 1;
                                 ++i)
                            {
                                data_device(subdomain_index, i, next) =
                                    workspace[i];
                                test_value += workspace[i];
                            }

                            test_value_device[subdomain_index] = test_value;
                            checksum_device[subdomain_index] = checksum;

                            return hpx::make_tuple(test_value, checksum);
                        });
                    tasks.emplace_back(std::move(fut));
                }
                hpx::wait_all(tasks);
            }
        }
    }

    Kokkos::finalize();

    return 0;
}