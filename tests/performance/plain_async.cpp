#include <hpx/kokkos.hpp>

#include <Kokkos_Random.hpp>

#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hkr/executor/returning-executor.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>
#include <hkr/hpx-kokkos-resiliency.hpp>

#include <boost/program_options.hpp>

#include <random>

// Global variables
constexpr int num_iterations = 1000;

struct universal_ans
{
    HPX_HOST_DEVICE int operator()(std::uint64_t delay_s) const
    {
        if (delay_s == 0)
            return 42;

        Kokkos::Timer timer;

        while (true)
        {
            // Check if we've reached the specified delay
            if ((timer.seconds() >= delay_s))
            {
                break;
            }
        }

        return 42;
    }
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        hpx::kokkos::returning_executor exec_;
        hpx::kokkos::returning_host_executor host_exec_;

        namespace bpo = boost::program_options;

        bpo::options_description desc("Options");

        desc.add_options()("exec-time",
            bpo::value<std::uint64_t>()->default_value(100),
            "Time in us taken by a thread to execute before it terminates.");

        bpo::variables_map vm;

        // Setup commandline arguments
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);

        // Start application work
        std::uint64_t delay = vm["exec-time"].as<std::uint64_t>();

        {
            std::cout << "Starting plain async" << std::endl;

            std::vector<hpx::shared_future<int>> vect;
            vect.reserve(num_iterations);

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                hpx::shared_future<int> f =
                    hpx::async(exec_, universal_ans{}, delay / 1e6);

                vect.push_back(std::move(f));
            }

            for (int i = 0; i < num_iterations; ++i)
            {
                vect[i].get();
            }

            double elapsed = t.elapsed();
            hpx::util::format_to(
                std::cout, "Plain Async execution time = {1}\n", elapsed);
        }

        {
            std::cout << "Starting plain async" << std::endl;

            std::vector<hpx::shared_future<int>> vect;
            vect.reserve(num_iterations);

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                hpx::shared_future<int> f =
                    hpx::async(host_exec_, universal_ans{}, delay / 1e6);

                vect.push_back(std::move(f));
            }

            for (int i = 0; i < num_iterations; ++i)
            {
                vect[i].get();
            }

            double elapsed = t.elapsed();
            hpx::util::format_to(
                std::cout, "Plain Async execution time = {1}\n", elapsed);
        }
    }

    Kokkos::finalize();

    return 0;
}