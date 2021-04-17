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

struct universal_ans_device
{
    HPX_HOST_DEVICE int operator()(std::uint64_t delay_ns) const
    {
        if (delay_ns == 0)
            return 42;

        // Get current time from Nvidia GPU register
        std::uint64_t cur;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cur));

        while (true)
        {
            std::uint64_t now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));

            // Check if we've reached the specified delay
            if (now - cur >= delay_ns)
                break;
        }

        return 42;
    }
};

struct universal_ans_host
{
    HPX_HOST_DEVICE int operator()(std::uint64_t delay_us) const
    {
        if (delay_us == 0)
            return 42;

        Kokkos::Timer timer;

        while (true)
        {
            // Check if we've reached the specified delay
            if ((timer.seconds() * 1e6 >= delay_us))
                break;
        }

        return 42;
    }
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

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

            std::vector<hpx::future<int>> vect;
            vect.reserve(num_iterations);

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                hpx::future<int> f = hpx::async(
                    hpx::kokkos::returning_executor{
                        hpx::kokkos::execution_space_mode::independent},
                    universal_ans_device{}, delay * 1e3);

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

            std::vector<hpx::future<int>> vect;
            vect.reserve(num_iterations);

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                hpx::future<int> f = hpx::async(
                    hpx::kokkos::returning_host_executor{
                        hpx::kokkos::execution_space_mode::independent},
                    universal_ans_host{}, delay);

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