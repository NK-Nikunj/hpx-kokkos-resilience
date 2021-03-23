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

struct vogon_exception : std::exception
{
};

struct validate
{
    HPX_HOST_DEVICE bool operator()(int result) const
    {
        return result == 42;
    }
};

struct universal_ans
{
    HPX_HOST_DEVICE int operator()(std::uint64_t delay_s, std::uint64_t error,
        Kokkos::Random_XorShift64_Pool<> rand_pool) const
    {
        typename Kokkos::Random_XorShift64_Pool<>::generator_type rand_gen =
            rand_pool.get_state();

        if (delay_s == 0)
            return 42;

        std::uint64_t num = rand_gen.urand64() % 100;
        bool error_flag = false;

        // Probability of error occurrence is proportional to error rate
        if (num < error)
        {
            error_flag = true;
        }

        Kokkos::Timer timer;

        while (true)
        {
            // Check if we've reached the specified delay
            if ((timer.seconds() >= delay_s))
            {
                // Re-run the thread if the thread was meant to re-run
                if (error_flag)
                    throw vogon_exception();

                // No error has to occur with this thread, simply break the loop
                // after execution is done for the desired time
                else
                    break;
            }
        }

        rand_pool.free_state(rand_gen);
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

        desc.add_options()("n-value",
            bpo::value<std::uint64_t>()->default_value(3),
            "Number of repeat launches for async replay.");
        desc.add_options()("error-rate",
            bpo::value<std::uint64_t>()->default_value(2),
            "Average rate at which error is likely to occur.");
        desc.add_options()("exec-time",
            bpo::value<std::uint64_t>()->default_value(100),
            "Time in us taken by a thread to execute before it terminates.");

        bpo::variables_map vm;

        // Setup commandline arguments
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);

        // Start application work
        std::uint64_t n = vm["n-value"].as<std::uint64_t>();
        std::uint64_t error = vm["error-rate"].as<std::uint64_t>();
        std::uint64_t delay = vm["exec-time"].as<std::uint64_t>();

        {
            std::cout << "Starting async replicate" << std::endl;

            std::vector<hpx::shared_future<int>> vect;
            vect.reserve(num_iterations);

            // Random number generator
            Kokkos::Random_XorShift64_Pool<> rand_pool(std::time(nullptr));

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                hpx::shared_future<int> f =
                    hpx::kokkos::resiliency::async_replicate_validate(exec_, n,
                        validate{}, universal_ans{}, delay / 1e6, error,
                        rand_pool);

                vect.push_back(std::move(f));
            }

            try
            {
                for (int i = 0; i < num_iterations; ++i)
                {
                    vect[i].get();
                }
            }
            catch (vogon_exception const&)
            {
                std::cout << "Number of repeat launches were not enough to get "
                             "past the injected error levels"
                          << std::endl;
            }

            double elapsed = t.elapsed();
            hpx::util::format_to(
                std::cout, "Async replicate execution time = {1}\n", elapsed);
        }

        {
            std::cout << "Starting async replicate" << std::endl;

            std::vector<hpx::shared_future<int>> vect;
            vect.reserve(num_iterations);

            // Random number generator
            Kokkos::Random_XorShift64_Pool<> rand_pool(std::time(nullptr));

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                hpx::shared_future<int> f =
                    hpx::kokkos::resiliency::async_replicate_validate(
                        host_exec_, n, validate{}, universal_ans{}, delay / 1e6,
                        error, rand_pool);

                vect.push_back(std::move(f));
            }

            try
            {
                for (int i = 0; i < num_iterations; ++i)
                {
                    vect[i].get();
                }
            }
            catch (vogon_exception const&)
            {
                std::cout << "Number of repeat launches were not enough to get "
                             "past the injected error levels"
                          << std::endl;
            }

            double elapsed = t.elapsed();
            hpx::util::format_to(
                std::cout, "Async replicate execution time = {1}\n", elapsed);
        }
    }

    Kokkos::finalize();

    return 0;
}