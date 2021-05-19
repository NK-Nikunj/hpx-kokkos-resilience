#include <hpx/kokkos.hpp>

#include <Kokkos_Random.hpp>

#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hkr/executor/returning-executor.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>
#include <hkr/hpx-kokkos-resiliency.hpp>

#include <boost/program_options.hpp>

#include <random>

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

#if defined(KOKKOS_ENABLE_CUDA)
struct universal_ans_device
{
    HPX_HOST_DEVICE int operator()(std::uint64_t delay_ns,
        Kokkos::View<bool*, Kokkos::DefaultExecutionSpace> error_device,
        int num_iterations) const
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
            {
                // Re-run the thread if the thread was meant to re-run
                if (error_device[cur % num_iterations])
                    return 41;

                // No error has to occur with this thread, simply break the loop
                // after execution is done for the desired time
                else
                    break;
            }
        }

        return 42;
    }
};
#endif

struct universal_ans_host
{
    HPX_HOST_DEVICE int operator()(std::uint64_t delay_us,
        Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace> error_host,
        int num_iterations) const
    {
        if (delay_us == 0)
            return 42;

        Kokkos::Timer timer;

        while (true)
        {
            // Check if we've reached the specified delay
            if ((timer.seconds() * 1e6 >= delay_us))
            {
                // Re-run the thread if the thread was meant to re-run
                if (error_host[static_cast<int>(timer.seconds() * 1e6) %
                        num_iterations])
                    throw vogon_exception();

                // No error has to occur with this thread, simply break the loop
                // after execution is done for the desired time
                else
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dist(1, 100);

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
            bpo::value<std::uint64_t>()->default_value(1000),
            "Time in us taken by a thread to execute before it terminates.");
        desc.add_options()("iterations",
            bpo::value<std::uint64_t>()->default_value(10000),
            "Time in us taken by a thread to execute before it terminates.");

        bpo::variables_map vm;

        // Setup commandline arguments
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);

        // Start application work
        std::uint64_t n = vm["n-value"].as<std::uint64_t>();
        std::uint64_t error = vm["error-rate"].as<std::uint64_t>();
        std::uint64_t delay = vm["exec-time"].as<std::uint64_t>();
        std::uint64_t num_iterations = vm["iterations"].as<std::uint64_t>();

        Kokkos::View<bool*, Kokkos::DefaultExecutionSpace> error_device(
            "Error_device", num_iterations);
        Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace> error_host(
            "Error_host", num_iterations);

        hpx::for_loop(0, num_iterations,
            [&](int i) { error_host[i] = (dist(gen) < error ? true : false); });

        Kokkos::deep_copy(error_host, error_device);

#if defined(KOKKOS_ENABLE_CUDA)
        {
            std::cout << "Starting async replicate" << std::endl;

            std::vector<hpx::shared_future<int>> vect;
            vect.reserve(num_iterations);

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                bool err = (dist(gen) < error ? true : false);

                hpx::shared_future<int> f =
                    hpx::kokkos::resiliency::async_replicate_validate(exec_, n,
                        validate{}, universal_ans_device{}, delay * 1e3,
                        error_device, num_iterations);

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
#endif

        {
            std::cout << "Starting async replicate" << std::endl;

            std::vector<hpx::shared_future<int>> vect;
            vect.reserve(num_iterations);

            hpx::chrono::high_resolution_timer t;

            for (int i = 0; i < num_iterations; ++i)
            {
                bool err = (dist(gen) < error ? true : false);

                hpx::shared_future<int> f =
                    hpx::kokkos::resiliency::async_replicate_validate(
                        host_exec_, n, validate{}, universal_ans_host{}, delay,
                        error_host, num_iterations);

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