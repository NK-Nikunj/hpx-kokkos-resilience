#include <Kokkos_Random.hpp>

#include <hpx/chrono.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/algorithms.hpp>

#include <hkr/replay-execution-space.hpp>
#include <hkr/replicate-execution-space.hpp>

#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>

#include <random>

struct vogon_exception : std::exception
{
};

struct validate
{
    KOKKOS_FUNCTION bool operator()(std::uint64_t, int result) const
    {
        return result == 42;
    }
};

#if defined(KOKKOS_ENABLE_CUDA)
struct universal_ans_device
{
    universal_ans_device(std::uint64_t delay,
        Kokkos::View<bool*, Kokkos::DefaultExecutionSpace> error,
        std::uint64_t iterations)
      : delay_ns(delay)
      , error_device(error)
      , num_iterations(iterations)
    {
    }

    KOKKOS_FUNCTION int operator()(std::uint64_t) const
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

private:
    std::uint64_t delay_ns;
    Kokkos::View<bool*, Kokkos::DefaultExecutionSpace> error_device;
    std::uint64_t num_iterations;
};
#endif

struct universal_ans_host
{
    universal_ans_host(std::uint64_t delay,
        Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace> const& error,
        std::uint64_t iterations)
      : delay_us(delay)
      , error_host(error)
      , num_iterations(iterations)
    {
    }

    KOKKOS_FUNCTION int operator()(std::uint64_t) const
    {
        if (delay_us == 0)
            return 42;

        hpx::chrono::high_resolution_timer timer;

        while (true)
        {
            // Check if we've reached the specified delay
            if ((timer.elapsed() * 1e6 >= delay_us))
            {
                // Re-run the thread if the thread was meant to re-run
                if (error_host[static_cast<int>(timer.elapsed() * 1e6) %
                        num_iterations])
                    return 41;

                // No error has to occur with this thread, simply break the loop
                // after execution is done for the desired time
                else
                    break;
            }
        }

        return 42;
    }

private:
    std::uint64_t delay_us;
    Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace> const& error_host;
    std::uint64_t num_iterations;
};

int hpx_main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dist(1, 100);

        namespace bpo = boost::program_options;

        bpo::options_description desc("Options");

        desc.add_options()("n-value",
            bpo::value<std::uint64_t>()->default_value(3),
            "Number of repeat launches for async replay.");
        desc.add_options()("error-rate",
            bpo::value<std::uint64_t>()->default_value(0),
            "Average rate at which error is likely to occur.");
        desc.add_options()("grain-size",
            bpo::value<std::uint64_t>()->default_value(100),
            "Time in us taken by a single iteration to execute.");
        desc.add_options()("iterations",
            bpo::value<std::uint64_t>()->default_value(1000000),
            "Kokkos parallel_for ranges between 0-<iterations>.");

        bpo::variables_map vm;

        // Setup commandline arguments
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);

        // Start application work
        std::uint64_t n = vm["n-value"].as<std::uint64_t>();
        std::uint64_t error = vm["error-rate"].as<std::uint64_t>();
        std::uint64_t delay = vm["grain-size"].as<std::uint64_t>();
        std::uint64_t num_iterations = vm["iterations"].as<std::uint64_t>();

        Kokkos::View<bool*, Kokkos::DefaultExecutionSpace> error_device(
            "Error_device", num_iterations);
        Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace> error_host(
            "Error_host", num_iterations);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                Kokkos::DefaultHostExecutionSpace{}, 0, num_iterations),
            [&](int i) { error_host[i] = (dist(gen) < error ? true : false); });

        Kokkos::deep_copy(error_host, error_device);

        universal_ans_host host_functor(delay, error_host, num_iterations);

        if (!error)
        {
            std::cout << "Starting Kokkos parallel_for host" << std::endl;

            Kokkos::DefaultHostExecutionSpace inst{};

            hpx::chrono::high_resolution_timer timer;

            for (int i = 0; i != 10; ++i)
            {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                        inst, 0, num_iterations),
                    host_functor);
                Kokkos::fence();
            }

            double elapsed = timer.elapsed();

            std::cout << "Execution Time: " << elapsed / 10 << std::endl;
        }

        {
            std::cout << "Starting Kokkos Resient Replay host" << std::endl;

            Kokkos::DefaultHostExecutionSpace inst{};
            // Replay Strategy
            Kokkos::ResilientReplay<Kokkos::DefaultHostExecutionSpace, validate>
                replay_inst(n, validate{}, inst);

            hpx::chrono::high_resolution_timer timer;

            for (int i = 0; i != 10; ++i)
            {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<Kokkos::ResilientReplay<
                        Kokkos::DefaultHostExecutionSpace, validate>>(
                        replay_inst, 0, num_iterations),
                    host_functor);
                Kokkos::fence();
            }

            double elapsed = timer.elapsed();

            std::cout << "Execution Time: " << elapsed / 10 << std::endl;
        }

        {
            std::cout << "Starting Kokkos Resient Replicate host" << std::endl;

            Kokkos::DefaultHostExecutionSpace inst{};
            // Replay Strategy
            Kokkos::ResilientReplicate<Kokkos::DefaultHostExecutionSpace,
                validate>
                replay_inst(n, validate{}, inst);

            hpx::chrono::high_resolution_timer timer;

            for (int i = 0; i != 10; ++i)
            {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<Kokkos::ResilientReplicate<
                        Kokkos::DefaultHostExecutionSpace, validate>>(
                        replay_inst, 0, num_iterations),
                    host_functor);
                Kokkos::fence();
            }

            double elapsed = timer.elapsed();

            std::cout << "Execution Time: " << elapsed / 10 << std::endl;
        }

#if defined(KOKKOS_ENABLE_CUDA)

        universal_ans_device device_functor(
            delay * 1000, error_device, num_iterations);

        if (!error)
        {
            std::cout << "Starting Kokkos parallel_for device" << std::endl;

            Kokkos::Cuda inst{};

            hpx::chrono::high_resolution_timer timer;

            for (int i = 0; i != 10; ++i)
            {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<Kokkos::Cuda>(inst, 0, num_iterations),
                    device_functor);
                Kokkos::fence();
            }

            double elapsed = timer.elapsed();

            std::cout << "Execution Time: " << elapsed / 10 << std::endl;
        }

        {
            std::cout << "Starting Kokkos Resient Replay device" << std::endl;

            Kokkos::Cuda inst{};
            // Replay Strategy
            Kokkos::ResilientReplay<Kokkos::Cuda, validate> replay_inst(
                n, validate{}, inst);

            hpx::chrono::high_resolution_timer timer;

            for (int i = 0; i != 10; ++i)
            {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<
                        Kokkos::ResilientReplay<Kokkos::Cuda, validate>>(
                        replay_inst, 0, num_iterations),
                    device_functor);
                Kokkos::fence();
            }

            double elapsed = timer.elapsed();

            std::cout << "Execution Time: " << elapsed / 10 << std::endl;
        }

        {
            std::cout << "Starting Kokkos Resient Replicate device"
                      << std::endl;

            Kokkos::Cuda inst{};
            // Replay Strategy
            Kokkos::ResilientReplicate<Kokkos::Cuda, validate> replay_inst(
                n, validate{}, inst);

            hpx::chrono::high_resolution_timer timer;

            for (int i = 0; i != 10; ++i)
            {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<
                        Kokkos::ResilientReplicate<Kokkos::Cuda, validate>>(
                        replay_inst, 0, num_iterations),
                    device_functor);
                Kokkos::fence();
            }

            double elapsed = timer.elapsed();

            std::cout << "Execution Time: " << elapsed / 10 << std::endl;
        }
#endif
    }

    Kokkos::finalize();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
