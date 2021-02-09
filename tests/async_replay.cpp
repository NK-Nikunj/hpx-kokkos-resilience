#include <hpx/kokkos.hpp>

#include <hkr/hpx-kokkos-resiliency-executor.hpp>
#include <hkr/hpx-kokkos-resiliency.hpp>

#include <random>

void test_func(int random_arg)
{
    std::cout << "Function is running..." << std::endl;
}

bool validate(int unused_arg)
{
    std::cout << "Validating result..." << std::endl;
    return true;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::default_executor exec_;

        int random_arg = std::rand();

        // Using API directly
        hpx::shared_future<void> f1 =
            hpx::kokkos::resiliency::async_replay_validate(
                exec_, 3, validate, test_func, random_arg);
        f1.get();

        // Using async with replay executors
        auto exec =
            hpx::kokkos::resiliency::make_replay_executor(exec_, 3, validate);
        hpx::shared_future<void> f2 = hpx::async(exec, test_func, random_arg);
        f2.get();

        std::cout << "Program ran correctly!" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}