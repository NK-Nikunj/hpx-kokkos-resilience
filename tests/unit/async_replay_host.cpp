#include <hpx/kokkos.hpp>

#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hkr/executor/returning-executor.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>
#include <hkr/hpx-kokkos-resiliency.hpp>

#include <random>

int test_func(int random_arg)
{
    return 42;
}

bool validate(int unused_arg)
{
    return true;
}

bool false_validate(int unused_arg)
{
    return false;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        hpx::kokkos::returning_host_executor exec_;

        int random_arg = std::rand();

        // Using API directly
        hpx::shared_future<int> f1 =
            hpx::kokkos::resiliency::async_replay_validate(
                exec_, 3, validate, test_func, random_arg);
        std::cout << "Returned value from direct API:" << f1.get() << std::endl;

        // Using async with replay executors
        auto exec =
            hpx::kokkos::resiliency::make_replay_executor(exec_, 3, validate);
        hpx::shared_future<int> f2 = hpx::async(exec, test_func, random_arg);
        std::cout << "Returned value from replay executor:" << f2.get()
                  << std::endl;

        // Catching exceptions
        try
        {
            auto except_exec = hpx::kokkos::resiliency::make_replay_executor(
                exec_, 3, false_validate);
            hpx::shared_future<int> f3 =
                hpx::async(except_exec, test_func, random_arg);

            std::cout << "Trying to return value: " << f3.get() << std::endl;
        }
        catch (hpx::kokkos::resiliency::detail::resiliency_exception& e)
        {
            std::cout << e.what() << std::endl;
        }

        std::cout << "Program ran correctly!" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}