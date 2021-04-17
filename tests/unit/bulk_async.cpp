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
    std::cout << "here" << std::endl;
    return false;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        hpx::kokkos::returning_executor exec_;
        hpx::kokkos::returning_host_executor host_exec_;

        // Bulk aync facility
        auto exec =
            hpx::kokkos::resiliency::make_replay_executor(exec_, 3, validate);
        hpx::for_loop(hpx::execution::par.on(exec), 0, 100, test_func);

        auto host_exec = hpx::kokkos::resiliency::make_replay_executor(
            host_exec_, 3, false_validate);

        try
        {
            hpx::for_loop(hpx::execution::par.on(host_exec), 0, 100, test_func);
        }
        catch (...)
        {
            std::cout << "Error captured!" << std::endl;
        }

        std::cout << "Program ran correctly!" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}