#include <hpx/kokkos.hpp>

#include <hkr/executor/returning-executor.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>
#include <hkr/hpx-kokkos-resiliency.hpp>

#include <random>

struct test_func
{
    HPX_HOST_DEVICE int operator()(int random_arg) const
    {
        return 42;
    }
};

struct validate
{
    HPX_HOST_DEVICE constexpr bool operator()(int unused_arg) const
    {
        return true;
    }
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        hpx::kokkos::returning_executor exec_;

        int random_arg = std::rand();

        // Using API directly
        hpx::shared_future<int> f1 =
            hpx::kokkos::resiliency::async_replay_validate(
                exec_, 3, validate{}, test_func{}, random_arg);
        std::cout << "Returned value from direct API:" << f1.get() << std::endl;

        // Using async with replay executors
        auto exec =
            hpx::kokkos::resiliency::make_replay_executor(exec_, 3, validate{});
        hpx::shared_future<int> f2 = hpx::async(exec, test_func{}, random_arg);
        std::cout << "Returned value from replay executor:" << f2.get() << std::endl;

        std::cout << "Program ran correctly!" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}