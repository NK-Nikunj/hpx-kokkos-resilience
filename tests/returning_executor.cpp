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

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        hpx::kokkos::default_returning_executor exec;
        hpx::kokkos::default_host_returning_executor host_exec;

        int random_arg = std::rand();

        hpx::future<int> f1 = hpx::async(exec, test_func{}, random_arg);
        std::cout << "Device returned with value: " << f1.get() << std::endl;
    
        hpx::future<int> f2 = hpx::async(host_exec, test_func{}, random_arg);
        std::cout << "Host returned with value: " << f2.get() << std::endl;

        std::cout << "Program ran correctly!" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}