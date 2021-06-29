#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hkr/kokkos-executor.hpp>
#include <hkr/util.hpp>

#include <random>

struct test_function
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

struct false_validate
{
    HPX_HOST_DEVICE constexpr bool operator()(int unused_arg) const
    {
        return false;
    }
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        int random_arg = std::rand();

        Kokkos::Experimental::HPX host_inst{};
        Kokkos::DefaultExecutionSpace device_inst{};

        // Create replay executor
        auto exec = hpx::kokkos::experimental::resiliency::make_replay_executor(
            host_inst, 3, validate{});

        hpx::future<int> host_f = hpx::async(exec, test_function{}, random_arg);
        std::cout << "Returned value from replay executor:" << host_f.get()
                  << std::endl;

        hpx::future<int> device_f =
            hpx::async(exec, test_function{}, random_arg);
        std::cout << "Returned value from replay executor:" << device_f.get()
                  << std::endl;

        // Catching exceptions
        try
        {
            auto except_exec =
                hpx::kokkos::experimental::resiliency::make_replay_executor(
                    host_inst, 3, false_validate{});
            hpx::future<int> except_f =
                hpx::async(except_exec, test_function{}, random_arg);

            std::cout << "Trying to return value: " << except_f.get()
                      << std::endl;
        }
        catch (hpx::kokkos::resiliency::detail::resiliency_exception& e)
        {
            std::cout << e.what() << std::endl;
        }

        std::cout << "Program ran correctly!" << std::endl;
    }

    Kokkos::finalize();
}