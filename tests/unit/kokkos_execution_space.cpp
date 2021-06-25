#include <hkr/kokkos-execution-space.hpp>

#include <Kokkos_Core.hpp>

struct validator
{
    bool operator()(int, int) const
    {
        return true;
    }
};

struct operation
{
    KOKKOS_FUNCTION int operator()(int) const
    {
        return 42;
    }
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        validator validate{};
        operation op{};

        Kokkos::Experimental::HPX inst{};
        Kokkos::ResilientReplay<Kokkos::Experimental::HPX, validator> rinst(
            3, validate, inst);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<
                Kokkos::ResilientReplay<Kokkos::Experimental::HPX, validator>>(
                rinst, 0, 100),
            op);
        Kokkos::fence();

        std::cout << "Execution Complete" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}