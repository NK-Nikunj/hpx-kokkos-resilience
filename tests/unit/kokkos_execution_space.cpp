#include <hkr/kokkos-execution-space.hpp>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        Kokkos::DefaultHostExecutionSpace inst();
        Kokkos::ResilientReplay rinst(
            3, KOKKOS_LAMBDA(int, int) { return true; }, inst);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::ResilientReplay>(rinst, 0, 100),
            KOKKOS_LAMBDA(int i) { return i; });
    }

    Kokkos::finalize();

    return 0;
}