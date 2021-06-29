#include <hkr/replay-execution-space.hpp>
#include <hkr/replicate-execution-space.hpp>

#include <Kokkos_Core.hpp>

struct validator
{
    KOKKOS_FUNCTION bool operator()(int, int) const
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

        // Host only variant
        {
            Kokkos::Experimental::HPX inst{};
            // Replay Strategy
            Kokkos::ResilientReplay<Kokkos::Experimental::HPX, validator>
                replay_inst(3, validate, inst);
            Kokkos::ResilientReplicate<Kokkos::Experimental::HPX, validator>
                replicate_inst(3, validate, inst);

            Kokkos::parallel_for(
                Kokkos::RangePolicy<Kokkos::ResilientReplay<
                    Kokkos::Experimental::HPX, validator>>(replay_inst, 0, 100),
                op);
            Kokkos::fence();

            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::ResilientReplicate<
                                     Kokkos::Experimental::HPX, validator>>(
                                     replicate_inst, 0, 100),
                op);
            Kokkos::fence();
        }

        // Device only variant
        {
            Kokkos::Cuda inst{};
            // Replay Strategy
            Kokkos::ResilientReplay<Kokkos::Cuda, validator> replay_inst(
                3, validate, inst);
            Kokkos::ResilientReplicate<Kokkos::Cuda, validator> replicate_inst(
                3, validate, inst);

            Kokkos::parallel_for(
                Kokkos::RangePolicy<
                    Kokkos::ResilientReplay<Kokkos::Cuda, validator>>(
                    replay_inst, 0, 100),
                op);
            Kokkos::fence();

            Kokkos::parallel_for(
                Kokkos::RangePolicy<
                    Kokkos::ResilientReplicate<Kokkos::Cuda, validator>>(
                    replicate_inst, 0, 100),
                op);
            Kokkos::fence();
        }
        std::cout << "Execution Complete" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}