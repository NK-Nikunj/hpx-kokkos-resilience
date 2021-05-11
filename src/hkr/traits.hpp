#pragma once

#include <Kokkos_Core.hpp>

namespace hpx { namespace kokkos { namespace traits {

    template <typename ExecutionSpace>
    struct to_memory_space
    {
        using type = Kokkos::HostSpace;
    };

    template <>
    struct to_memory_space<Kokkos::Cuda>
    {
        using type = Kokkos::CudaSpace;
    };

}}}    // namespace hpx::kokkos::traits