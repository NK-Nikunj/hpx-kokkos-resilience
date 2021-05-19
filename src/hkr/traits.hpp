#pragma once

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace hpx { namespace kokkos { namespace traits {

    template <typename ExecutionSpace>
    struct to_memory_space
    {
        using type = Kokkos::HostSpace;
    };

#if defined(KOKKOS_ENABLE_CUDA)
    template <>
    struct to_memory_space<Kokkos::Cuda>
    {
        using type = Kokkos::CudaSpace;
    };
#endif

    template <typename ExecutionSpace>
    struct is_device_execution_space : std::false_type
    {
    };

#if defined(KOKKOS_ENABLE_CUDA)
    template <>
    struct is_device_execution_space<Kokkos::Cuda> : std::true_type
    {
    };
#endif

#if defined(KOKKOS_ENABLE_HIP)
    template <>
    struct is_device_execution_space<Kokkos::Experimental::HIP> : std::true_type
    {
    };
#endif

}}}    // namespace hpx::kokkos::traits