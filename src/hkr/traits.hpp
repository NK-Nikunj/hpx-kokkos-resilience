#pragma once

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace hpx { namespace kokkos { namespace traits {

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