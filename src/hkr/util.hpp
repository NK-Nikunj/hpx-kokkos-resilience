#pragma once

#include <algorithm>
#include <exception>
#include <string>

namespace hpx { namespace kokkos { namespace resiliency { namespace detail {

    struct resiliency_exception : public std::exception
    {
        std::string ex;

        resiliency_exception()
          : ex("Resiliency based exception occured.")
        {
        }

        resiliency_exception(std::string&& s)
          : ex(std::move(s))
        {
        }

        const char* what() const throw()
        {
            return ex.data();
        }
    };

}}}}    // namespace hpx::kokkos::resiliency::detail

namespace Kokkos { namespace Impl { namespace traits {

    template <typename ExecutionSpace, typename... Traits>
    struct RangePolicyBase
    {
        using execution_space = ExecutionSpace;
        using base_execution_space =
            typename execution_space::base_execution_space;
        using RangePolicy =
            Kokkos::RangePolicy<base_execution_space, Traits...>;
        using validator = typename execution_space::validator_type;
    };

}}}    // namespace Kokkos::Impl::traits