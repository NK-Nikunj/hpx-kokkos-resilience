#pragma once

#include <hpx/config.hpp>

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