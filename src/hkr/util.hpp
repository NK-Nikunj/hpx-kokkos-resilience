#pragma once

#include <hpx/config.hpp>

#include <algorithm>
#include <exception>
#include <string>

namespace hpx { namespace kokkos { namespace resiliency { namespace detail {

    struct replay_exception : public std::exception
    {
        std::string ex;

        replay_exception()
          : ex("Replay exception occured.")
        {
        }

        replay_exception(std::string&& s)
          : ex(std::move(s))
        {
        }

        const char* what() const throw()
        {
            return ex.data();
        }
    };

}}}}    // namespace hpx::kokkos::resiliency::detail