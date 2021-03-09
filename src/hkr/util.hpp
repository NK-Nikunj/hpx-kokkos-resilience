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

    // Utility wrapper
    template <typename T1, typename T2>
    struct hd_pair
    {
        T1 first;
        T2 second;

        HPX_HOST_DEVICE hd_pair()
          : first(T1{})
          , second(T2{})
        {
        }

        HPX_HOST_DEVICE hd_pair(T1 f, T2 s)
          : first(f)
          , second(s)
        {
        }

        HPX_HOST_DEVICE hd_pair(hd_pair const& pair_)
          : first(pair_.first)
          , second(pair_.second)
        {
        }

        HPX_HOST_DEVICE hd_pair(hd_pair&& pair_)
          : first(std::move(pair_.first))
          , second(std::move(pair_.second))
        {
        }

        HPX_HOST_DEVICE hd_pair& operator=(hd_pair const& pair_)
        {
            first = pair_.first;
            second = pair_.second;

            return *this;
        }

        HPX_HOST_DEVICE hd_pair& operator=(hd_pair&& pair_)
        {
            first = std::move(pair_.first);
            second = std::move(pair_.second);

            return *this;
        }
    };

}}}}    // namespace hpx::kokkos::resiliency::detail