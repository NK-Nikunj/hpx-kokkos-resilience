#pragma once

#include <hpx/kokkos.hpp>

#include <hkr/hpx-kokkos-resiliency-cpos.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>
#include <hkr/util.hpp>

#include <hpx/future.hpp>

#include <exception>
#include <memory>
#include <tuple>

namespace hpx { namespace kokkos { namespace resiliency {

    template <typename Executor, typename Pred, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_two_way_executor<Executor>::value)>
    hpx::shared_future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    tag_invoke(async_replay_validate_t, Executor&& exec, std::size_t n,
        Pred&& pred, F&& f, Ts&&... ts)
    {
        // Generate necessary components
        using result_t =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;
        auto tuple = hpx::make_tuple(std::forward<Ts>(ts)...);

        return hpx::async(
            exec,
            KOKKOS_LAMBDA() {
                // Ensure the value of n is greater than 0
                HPX_ASSERT(n > 0);

                for (std::size_t i = 0u; i < n; ++i)
                {
                    result_t res =
                        hpx::util::invoke_fused_r<result_t>(f, tuple);

                    bool result = pred(res);

                    if (result)
                    {
                        return detail::hd_pair<bool, result_t>{true, result};
                    }

                    return detail::hd_pair<bool, result_t>{false, result};
                }
            })
            .then([](hpx::future<detail::hd_pair<bool, result_t>> && f) {
                // Get pair
                auto result = f.get();

                if (!result.first)
                    throw detail::replay_exception();

                return result.second;
            });
    }

}}}    // namespace hpx::kokkos::resiliency