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

                result_t res{};

                for (std::size_t i = 0u; i < n; ++i)
                {
                    res = hpx::util::invoke_fused_r<result_t>(f, tuple);

                    bool result = pred(res);

                    if (result)
                    {
                        return hpx::make_tuple(true, std::move(res));
                    }
                }

                return hpx::make_tuple(false, std::move(res));
            })
            .then([](hpx::future<hpx::tuple<bool, result_t>>&& f) {
                // Get pair
                auto result = f.get();

                if (!std::get<0>(result))
                    throw detail::replay_exception();

                return std::get<1>(result);
            });
    }

}}}    // namespace hpx::kokkos::resiliency