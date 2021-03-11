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
                        return hpx::make_tuple(true, std::move(res));
                }

                return hpx::make_tuple(false, result_t{});
            })
            .then([](hpx::future<hpx::tuple<bool, result_t>>&& f) {
                // Get pair
                auto&& result = f.get();

                if (!hpx::get<0>(result))
                    throw detail::resiliency_exception(
                        "Replay Exception occured.");

                return hpx::get<1>(std::move(result));
            });
    }

    template <typename Executor, typename Pred, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_two_way_executor<Executor>::value)>
    hpx::shared_future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    tag_invoke(async_replicate_validate_t, Executor&& exec, std::size_t n,
        Pred&& pred, F&& f, Ts&&... ts)
    {
        // Generate necessary components
        using result_t =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;
        auto tuple = hpx::make_tuple(std::forward<Ts>(ts)...);

        Kokkos::View<result_t*, Kokkos::DefaultHostExecutionSpace> host_result(
            "host_result", 1);
        Kokkos::View<result_t*,
            typename std::decay<Executor>::type::execution_space>
            exec_result("execution_space_result", 1);

        Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace> host_bool(
            "host_bool", 1);
        Kokkos::View<bool*,
            typename std::decay<Executor>::type::execution_space>
            exec_bool("execution_space_bool", 1);

        hpx::for_loop(
            hpx::kokkos::kok.on(exec).label("replicate_validate"), 0u, n,
            KOKKOS_LAMBDA(std::size_t i) {
                result_t res = hpx::util::invoke_fused_r<result_t>(f, tuple);

                bool result = pred(res);

                // Store only the first valid result generated
                if (result && !exec_bool[0])
                {
                    exec_result[0] = res;
                    exec_bool[0] = true;
                }
            });

        Kokkos::deep_copy(host_result, exec_result);
        Kokkos::deep_copy(host_bool, exec_bool);

        return hpx::async(hpx::launch::sync, [&]() {
            if (host_bool[0])
                return host_result[0];

            throw detail::resiliency_exception("Replicate Exception occured.");
            return host_result[0];
        });
    }

}}}    // namespace hpx::kokkos::resiliency