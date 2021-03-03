#pragma once

#include <hpx/kokkos.hpp>

#include <hkr/hpx-kokkos-resiliency-cpos.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>

#include <hpx/future.hpp>

#include <memory>

namespace hpx { namespace kokkos { namespace resiliency {
    namespace detail {

        template <typename Result, typename Pred, typename F, typename Tuple>
        struct async_replay_executor_helper
          : std::enable_shared_from_this<
                async_replay_executor_helper<Result, Pred, F, Tuple>>
        {
            template <typename Pred_, typename F_, typename Tuple_>
            async_replay_executor_helper(Pred_&& pred, F_&& f, Tuple_&& tuple)
              : pred_(std::forward<Pred_>(pred))
              , f_(std::forward<F_>(f))
              , tuple_(std::forward<Tuple_>(tuple))
            {
            }

            template <typename Executor>
            decltype(auto) call(Executor&& exec, std::size_t n)
            {
                auto f = this->f_;
                auto pred = this->pred_;
                auto tuple = this->tuple_;

                return hpx::async(
                    exec, KOKKOS_LAMBDA() {
                        for (std::size_t i = 0u; i < n; ++i)
                        {
                            Result res =
                                hpx::util::invoke_fused_r<Result>(f, tuple);

                            bool result = pred(res);

                            if (result)
                            {
                                return res;
                            }
                        }
                    });
            }

            Pred pred_;
            F f_;
            Tuple tuple_;
        };

        template <typename Result, typename Pred, typename F, typename... Ts>
        std::shared_ptr<async_replay_executor_helper<Result,
            typename std::decay<Pred>::type, typename std::decay<F>::type,
            std::tuple<typename std::decay<Ts>::type...>>>
        make_async_replay_executor_helper(Pred&& pred, F&& f, Ts&&... ts)
        {
            using result_t = async_replay_executor_helper<Result,
                typename std::decay<Pred>::type, typename std::decay<F>::type,
                std::tuple<typename std::decay<Ts>::type...>>;

            return std::make_shared<result_t>(std::forward<Pred>(pred),
                std::forward<F>(f), std::make_tuple(std::forward<Ts>(ts)...));
        }

    }    // namespace detail

    template <typename Executor, typename Pred, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_two_way_executor<Executor>::value)>
    hpx::shared_future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    tag_invoke(async_replay_validate_t, Executor&& exec, std::size_t n,
        Pred&& pred, F&& f, Ts&&... ts)
    {
        using result_t =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        auto helper = detail::make_async_replay_executor_helper<result_t>(
            std::forward<Pred>(pred), std::forward<F>(f),
            std::forward<Ts>(ts)...);

        return helper->call(std::forward<Executor>(exec), n);
    }
}}}    // namespace hpx::kokkos::resiliency