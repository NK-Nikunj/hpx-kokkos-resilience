#pragma once

#include <hpx/kokkos.hpp>

#include <hkr/hpx-kokkos-resiliency-cpos.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>

#include <hpx/future.hpp>

#include <memory>

namespace hpx { namespace kokkos { namespace resiliency {
    namespace detail {

        template <typename Pred, typename F, typename Tuple>
        struct async_replay_executor_helper
          : std::enable_shared_from_this<
                async_replay_executor_helper<Pred, F, Tuple>>
        {
            template <typename Pred_, typename F_, typename Tuple_>
            async_replay_executor_helper(Pred_&& pred, F_&& f, Tuple_&& tuple)
              : pred_(std::forward<Pred_>(pred))
              , f_(std::forward<F_>(f))
              , tuple_(std::forward<Tuple_>(tuple))
            {
            }

            template <std::size_t... Is>
            void invoke(hpx::util::index_pack<Is...>)
            {
                f_(std::get<Is>(tuple_)...);
            }

            template <typename Executor>
            hpx::shared_future<void> call(Executor&& exec, std::size_t n)
            {
                return hpx::async(
                    exec, KOKKOS_LAMBDA() {
                        for (std::size_t i = 0u; i < n; ++i)
                        {
                            using pack_type = hpx::util::make_index_pack<
                                std::tuple_size<Tuple>::value>;

                            invoke(pack_type{});

                            bool result = pred_(1);

                            if (result)
                                return;
                        }
                    });
            }

            Pred pred_;
            F f_;
            Tuple tuple_;
        };

        template <typename Pred, typename F, typename... Ts>
        std::shared_ptr<async_replay_executor_helper<
            typename std::decay<Pred>::type, typename std::decay<F>::type,
            std::tuple<typename std::decay<Ts>::type...>>>
        make_async_replay_executor_helper(Pred&& pred, F&& f, Ts&&... ts)
        {
            return std::make_shared<async_replay_executor_helper<
                typename std::decay<Pred>::type, typename std::decay<F>::type,
                std::tuple<typename std::decay<Ts>::type...>>>(
                std::forward<Pred>(pred), std::forward<F>(f),
                std::make_tuple(std::forward<Ts>(ts)...));
        }

    }    // namespace detail

    template <typename Executor, typename Pred, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_two_way_executor<Executor>::value)>
    hpx::shared_future<void> tag_invoke(async_replay_validate_t,
        Executor&& exec, std::size_t n, Pred&& pred, F&& f, Ts&&... ts)
    {
        auto helper =
            detail::make_async_replay_executor_helper(std::forward<Pred>(pred),
                std::forward<F>(f), std::forward<Ts>(ts)...);

        return helper->call(std::forward<Executor>(exec), n);
    }
}}}    // namespace hpx::kokkos::resiliency