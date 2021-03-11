#pragma once

#include <hpx/resiliency/resiliency.hpp>

#include <hpx/kokkos.hpp>

#include <hkr/hpx-kokkos-resiliency-cpos.hpp>

namespace hpx { namespace kokkos { namespace resiliency {

    template <typename BaseExecutor, typename Validate>
    class replay_executor
    {
    public:
        static constexpr int num_spread = 4;
        static constexpr int num_task = 128;

        using execution_category = typename BaseExecutor::execution_category;
        using execution_parameters_type =
            typename hpx::parallel::execution::extract_executor_parameters<
                BaseExecutor>::type;
        using execution_space = typename BaseExecutor::execution_space;

        template <typename Result>
        using future_type =
            typename hpx::parallel::execution::executor_future<BaseExecutor,
                Result>::type;

        template <typename F>
        explicit replay_executor(BaseExecutor& exec, std::size_t n, F&& f)
          : exec_(exec)
          , replay_count_(n)
          , validator_(std::forward<F>(f))
        {
        }

        bool operator==(replay_executor const& rhs) const noexcept
        {
            return exec_ = rhs.exec_;
        }

        bool operator!=(replay_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        replay_executor const& context() const noexcept
        {
            return *this;
        }

        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return async_replay_validate(exec_, replay_count_, validator_,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        BaseExecutor& exec_;
        std::size_t replay_count_;
        Validate validator_;
    };

    template <typename BaseExecutor, typename Validate>
    class replicate_executor
    {
    public:
        static constexpr int num_spread = 4;
        static constexpr int num_task = 128;

        using execution_category = typename BaseExecutor::execution_category;
        using execution_parameters_type =
            typename hpx::parallel::execution::extract_executor_parameters<
                BaseExecutor>::type;
        using execution_space = typename BaseExecutor::execution_space;

        template <typename Result>
        using future_type =
            typename hpx::parallel::execution::executor_future<BaseExecutor,
                Result>::type;

        template <typename F>
        explicit replicate_executor(BaseExecutor& exec, std::size_t n, F&& f)
          : exec_(exec)
          , replicate_count_(n)
          , validator_(std::forward<F>(f))
        {
        }

        bool operator==(replicate_executor const& rhs) const noexcept
        {
            return exec_ = rhs.exec_;
        }

        bool operator!=(replicate_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        replicate_executor const& context() const noexcept
        {
            return *this;
        }

        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return async_replicate_validate(exec_, replicate_count_, validator_,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        BaseExecutor& exec_;
        std::size_t replicate_count_;
        Validate validator_;
    };

    ////////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor, typename Validate>
    replay_executor<BaseExecutor, typename std::decay<Validate>::type>
    make_replay_executor(BaseExecutor& exec, std::size_t n, Validate&& validate)
    {
        return replay_executor<BaseExecutor,
            typename std::decay<Validate>::type>(
            exec, n, std::forward<Validate>(validate));
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor, typename Validate>
    replicate_executor<BaseExecutor, typename std::decay<Validate>::type>
    make_replicate_executor(
        BaseExecutor& exec, std::size_t n, Validate&& validate)
    {
        return replicate_executor<BaseExecutor,
            typename std::decay<Validate>::type>(
            exec, n, std::forward<Validate>(validate));
    }

}}}    // namespace hpx::kokkos::resiliency

namespace hpx { namespace parallel { namespace execution {

    template <typename BaseExecutor, typename Validator>
    struct is_two_way_executor<
        hpx::kokkos::resiliency::replay_executor<BaseExecutor, Validator>>
      : std::true_type
    {
    };

    template <typename BaseExecutor, typename Validator>
    struct is_two_way_executor<
        hpx::kokkos::resiliency::replicate_executor<BaseExecutor, Validator>>
      : std::true_type
    {
    };

}}}    // namespace hpx::parallel::execution
