#pragma once

#include <hpx/kokkos.hpp>
#include <Kokkos_Core.hpp>

#include <hkr/traits.hpp>
#include <hkr/util.hpp>

#include <type_traits>

namespace hpx {
    namespace kokkos {
        namespace experimental {
            namespace resiliency {

    template <typename ExecutionSpace, typename Validate>
    class replay_executor
    {
    public:
        using execution_space = ExecutionSpace;
        using execution_category = hpx::execution::parallel_execution_tag;

        template <typename F>
        explicit replay_executor(
            execution_space const& instance, std::size_t n, F&& f)
          : inst_(instance)
          , replay_count_(n)
          , validator_(std::forward<F>(f))
        {
        }

        execution_space instance() const
        {
            return inst_;
        }

        template <typename F, typename... Ts>
        decltype(auto) device_execution(F&& f, Ts&&... ts)
        {
            using return_t =
                typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type;

            return hpx::async(
                [&inst = inst_, n = replay_count_, pred = validator_,
                    func = std::forward<F>(f),
                    ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...)]() {
                    // Initialize result to be returned
                    Kokkos::View<return_t*, execution_space> exec_result(
                        "device_execution_space_result", 1);
                    Kokkos::View<return_t*, Kokkos::DefaultHostExecutionSpace>
                        host_result("device_host_result", 1);

                    Kokkos::View<bool*, execution_space> exec_bool(
                        "device_execution_space_bool", 1);
                    Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace>
                        host_bool("device_host_bool", 1);

                    Kokkos::parallel_for(
                        "async_replay",
                        Kokkos::RangePolicy<execution_space>(inst, 0, 1),
                        KOKKOS_LAMBDA(int) {
                            for (std::size_t i = 0; i < n; ++i)
                            {
                                return_t res =
                                    hpx::util::invoke_fused_r<return_t>(
                                        func, ts_pack);

                                bool valid = pred(res);

                                if (valid)
                                {
                                    exec_result[0] = res;
                                    exec_bool[0] = true;

                                    break;
                                }
                            }
                        });
                    // Let parallel_for run to completion
                    inst.fence();

                    Kokkos::deep_copy(host_result, exec_result);
                    Kokkos::deep_copy(host_bool, exec_bool);

                    if (host_bool[0])
                        return std::move(host_result[0]);

                    throw hpx::kokkos::resiliency::detail::resiliency_exception(
                        "Replay Execption Occured.");
                });
        }

        template <typename F, typename... Ts>
        decltype(auto) host_execution(F&& f, Ts&&... ts)
        {
            using return_t =
                typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type;

            return hpx::async(
                [&inst = inst_, n = replay_count_, pred = validator_,
                    func = std::forward<F>(f),
                    ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...)]() {
                    // Initialize result to be returned
                    Kokkos::View<return_t*, execution_space> exec_result(
                        "host_execution_space_result", 1);

                    Kokkos::View<bool*, execution_space> exec_bool(
                        "host_execution_space_bool", 1);

                    Kokkos::Experimental::HPX hpx_inst{
                        Kokkos::Experimental::HPX::instance_mode::independent};

                    Kokkos::parallel_for(
                        "async_replay",
                        Kokkos::RangePolicy<execution_space>(hpx_inst, 0, 1),
                        KOKKOS_LAMBDA(int) {
                            for (std::size_t i = 0; i < n; ++i)
                            {
                                return_t res =
                                    hpx::util::invoke_fused_r<return_t>(
                                        func, ts_pack);

                                bool valid = pred(res);

                                if (valid)
                                {
                                    exec_result[0] = std::move(res);
                                    exec_bool[0] = true;

                                    break;
                                }
                            }
                        });
                    // Let parallel_for run to completion
                    hpx_inst.fence();

                    if (exec_bool[0])
                        return std::move(exec_result[0]);

                    throw hpx::kokkos::resiliency::detail::resiliency_exception(
                        "Replay Execption Occured.");
                });
        }

        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            if constexpr (hpx::kokkos::traits::is_device_execution_space<
                              execution_space>::value)
                return device_execution(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            else
                return host_execution(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        execution_space inst_;
        std::size_t replay_count_;
        Validate validator_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExecutionSpace, typename Validate>
    replay_executor<ExecutionSpace, typename std::decay<Validate>::type>
    make_replay_executor(
        ExecutionSpace const& inst, std::size_t n, Validate&& validate)
    {
        return replay_executor<ExecutionSpace,
            typename std::decay<Validate>::type>(
            inst, n, std::forward<Validate>(validate));
    }

    template <typename ExecutionSpace, typename Validate>
    class replicate_executor
    {
    public:
        using execution_space = ExecutionSpace;
        using execution_category = hpx::execution::parallel_execution_tag;

        template <typename F>
        explicit replicate_executor(
            execution_space const& instance, std::size_t n, F&& f)
          : inst_(instance)
          , replicate_count_(n)
          , validator_(std::forward<F>(f))
        {
        }

        execution_space instance() const
        {
            return inst_;
        }

        template <typename F, typename... Ts>
        decltype(auto) device_execution(F&& f, Ts&&... ts)
        {
            using return_t =
                typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type;

            return hpx::async([&inst = inst_, n = replicate_count_,
                                  pred = validator_, func = std::forward<F>(f),
                                  ts_pack = hpx::make_tuple(
                                      std::forward<Ts>(ts)...)]() {
                // Initialize result to be returned
                Kokkos::View<return_t*, execution_space> exec_result(
                    "device_execution_space_result", 1);
                Kokkos::View<return_t*, Kokkos::DefaultHostExecutionSpace>
                    host_result("device_host_result", 1);

                Kokkos::View<bool*, execution_space> exec_bool(
                    "device_execution_space_bool", 1);
                Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace>
                    host_bool("device_host_bool", 1);

                Kokkos::parallel_for(
                    "async_replay",
                    Kokkos::RangePolicy<execution_space>(inst, 0, n),
                    KOKKOS_LAMBDA(int) {
                        return_t res =
                            hpx::util::invoke_fused_r<return_t>(func, ts_pack);

                        bool valid = pred(res);

                        if (valid)
                        {
                            if (!Kokkos::atomic_exchange(&exec_bool[0], true))
                                exec_result[0] = std::move(res);
                        }
                    });
                // Let parallel_for run to completion
                inst.fence();

                Kokkos::deep_copy(host_result, exec_result);
                Kokkos::deep_copy(host_bool, exec_bool);

                if (host_bool[0])
                    return std::move(host_result[0]);

                throw hpx::kokkos::resiliency::detail::resiliency_exception(
                    "Replicate Execption Occured.");
            });
        }

        template <typename F, typename... Ts>
        decltype(auto) host_execution(F&& f, Ts&&... ts)
        {
            using return_t =
                typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type;

            return hpx::async([&inst = inst_, n = replicate_count_,
                                  pred = validator_, func = std::forward<F>(f),
                                  ts_pack = hpx::make_tuple(
                                      std::forward<Ts>(ts)...)]() {
                // Initialize result to be returned
                Kokkos::View<return_t*, execution_space> exec_result(
                    "host_execution_space_result", 1);

                Kokkos::View<bool*, execution_space> exec_bool(
                    "host_execution_space_bool", 1);

                Kokkos::Experimental::HPX hpx_inst{
                    Kokkos::Experimental::HPX::instance_mode::independent};

                Kokkos::parallel_for(
                    "async_replay",
                    Kokkos::RangePolicy<execution_space>(hpx_inst, 0, n),
                    KOKKOS_LAMBDA(int) {
                        return_t res =
                            hpx::util::invoke_fused_r<return_t>(func, ts_pack);

                        bool valid = pred(res);

                        if (valid)
                        {
                            if (!Kokkos::atomic_exchange(&exec_bool[0], true))
                                exec_result[0] = std::move(res);
                        }
                    });
                // Let parallel_for run to completion
                hpx_inst.fence();

                if (exec_bool[0])
                    return std::move(exec_result[0]);

                throw hpx::kokkos::resiliency::detail::resiliency_exception(
                    "Replicate Execption Occured.");
            });
        }

        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            if constexpr (hpx::kokkos::traits::is_device_execution_space<
                              execution_space>::value)
                return device_execution(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            else
                return host_execution(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        execution_space inst_;
        std::size_t replicate_count_;
        Validate validator_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExecutionSpace, typename Validate>
    replicate_executor<ExecutionSpace, typename std::decay<Validate>::type>
    make_replicate_executor(
        ExecutionSpace const& inst, std::size_t n, Validate&& validate)
    {
        return replicate_executor<ExecutionSpace,
            typename std::decay<Validate>::type>(
            inst, n, std::forward<Validate>(validate));
    }

}}}}    // namespace hpx::kokkos::experimental::resiliency

namespace hpx { namespace parallel { namespace execution {

    template <typename ExecutionSpace, typename Validator>
    struct is_two_way_executor<hpx::kokkos::experimental::resiliency::
            replay_executor<ExecutionSpace, Validator>> : std::true_type
    {
    };

    template <typename ExecutionSpace, typename Validator>
    struct is_two_way_executor<hpx::kokkos::experimental::resiliency::
            replicate_executor<ExecutionSpace, Validator>> : std::true_type
    {
    };

}}}    // namespace hpx::parallel::execution