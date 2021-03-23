//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2021 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains HPX executors that forward to a Kokkos backend.

#pragma once

#include <hpx/kokkos/deep_copy.hpp>
#include <hpx/kokkos/detail/logging.hpp>
#include <hpx/kokkos/kokkos_algorithms.hpp>
#include <hpx/kokkos/make_instance.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/numeric.hpp>
#include <hpx/tuple.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace hpx { namespace kokkos {

    /// \brief HPX executor wrapping a Kokkos execution space.
    template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
    class ret_executor
    {
    public:
        using execution_space = ExecutionSpace;
        using execution_category = hpx::execution::parallel_execution_tag;

        explicit ret_executor(
            execution_space_mode mode = execution_space_mode::global)
          : inst(mode == execution_space_mode::global ?
                    ExecutionSpace{} :
                    detail::make_independent_execution_space_instance<
                        ExecutionSpace>())
        {
        }
        explicit ret_executor(execution_space const& instance)
          : inst(instance)
        {
        }

        execution_space instance() const
        {
            return inst;
        }

        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            auto ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...);
            parallel_for_async(
                Kokkos::Experimental::require(
                    Kokkos::RangePolicy<execution_space>(inst, 0, 1),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(
                    int) { hpx::util::invoke_fused_r<void>(f, ts_pack); });
        }

        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            // Get return type of the invocable
            using return_t =
                typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type;

            auto ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...);

            Kokkos::View<return_t*, ExecutionSpace> result("result", 1);
            Kokkos::View<return_t*, Kokkos::DefaultHostExecutionSpace>
                result_host("host_result", 1);

            // Get a handle of future
            hpx::shared_future<void> fut = parallel_for_async(
                Kokkos::Experimental::require(
                    Kokkos::RangePolicy<execution_space>(inst, 0, 1),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(int) {
                    result[0] = hpx::util::invoke_fused_r<return_t>(f, ts_pack);
                });

            // Attach a continuation and return result
            return fut.then(
                hpx::launch::sync, [=](hpx::shared_future<void>&& f) {
                    // Throw any error reported by f
                    f.get();

                    // Deep copy it to the host version and return the result
                    Kokkos::deep_copy(result_host, result);
                    return result_host[0];
                });
        }

        template <typename F, typename S, typename... Ts>
        std::vector<hpx::shared_future<void>> bulk_async_execute(
            F&& f, S const& s, Ts&&... ts)
        {
            HPX_KOKKOS_DETAIL_LOG("bulk_async_execute");
            auto ts_pack = hpx::make_tuple(std::forward<Ts>(ts)...);
            auto size = hpx::util::size(s);
            auto b = hpx::util::begin(s);

            std::vector<hpx::shared_future<void>> result;
            result.push_back(parallel_for_async(
                Kokkos::Experimental::require(
                    Kokkos::RangePolicy<ExecutionSpace>(inst, 0, size),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(int i) {
                    HPX_KOKKOS_DETAIL_LOG("bulk_async_execute i = %d", i);
                    using index_pack_type =
                        typename hpx::util::detail::fused_index_pack<decltype(
                            ts_pack)>::type;
                    detail::invoke_helper(
                        index_pack_type{}, f, *(b + i), ts_pack);
                }));

            return result;
        }

        template <typename Parameters, typename F>
        constexpr std::size_t get_chunk_size(Parameters&& params, F&& f,
            std::size_t cores, std::size_t count) const
        {
            return std::size_t(-1);
        }

    private:
        execution_space inst{};
    };

    // Define type aliases
    using returning_executor = ret_executor<Kokkos::DefaultExecutionSpace>;
    using returning_host_executor =
        ret_executor<Kokkos::DefaultHostExecutionSpace>;

#if defined(KOKKOS_ENABLE_CUDA)
    using cuda_returning_executor = ret_executor<Kokkos::Cuda>;
#endif

#if defined(KOKKOS_ENABLE_HIP)
    using hip_returning_executor = ret_executor<Kokkos::Experimental::HIP>;
#endif

#if defined(KOKKOS_ENABLE_HPX)
    using hpx_returning_executor = ret_executor<Kokkos::Experimental::HPX>;
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
    using openmp_returning_executor = ret_executor<Kokkos::OpenMP>;
#endif

#if defined(KOKKOS_ENABLE_ROCM)
    using rocm_returning_executor = ret_executor<Kokkos::ROCm>;
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
    using serial_returning_executor = ret_executor<Kokkos::Serial>;
#endif

    template <typename returning_executor>
    struct is_kokkos_returning_executor : std::false_type
    {
    };

    template <typename ExecutionSpace>
    struct is_kokkos_executor<ret_executor<ExecutionSpace>> : std::true_type
    {
    };
}}    // namespace hpx::kokkos

namespace hpx { namespace parallel { namespace execution {
    template <typename ExecutionSpace>
    struct is_one_way_executor<hpx::kokkos::ret_executor<ExecutionSpace>>
      : std::true_type
    {
    };

    template <typename ExecutionSpace>
    struct is_two_way_executor<hpx::kokkos::ret_executor<ExecutionSpace>>
      : std::true_type
    {
    };

    template <typename ExecutionSpace>
    struct is_bulk_two_way_executor<hpx::kokkos::ret_executor<ExecutionSpace>>
      : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution
