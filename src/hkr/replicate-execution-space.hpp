#pragma once

#include <Kokkos_Core.hpp>

#include <hkr/util.hpp>

#include <cstdint>
#include <exception>
#include <type_traits>

namespace Kokkos {

    namespace Impl {

        template <typename Functor, typename Validator>
        class ResilientReplicateFunctor
        {
        public:
            KOKKOS_FUNCTION ResilientReplicateFunctor(
                Functor const& f, Validator const& v, std::uint64_t n)
              : functor(f)
              , validator(v)
              , replicates(n)
            {
            }

            template <typename ValueType>
            KOKKOS_FUNCTION void operator()(ValueType i) const
            {
                using return_type =
                    typename std::invoke_result<Functor, ValueType>::type;

                bool is_valid = false;
                return_type final_result{};

                for (std::uint64_t n = 0u; n != replicates; ++n)
                {
                    auto result = functor(i);
                    bool is_correct = validator(i, result);

                    if (is_correct && !is_valid)
                    {
                        final_result = result;
                        is_valid = true;
                    }
                }

                if (!is_valid)
                {
#if defined(__CUDA_ARCH__)
                    // Define something
#else
                    throw std::runtime_error("All Replicates lead to failure.");
#endif
                }
            }

        private:
            Functor const& functor;
            Validator const& validator;
            std::uint64_t replicates;
        };

    }    // namespace Impl

    template <typename ExecutionSpace, typename Validator>
    class ResilientReplicate : public ExecutionSpace
    {
    public:
        // Typedefs for the ResilientReplicate Execution Space
        using base_execution_space = ExecutionSpace;
        using validator_type = Validator;

        using execution_space = ResilientReplicate;
        using memory_space = typename ExecutionSpace::memory_space;
        using device_type = typename ExecutionSpace::device_type;
        using size_type = typename ExecutionSpace::size_type;
        using scratch_memory_space =
            typename ExecutionSpace::scratch_memory_space;

        template <typename... Args>
        ResilientReplicate(std::uint64_t n, Validator const& validator,
            Args&&... args) noexcept
          : replays_(n)
          , validator_(validator)
          , ExecutionSpace(args...)
        {
        }

        Validator const& validator() const noexcept
        {
            return validator_;
        }

        std::uint64_t replays() const noexcept
        {
            return replays_;
        }

        KOKKOS_FUNCTION ResilientReplicate(
            ResilientReplicate&& other) noexcept = default;
        KOKKOS_FUNCTION ResilientReplicate(
            ResilientReplicate const& other) = default;

    private:
        Validator const& validator_;
        std::uint64_t replays_;
    };

    namespace Impl {

        template <typename FunctorType, typename... Traits>
        class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
            ResilientReplicate<typename traits::RangePolicyBase<
                                   Traits...>::base_execution_space,
                typename traits::RangePolicyBase<Traits...>::validator>>
          : public ParallelFor<
                ResilientReplicateFunctor<FunctorType,
                    typename traits::RangePolicyBase<Traits...>::validator>,
                typename traits::RangePolicyBase<Traits...>::RangePolicy,
                typename traits::RangePolicyBase<
                    Traits...>::base_execution_space>
        {
        public:
            using Policy = Kokkos::RangePolicy<Traits...>;
            using BasePolicy =
                typename traits::RangePolicyBase<Traits...>::RangePolicy;
            using validator_type =
                typename traits::RangePolicyBase<Traits...>::validator;
            using base_type = ParallelFor<
                ResilientReplicateFunctor<FunctorType, validator_type>,
                typename traits::RangePolicyBase<Traits...>::RangePolicy,
                typename traits::RangePolicyBase<
                    Traits...>::base_execution_space>;
            using base_execution_space = typename traits::RangePolicyBase<
                Traits...>::base_execution_space;

            ParallelFor(
                FunctorType const& arg_functor, const Policy& arg_policy)
              : base_type(
                    ResilientReplicateFunctor<FunctorType, validator_type>(
                        arg_functor, arg_policy.space().validator(),
                        arg_policy.space().replays()),
                    arg_policy)
            {
            }
        };

    }    // namespace Impl

}    // namespace Kokkos

namespace Kokkos { namespace Tools { namespace Experimental {

    template <typename ExecutionSpace, typename Validator>
    struct DeviceTypeTraits<
        Kokkos::ResilientReplicate<ExecutionSpace, Validator>>
    {
        static constexpr DeviceType id = DeviceTypeTraits<ExecutionSpace>::id;
    };

}}}    // namespace Kokkos::Tools::Experimental