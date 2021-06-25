#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <exception>
#include <type_traits>

namespace Kokkos {

    namespace Impl {

        template <typename Functor, typename Validator>
        class ResilientReplayFunctor
        {
        public:
            KOKKOS_FUNCTION ResilentFunctor(
                Functor const& f, Validator const& v, std::uint64_t n)
              : functor(f)
              , validator(v)
              , replays(n)
            {
            }

            template <typename ValueType>
            KOKKOS_FUNCTION void operator()(ValueType i) const
            {
                for (std::uint64_t n = 0u; n != replays; ++n)
                {
                    auto result = functor(i);
                    bool is_correct = validator(i, result);

                    if (is_correct)
                        break;

                    if (n == replays - 1)
                        throw std::runtime_error("Out of Replays");
                }
            }

        private:
            Functor const& functor;
            Validator const& validator;
            std::uint64_t replays;
        };

    }    // namespace Impl

    template <typename ExecutionSpace, typename Validator>
    class ResilientReplay : public ExecutionSpace
    {
    public:
        // Typedefs for the ResilientReplay Execution Space
        using base_execution_space = ExecutionSpace;
        using execution_space = ResilientReplay;
        using memory_space = typename ExecutionSpace::memory_space;
        using device_type = typename ExecutionSpace::device_type;
        using array_layout = typename ExecutionSpace::array_type;
        using size_type = typename ExecutionSpace::size_type;
        using scratch_memory_space =
            typename ExecutionSpace::scratch_memory_space;

        template <typename... Args>
        ResilientReplay(std::uint64_t n, Validator const& validator,
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

        KOKKOS_FUNCTION ResilientReplay(
            ResilientReplay&& other) noexcept = default;
        KOKKOS_FUNCTION ResilientReplay(ResilientReplay const& other) = default;

    private:
        Validator const& validator_;
        std::uint64_t replays_;
    };

    namespace Impl {

        namespace traits {

            template <typename ExecutionSpace, typename... Traits>
            struct RangePolicyBase
            {
                using traits = Kokkos::RangePolicy<
                    typename ExecutionSpace::base_execution_space, Traits...>;
            }
        }    // namespace traits

        template <typename FunctorType, typename... Traits>
        class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
            typename Kokkos::RangePolicy<Traits...>::traits::execution_space>
          : ParallelFor<FunctorType,
                Kokkos::RangePolicy<typename traits::RangePolicyBase::traits>,
                typename traits::RangePolicyBase::traits::traits::
                    execution_space>
        {
        public:
            using base_type = ParallelFor<FunctorType,
                Kokkos::RangePolicy<typename traits::RangePolicyBase::traits>,
                typename traits::RangePolicyBase::traits::traits::
                    execution_space>;

            ParallelFor(
                FunctorType const& arg_functor, const Policy& arg_policy)
              : base_type(ResilientReplayFunctor(arg_functor,
                              arg_policy.space().validator(),
                              arg_policy.space().replays()),
                    arg_policy)
            {
            }
        };

    }    // namespace Impl

}    // namespace Kokkos