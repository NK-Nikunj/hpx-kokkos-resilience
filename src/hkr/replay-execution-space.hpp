#pragma once

#include <Kokkos_Core.hpp>

#include <hkr/util.hpp>

#include <cstdint>
#include <exception>
#include <type_traits>

namespace Kokkos {

    namespace Impl {

        template <typename ExecutionSpace, typename Functor, typename Validator>
        class ResilientReplayFunctor
        {
        public:
            KOKKOS_FUNCTION ResilientReplayFunctor(
                Functor const& f, Validator const& v, std::uint64_t n)
              : functor(f)
              , validator(v)
              , replays(n)
              , incorrect_("result_correctness", 1)
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
                        incorrect_[0] = true;
                }
            }

            bool is_incorrect() const
            {
                Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace>
                    return_result("is_correct", 1);

                Kokkos::deep_copy(return_result, incorrect_);

                return return_result[0];
            }

        private:
            const Functor functor;
            const Validator validator;
            std::uint64_t replays;
            Kokkos::View<bool*, ExecutionSpace> incorrect_;
        };

    }    // namespace Impl

    template <typename ExecutionSpace, typename Validator>
    class ResilientReplay : public ExecutionSpace
    {
    public:
        // Typedefs for the ResilientReplay Execution Space
        using base_execution_space = ExecutionSpace;
        using validator_type = Validator;

        using execution_space = ResilientReplay;
        using memory_space = typename ExecutionSpace::memory_space;
        using device_type = typename ExecutionSpace::device_type;
        using size_type = typename ExecutionSpace::size_type;
        using scratch_memory_space =
            typename ExecutionSpace::scratch_memory_space;

        template <typename... Args>
        ResilientReplay(std::uint64_t n, Validator const& validator,
            Args&&... args) noexcept
          : ExecutionSpace(args...)
          , validator_(validator)
          , replays_(n)
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

        template <typename FunctorType, typename... Traits>
        class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
            ResilientReplay<typename traits::RangePolicyBase<
                                Traits...>::base_execution_space,
                typename traits::RangePolicyBase<Traits...>::validator>>
        {
        public:
            using Policy = Kokkos::RangePolicy<Traits...>;
            using BasePolicy =
                typename traits::RangePolicyBase<Traits...>::RangePolicy;
            using validator_type =
                typename traits::RangePolicyBase<Traits...>::validator;
            using base_execution_space = typename traits::RangePolicyBase<
                Traits...>::base_execution_space;
            using base_type =
                ParallelFor<ResilientReplayFunctor<base_execution_space,
                                FunctorType, validator_type>,
                    typename traits::RangePolicyBase<Traits...>::RangePolicy,
                    typename traits::RangePolicyBase<
                        Traits...>::base_execution_space>;

            ParallelFor(
                FunctorType const& arg_functor, const Policy& arg_policy)
              : m_functor(arg_functor)
              , m_policy(arg_policy)
            {
            }

            void execute() const
            {
                ResilientReplayFunctor<base_execution_space, FunctorType,
                    validator_type>
                    inst(m_functor, m_policy.space().validator(),
                        m_policy.space().replays());

                // Call the underlying ParallelFor
                base_type closure(inst, m_policy);
                closure.execute();

                if (inst.is_incorrect())
                    throw std::runtime_error(
                        "Program ran out of replay options.");
            }

        private:
            const FunctorType m_functor;
            const Policy m_policy;
        };

    }    // namespace Impl

}    // namespace Kokkos

namespace Kokkos { namespace Tools { namespace Experimental {

    template <typename ExecutionSpace, typename Validator>
    struct DeviceTypeTraits<Kokkos::ResilientReplay<ExecutionSpace, Validator>>
    {
        static constexpr DeviceType id = DeviceTypeTraits<ExecutionSpace>::id;
    };

}}}    // namespace Kokkos::Tools::Experimental