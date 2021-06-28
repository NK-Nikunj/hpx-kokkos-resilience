#pragma once

#include <Kokkos_Core.hpp>

#include <hkr/util.hpp>

#include <cstdint>
#include <exception>
#include <type_traits>

namespace Kokkos {

    namespace Impl {

        template <typename ExecutionSpace, typename Functor, typename Validator>
        class ResilientReplicateFunctor
        {
        public:
            KOKKOS_FUNCTION ResilientReplicateFunctor(
                Functor const& f, Validator const& v, std::uint64_t n)
              : functor(f)
              , validator(v)
              , replicates(n)
              , incorrect_("result_correctness", 1)
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
                    incorrect_[0] = true;
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
            std::uint64_t replicates;
            Kokkos::View<bool*, ExecutionSpace> incorrect_;
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
          : replicates_(n)
          , validator_(validator)
          , ExecutionSpace(args...)
        {
        }

        Validator const& validator() const noexcept
        {
            return validator_;
        }

        std::uint64_t replicates() const noexcept
        {
            return replicates_;
        }

        KOKKOS_FUNCTION ResilientReplicate(
            ResilientReplicate&& other) noexcept = default;
        KOKKOS_FUNCTION ResilientReplicate(
            ResilientReplicate const& other) = default;

    private:
        const Validator validator_;
        const std::uint64_t replicates_;
    };

    namespace Impl {

        template <typename FunctorType, typename... Traits>
        class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
            ResilientReplicate<typename traits::RangePolicyBase<
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
                ParallelFor<ResilientReplicateFunctor<base_execution_space,
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
                ResilientReplicateFunctor<base_execution_space, FunctorType,
                    validator_type>
                    inst(m_functor, m_policy.space().validator(),
                        m_policy.space().replicates());

                // Call the underlying ParallelFor
                base_type closure(inst, m_policy);
                closure.execute();

                if (inst.is_incorrect())
                    throw std::runtime_error(
                        "All replicate returned incorrect result.");
            }

        private:
            const FunctorType m_functor;
            const Policy m_policy;
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