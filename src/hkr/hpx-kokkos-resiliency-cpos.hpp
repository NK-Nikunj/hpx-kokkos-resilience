#pragma once

#include <hpx/functional/tag_invoke.hpp>

namespace hpx { namespace kokkos { namespace resiliency {

    ///////////////////////////////////////////////////////////////////////////
    // helper base class implementing the deferred tag_invoke logic for CPOs
    template <typename Tag, typename BaseTag>
    struct tag_deferred : hpx::functional::tag<Tag>
    {
        // force unwrapping of the inner future on return
        template <typename... Args>
        friend HPX_FORCEINLINE auto tag_invoke(Tag, Args&&... args) ->
            typename hpx::functional::tag_invoke_result<BaseTag,
                Args&&...>::type
        {
            return hpx::dataflow(BaseTag{}, std::forward<Args>(args)...);
        }
    };

    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replay_validate_t final
      : hpx::functional::tag<async_replay_validate_t>
    {
    } async_replay_validate{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replay_validate_t final
      : tag_deferred<dataflow_replay_validate_t, async_replay_validate_t>
    {
    } dataflow_replay_validate{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replicate_validate_t final
      : hpx::functional::tag<async_replicate_validate_t>
    {
    } async_replicate_validate{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replicate_validate_t final
      : tag_deferred<dataflow_replicate_validate_t, async_replicate_validate_t>
    {
    } dataflow_replicate_validate{};

}}}    // namespace hpx::kokkos::resiliency