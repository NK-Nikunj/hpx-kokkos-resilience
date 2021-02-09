#pragma once

#include <hpx/functional/tag_invoke.hpp>

namespace hpx { namespace kokkos { namespace resiliency {

    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replay_validate_t final
      : hpx::functional::tag<async_replay_validate_t>
    {
    } async_replay_validate{};

}}}    // namespace hpx::kokkos::resiliency