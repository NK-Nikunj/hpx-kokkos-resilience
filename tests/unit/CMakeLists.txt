add_custom_target(unit)

set(_tests
    async_replay_device
    async_replay_host
    async_replicate_device
    async_replicate_host
    experimental_replay
    experimental_replicate
    kokkos_execution_space
    returning_executor
    bulk_async
)

foreach(_test ${_tests})
    set(_test_name ${_test}_test)
    add_executable(${_test_name} ${_test}.cpp)
    target_link_libraries(${_test_name} PUBLIC HPX::hpx HPXKokkos::hpx_kokkos)
    add_dependencies(unit ${_test_name})
    add_test(NAME ${_test} COMMAND ${_test_name})
endforeach(_test ${_tests})
