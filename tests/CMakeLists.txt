add_custom_target(tests)

set(_tests
    # async_replay_device
    # async_replay_host
    returning_executor
)

foreach(_test ${_tests})
    set(_test_name ${_test}_test)
    add_executable(${_test_name} ${_test}.cpp)
    target_link_libraries(${_test_name} PUBLIC HPX::hpx HPXKokkos::hpx_kokkos)
    add_dependencies(tests ${_test_name})
    add_test(NAME ${_test} COMMAND ${_test_name})
endforeach(_test ${_tests})
