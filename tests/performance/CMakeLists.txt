add_custom_target(performance)

set(_tests
    async_replay
    async_replicate
    experimental_replay
    experimental_replicate
    kokkos_execution_space
    plain_async
)

foreach(_test ${_tests})
    set(_test_name ${_test}_perf_test)
    add_executable(${_test_name} ${_test}.cpp)
    target_link_libraries(${_test_name} PUBLIC HPX::hpx HPX::wrap_main HPXKokkos::hpx_kokkos Boost::program_options)
    add_dependencies(performance ${_test_name})
    add_test(NAME ${_test} COMMAND ${_test_name})
endforeach(_test ${_tests})
