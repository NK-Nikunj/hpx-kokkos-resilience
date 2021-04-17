// #include <hpx/kokkos.hpp>

// #include <hpx/kokkos/detail/polling_helper.hpp>
// #include <hkr/executor/returning-executor.hpp>
// #include <hkr/hpx-kokkos-resiliency-executor.hpp>
// #include <hkr/hpx-kokkos-resiliency.hpp>

// #include <hpx/modules/synchronization.hpp>
// #include <boost/program_options.hpp>

// #include <random>

// namespace stencil {

//     // HPX_HOST_DEVICE double const checksum_tol = 1.0e-10;

//     template <typename ExecutionSpace>
//     struct partition_data;

//     template <typename ExecutionSpace>
//     using partition = hpx::shared_future<partition_data<ExecutionSpace>>;

//     template <typename ExecutionSpace>
//     using space = Kokkos::View<partition<ExecutionSpace>*, ExecutionSpace>;

//     template <typename ExecutionSpace>
//     struct partition_data
//     {
//         using execution_space = ExecutionSpace;

//         HPX_HOST_DEVICE partition_data(std::uint64_t size)
//           : data_("subdomain_space", size)
//           , size_(size)
//           , checksum_(0.0)
//           , test_value_(0.0)
//         {
//         }

//         HPX_HOST_DEVICE partition_data(std::uint64_t subdomain_width,
//             double subdomain_index, std::uint64_t subdomains)
//           : data_("subdomain_space", subdomain_width + 1)
//           , size_(subdomain_width + 1)
//           , test_value_(0.0)
//         {
//             constexpr double pi = 3.1415926535;

//             checksum_ = 0.0;
//             for (std::uint64_t k = 0; k != subdomain_width + 1; ++k)
//             {
//                 data_[k] = std::sin(2 * pi *
//                     ((0.0 + subdomain_width * subdomain_index + k) /
//                         static_cast<double>(subdomain_width * subdomains)));
//                 checksum_ += data_[k];
//             }
//         }

//         HPX_HOST_DEVICE partition_data(partition_data&& other)
//           : data_(std::move(other.data_))
//           , size_(other.size_)
//           , checksum_(other.checksum_)
//           , test_value_(other.test_value_)
//         {
//         }

//         // To support copying over KOKKOS_LAMBDA (Only shallow copy required)
//         HPX_HOST_DEVICE partition_data(partition_data const& other)
//           : data_(other.data_)
//           , size_(other.size_)
//           , checksum_(other.checksum_)
//           , test_value_(other.test_value_)
//         {
//         }

//         HPX_HOST_DEVICE double& operator[](std::uint64_t idx)
//         {
//             return data_[idx];
//         }
//         HPX_HOST_DEVICE double operator[](std::uint64_t idx) const
//         {
//             return data_[idx];
//         }

//         HPX_HOST_DEVICE std::uint64_t size() const
//         {
//             return size_;
//         }

//         HPX_HOST_DEVICE double checksum() const
//         {
//             return checksum_;
//         }

//         HPX_HOST_DEVICE void set_checksum()
//         {
//             checksum_ = std::accumulate(data_.begin(), data_.end(), 0.0);
//         }

//         HPX_HOST_DEVICE void set_test_value(double test_value)
//         {
//             test_value_ = test_value;
//         }

//         HPX_HOST_DEVICE double verify_result() const
//         {
//             return std::abs(checksum_ - test_value_);
//         }

//         HPX_HOST_DEVICE void resize(std::uint64_t size)
//         {
//             Kokkos::resize(data_, size);
//             size_ = size;
//         }

//     private:
//         Kokkos::View<double*, execution_space> data_;
//         std::uint64_t size_;
//         double checksum_;
//         double test_value_;
//     };

//     template <typename ExecutionSpace, typename Executor>
//     hpx::future<space<ExecutionSpace>> do_work(Executor& exec,
//         std::uint64_t subdomains, std::uint64_t subdomain_width,
//         std::uint64_t iterations, std::uint64_t sti, std::uint64_t nd,
//         std::uint64_t n_value, std::uint64_t error,
//         hpx::lcos::local::sliding_semaphore& sem)
//     {
//         using hpx::kokkos::resiliency::dataflow_replay_validate;

//         Kokkos::View<space<ExecutionSpace>*, ExecutionSpace> U(
//             "stencil_space", 2);
//         for (int i = 0; i != 2; ++i)
//             Kokkos::resize(U[i], subdomains);

//         hpx::for_loop(
//             hpx::kokkos::kok.on(exec), 0u, subdomains,
//             KOKKOS_LAMBDA(std::uint64_t i) {
//                 U[0][i] = hpx::async(
//                     exec, KOKKOS_LAMBDA() {
//                         return partition_data<ExecutionSpace>(
//                             subdomain_width, double(i), subdomains);
//                     });
//             });
//     }

// }    // namespace stencil

// int main(int argc, char* argv[])
// {
//     Kokkos::initialize(argc, argv);

//     {
//         hpx::kokkos::detail::polling_helper helper;

//         hpx::kokkos::returning_executor exec_;
//         hpx::kokkos::returning_host_executor host_exec_;

//         namespace bpo = boost::program_options;

//         bpo::options_description desc("Options");

//         desc.add_options()("n-value",
//             bpo::value<std::uint64_t>()->default_value(3),
//             "Number of repeat launches for async replay.");
//         desc.add_options()("error-rate",
//             bpo::value<std::uint64_t>()->default_value(1),
//             "Error rate for injecting errors.");
//         desc.add_options()("subdomain-width",
//             bpo::value<std::uint64_t>()->default_value(128),
//             "Local x dimension (of each partition).");
//         desc.add_options()("iterations",
//             bpo::value<std::uint64_t>()->default_value(10),
//             "Number of time steps.");
//         desc.add_options()("steps-per-iteration",
//             bpo::value<std::uint64_t>()->default_value(16),
//             "Number of time steps per iterations.");
//         desc.add_options()("subdomains",
//             bpo::value<std::uint64_t>()->default_value(10),
//             "Number of partitions.");
//         desc.add_options()("nd", bpo::value<std::uint64_t>()->default_value(10),
//             "Number of time steps to allow the dependency tree to grow to");

//         bpo::variables_map vm;

//         // Setup commandline arguments
//         bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
//         bpo::notify(vm);

//         // Start application work
//         std::uint64_t n = vm["n-value"].as<std::uint64_t>();
//         std::uint64_t error = vm["error-rate"].as<std::uint64_t>();
//         std::uint64_t subdomain_width =
//             vm["subdomain-width"].as<std::uint64_t>();
//         std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
//         std::uint64_t subdomains = vm["subdomains"].as<std::uint64_t>();
//         std::uint64_t sti = vm["steps-per-iteration"].as<std::uint64_t>();
//         std::uint64_t nd = vm["nd"].as<std::uint64_t>();

//         {
//             std::cout << "Starting 1d stencil with dataflow replay validate"
//                       << std::endl;

//             // Measure execution time.
//             std::uint64_t t = hpx::chrono::high_resolution_clock::now();

//             {
//                 // limit depth of dependency tree
//                 hpx::lcos::local::sliding_semaphore sem(nd);

//                 hpx::future<
//                     typename stencil::space<Kokkos::DefaultExecutionSpace>>
//                     result = stencil::do_work<Kokkos::DefaultExecutionSpace>(
//                         exec_, subdomains, subdomain_width, iterations, sti, nd,
//                         n, error, sem);

//                 typename stencil::space<Kokkos::DefaultExecutionSpace>
//                     solution = result.get();
//                 hpx::wait_all(solution);
//             }

//             std::cout << "Time elapsed: "
//                       << static_cast<double>(
//                              hpx::chrono::high_resolution_clock::now() - t) /
//                     1e9
//                       << std::endl;
//         }
//     }
// }