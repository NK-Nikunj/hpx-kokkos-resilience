#include <hpx/kokkos.hpp>

#include <hpx/kokkos/detail/polling_helper.hpp>
#include <hkr/executor/returning-executor.hpp>
#include <hkr/hpx-kokkos-resiliency-executor.hpp>
#include <hkr/hpx-kokkos-resiliency.hpp>

#include <boost/program_options.hpp>

#include <random>

struct validate_exception : std::exception
{
};

///////////////////////////////////////////////////////////////////////////////
double const pi = std::acos(-1.0);
double const checksum_tol = 1.0e-10;

// Variable to count the number of failed attempts
std::atomic<int> counter(0);

// Variables to generate errors
std::random_device rd;
std::mt19937 gen(rd());

///////////////////////////////////////////////////////////////////////////////
// Our partition data type
struct partition_data
{
public:
    partition_data(std::size_t size)
      : data_(size)
      , size_(size)
      , checksum_(0.0)
      , test_value_(0.0)
    {
    }

    partition_data(std::size_t subdomain_width, double subdomain_index,
        std::size_t subdomains)
      : data_(subdomain_width + 1)
      , size_(subdomain_width + 1)
      , test_value_(0.0)
    {
        checksum_ = 0.0;
        for (std::size_t k = 0; k != subdomain_width + 1; ++k)
        {
            data_[k] = std::sin(2 * pi *
                ((0.0 + subdomain_width * subdomain_index + k) /
                    static_cast<double>(subdomain_width * subdomains)));
            checksum_ += data_[k];
        }
    }

    partition_data(partition_data&& other)
      : data_(std::move(other.data_))
      , size_(other.size_)
      , checksum_(other.checksum_)
      , test_value_(other.test_value_)
    {
    }

    double& operator[](std::size_t idx)
    {
        return data_[idx];
    }
    double operator[](std::size_t idx) const
    {
        return data_[idx];
    }

    std::size_t size() const
    {
        return size_;
    }

    double checksum() const
    {
        return checksum_;
    }

    void set_checksum()
    {
        checksum_ = std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    void set_test_value(double test_value)
    {
        test_value_ = test_value;
    }

    double verify_result() const
    {
        return std::abs(checksum_ - test_value_);
    }

    friend std::vector<double>::const_iterator begin(const partition_data& v)
    {
        return begin(v.data_);
    }
    friend std::vector<double>::const_iterator end(const partition_data& v)
    {
        return end(v.data_);
    }

    void resize(std::size_t size)
    {
        data_.resize(size);
        size_ = size;
    }

private:
    // Kokkos::View<double*,
    std::vector<double> data_;
    std::size_t size_;
    double checksum_;
    double test_value_;
};

// Function declaration
bool validate_result(partition_data const& f);

///////////////////////////////////////////////////////////////////////////////
struct stepper
{
    // Our data for one time step
    typedef hpx::shared_future<partition_data> partition;
    typedef std::vector<partition> space;

    // Our operator
    static double stencil(double left, double center, double right)
    {
        return 0.5 * (0.75) * left + (0.75) * center - 0.5 * (0.25) * right;
    }

    static double left_flux(double left, double center)
    {
        return (0.625) * left - (0.125) * center;
    }

    static double right_flux(double center, double right)
    {
        return 0.5 * (0.75) * center + (1.125) * right;
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    static partition_data heat_part(std::size_t sti, double error,
        partition_data const& left_input, partition_data const& center_input,
        partition_data const& right_input)
    {
        // static thread_local std::exponential_distribution<> dist_(error);

        // double num = dist_(gen);
        // bool error_flag = false;

        // // Probability of error occurrence is proportional to exp(-error_rate)
        // if (num > 1.0)
        // {
        //     error_flag = true;
        //     ++counter;
        // }

        const std::size_t size = center_input.size() - 1;
        partition_data workspace(size + 2 * sti + 1);

        std::copy(
            end(left_input) - sti - 1, end(left_input) - 1, &workspace[0]);
        std::copy(begin(center_input), end(center_input) - 1, &workspace[sti]);
        std::copy(begin(right_input), begin(right_input) + sti + 1,
            &workspace[size + sti]);

        double left_checksum =
            std::accumulate(end(left_input) - sti - 1, end(left_input), 0.0);
        double right_checksum = std::accumulate(
            begin(right_input), begin(right_input) + sti + 1, 0.0);

        double checksum = left_checksum - center_input[0] +
            center_input.checksum() - right_input[0] + right_checksum;

        for (std::size_t t = 0; t != sti; ++t)
        {
            checksum -= left_flux(workspace[0], workspace[1]);
            checksum -= right_flux(workspace[size + 2 * sti - 1 - 2 * t],
                workspace[size + 2 * sti - 2 * t]);
            for (std::size_t k = 0; k != size + 2 * sti - 1 - 2 * t; ++k)
                workspace[k] =
                    stencil(workspace[k], workspace[k + 1], workspace[k + 2]);
        }

        workspace.resize(size + 1);
        workspace.set_checksum();
        workspace.set_test_value(checksum);

        // Artificial error injection to get replay in action
        // if (error_flag)
        //     throw validate_exception();

        return workspace;
    }

    hpx::future<space> do_work(std::size_t subdomains,
        std::size_t subdomain_width, std::size_t iterations, std::size_t sti,
        std::uint64_t nd, std::uint64_t n_value, double error,
        hpx::lcos::local::sliding_semaphore& sem)
    {
        using hpx::kokkos::resiliency::dataflow_replay_validate;
        using hpx::util::unwrapping;

        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2);
        for (space& s : U)
            s.resize(subdomains);

        std::size_t b = 0;
        auto range = boost::irange(b, subdomains);
        hpx::ranges::for_each(hpx::execution::par, range,
            [&U, subdomain_width, subdomains](std::size_t i) {
                U[0][i] = hpx::make_ready_future(
                    partition_data(subdomain_width, double(i), subdomains));
            });

        auto Op = unwrapping(&stepper::heat_part);

        // Actual time step loop
        for (std::size_t t = 0; t != iterations; ++t)
        {
            space const& current = U[t % 2];
            space& next = U[(t + 1) % 2];

            for (std::size_t i = 0; i != subdomains; ++i)
            {
                // explicitly unwrap future
                hpx::future<partition_data> f =
                    dataflow_replay_validate(n_value, validate_result, Op, sti,
                        error, current[(i - 1 + subdomains) % subdomains],
                        current[i], current[(i + 1) % subdomains]);
                next[i] = std::move(f);
            }

            // every nd time steps, attach additional continuation which will
            // trigger the semaphore once computation has reached this point
            if ((t % nd) == 0)
            {
                next[0].then([&sem, t](partition&&) {
                    // inform semaphore about new lower limit
                    sem.signal(t);
                });
            }

            // suspend if the tree has become too deep, the continuation above
            // will resume this thread once the computation has caught up
            sem.wait(t);
        }

        // Return the solution at time-step 'iterations'.
        return hpx::when_all(U[iterations % 2]);
    }
};

bool validate_result(partition_data const& f)
{
    if (f.verify_result() <= checksum_tol)
        return true;

    return false;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        hpx::kokkos::detail::polling_helper helper;

        hpx::kokkos::returning_executor exec_;
        hpx::kokkos::returning_host_executor host_exec_;

        namespace bpo = boost::program_options;

        bpo::options_description desc("Options");

        desc.add_option()("n-value",
            bpo::value<std::uint64_t>()->default_value(3),
            "Number of repeat launches for async replay.");
        desc.add_option()("error-rate",
            bpo::value<std::uint64_t>()->default_value(1),
            "Error rate for injecting errors.");
        desc.add_option()("subdomain-width",
            bpo::value<std::uint64_t>()->default_value(128),
            "Local x dimension (of each partition).");
        desc.add_option()("iterations",
            bpo::value<std::uint64_t>()->default_value(10),
            "Number of time steps.");
        desc.add_option()("steps-per-iteration",
            bpo::value<std::uint64_t>()->default_value(16),
            "Number of time steps per iterations.");
        desc.add_option()("subdomains",
            bpo::value<std::uint64_t>()->default_value(10),
            "Number of partitions.");
        desc_commandline.add_options()("nd",
            value<std::uint64_t>()->default_value(10),
            "Number of time steps to allow the dependency tree to grow to");

        bpo::variables_map vm;

        // Setup commandline arguments
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);

        // Start application work
        std::uint64_t n = vm["n-value"].as<std::uint64_t>;
        std::uint64_t error = vm["error-rate"].as<std::uint64_t>;
        std::uint64_t subdomain_width = vm["subdomain-width"].as<std::uint64_t>;
        std::uint64_t iterations = vm["iterations"].as<std::uint64_t>;
        std::uint64_t subdomains = vm["subdomains"].as<std::uint64_t>;
        std::uint64_t sti = vm["steps-per-iteration"].as<std::uint64_t>;
        std::uint64_t nd = vm["nd"].as<std::uint64_t>();

        {
            // Create the stepper object
            stepper step;

            std::cout << "Starting 1d stencil with dataflow replay validate"
                      << std::endl;

            // Measure execution time.
            std::uint64_t t = hpx::chrono::high_resolution_clock::now();

            {
                // limit depth of dependency tree
                hpx::lcos::local::sliding_semaphore sem(nd);

                hpx::future<stepper::space> result = step.do_work(subdomains,
                    subdomain_width, iterations, sti, nd, n, error, sem);

                stepper::space solution = result.get();
                hpx::wait_all(solution);
            }

            std::cout << "Time elapsed: "
                      << static_cast<double>(
                             hpx::chrono::high_resolution_clock::now() - t) /
                    1e9
                      << std::endl;
        }
    }
}
