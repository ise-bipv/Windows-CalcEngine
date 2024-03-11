#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include <memory>
#include <gtest/gtest.h>

// Inspired by https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/powell.cc

// Variables: Tg1, Tc1, Tso1, Tsi1, Tt1, Tg2, Tc2, Tso2, Tsi2, Tt2, Tg3, Tc3, Tso3, Tsi3, Tt3
// Variables: tGlass1, tCavity1, tScreenExterior1, tScreenInterior1, tOpening1

struct F1
{
    template<typename T>
    bool operator()(const T * const tGlass1, T * residual) const
    {
        residual[0] = tGlass1[0] - 5.0;
        return true;
    }
};
struct F2
{
    template<typename T>
    bool operator()(const T * const tCavity1, T * residual) const
    {
        residual[0] = tCavity1[0] - 10.0;
        return true;
    }
};
struct F3
{
    template<typename T>
    bool operator()(const T * const tScreenExterior1, T * residual) const
    {
        residual[0] = tScreenExterior1[0] - 15.0;
        return true;
    }
};
struct F4
{
    template<typename T>
    bool operator()(const T * const tScreenInterior1, T * residual) const
    {
        residual[0] = tScreenInterior1[0] - 20.0;
        return true;
    }
};
struct F5
{
    template<typename T>
    bool operator()(const T * const tOpening1, T * residual) const
    {
        residual[0] = tOpening1[0] - 25.0;
        return true;
    }
};

DEFINE_string(minimizer,
              "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");

class VentilatedScreenThermalNetwork : public testing::Test
{
private:
protected:
    void SetUp() override
    {}

public:
};

TEST_F(VentilatedScreenThermalNetwork, ProofOfConcept)
{
    SCOPED_TRACE("Begin Test: Vertical Thermal Network of a Ventilated Interior Screen");

    // GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    // google::InitGoogleLogging(argv[0]);

    // The variable to solve for with its initial value.
    double tGlass1 = 0;
    double tCavity1 = 0;
    double tScreenExterior1 = 0;
    double tScreenInterior1 = 0;
    double tOpening1 = 0;

    ceres::Problem problem;
    // Add residual terms to the problem using the autodiff
    // wrapper to get the derivatives automatically. Define the variables/parameters of each
    // Functor.
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(), nullptr, &tGlass1);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(), nullptr, &tCavity1);
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(), nullptr, &tScreenExterior1);
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(), nullptr, &tScreenInterior1);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F5, 1, 1, 1>(), nullptr, &tOpening1);

    ceres::Solver::Options options;

    LOG_IF(FATAL,
           !ceres::StringToMinimizerType(CERES_GET_FLAG(FLAGS_minimizer), &options.minimizer_type))
      << "Invalid minimizer: " << CERES_GET_FLAG(FLAGS_minimizer)
      << ", valid options are: trust_region and line_search.";
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // clang-format off
  std::cout << "Initial tGlass1 = " << tGlass1
            << ", tCavity1 = " << tCavity1
            << ", tScreenExterior1 = " << tScreenExterior1
            << ", tScreenInterior1 = " << tScreenInterior1
            << ", tOpening1 = " << tOpening1
            << "\n";
    // clang-format on

    // Run the solver!
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // clang-format off
  std::cout << "Final tGlass1 = " << tGlass1
            << ", tCavity1 = " << tCavity1
            << ", tScreenExterior1 = " << tScreenExterior1
            << ", tScreenInterior1 = " << tScreenInterior1
            << ", tOpening1 = " << tOpening1
            << "\n";
    // clang-format on

    EXPECT_NEAR(5.0, tGlass1, 1e-4);
    EXPECT_NEAR(10.0, tCavity1, 1e-4);
    EXPECT_NEAR(15.0, tScreenExterior1, 1e-4);
    EXPECT_NEAR(20.0, tScreenInterior1, 1e-4);
    EXPECT_NEAR(25.0, tOpening1, 1e-4);
}
