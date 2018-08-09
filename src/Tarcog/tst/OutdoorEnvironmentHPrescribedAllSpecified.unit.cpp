#include <memory>
#include <stdexcept>
#include <gtest/gtest.h>

#include "WCETarcog.hpp"

using namespace Tarcog;


class TestOutdoorEnvironmentHPrescribedAllSpecified : public testing::Test
{
private:
    std::shared_ptr<CEnvironment> Outdoor;
    std::shared_ptr<CSingleSystem> m_TarcogSystem;

protected:
    void SetUp() override
    {
        /////////////////////////////////////////////////////////
        /// Outdoor
        /////////////////////////////////////////////////////////
        auto airTemperature = 300.0;   // Kelvins
        auto pressure = 101325.0;      // Pascals
        auto airSpeed = 5.5;           // meters per second
        auto tSky = 270.0;   // Kelvins
        auto solarRadiation = 0.0;
        auto hout = 20.0;

        Outdoor = Environments::outdoor(airTemperature,
                                        pressure,
                                        airSpeed,
                                        solarRadiation,
                                        tSky,
                                        SkyModel::AllSpecified);
        ASSERT_TRUE(Outdoor != nullptr);
        Outdoor->setHCoeffModel(BoundaryConditionsCoeffModel::HPrescribed, hout);

        /////////////////////////////////////////////////////////
        /// Indoor
        /////////////////////////////////////////////////////////
        auto roomTemperature = 294.15;
        auto Indoor = Environments::indoor(roomTemperature, pressure);
        ASSERT_TRUE(Indoor != nullptr);

        /////////////////////////////////////////////////////////
        /// IGU
        /////////////////////////////////////////////////////////
        auto solidLayerThickness = 0.003048;   // [m]
        auto solidLayerConductance = 100.0;

        auto aSolidLayer = Layers::solid(solidLayerThickness, solidLayerConductance);
        ASSERT_TRUE(aSolidLayer != nullptr);

        auto windowWidth = 1.0;
        auto windowHeight = 1.0;
        CIGU aIGU(windowWidth, windowHeight);
        aIGU.addLayer(aSolidLayer);

        /////////////////////////////////////////////////////////
        /// System
        /////////////////////////////////////////////////////////
        m_TarcogSystem = std::make_shared<CSingleSystem>(aIGU, Indoor, Outdoor);
        m_TarcogSystem->solve();
        ASSERT_TRUE(m_TarcogSystem != nullptr);
    }

public:
    std::shared_ptr<CEnvironment> GetOutdoors() const
    {
        return Outdoor;
    };
};

TEST_F(TestOutdoorEnvironmentHPrescribedAllSpecified, HPrescribed_AllSpecified)
{
    SCOPED_TRACE("Begin Test: Outdoors -> H model = Prescribed; Sky Model = All Specified");

    auto aOutdoor = GetOutdoors();
    ASSERT_TRUE(aOutdoor != nullptr);

    auto radiosity = aOutdoor->getEnvironmentIR();
    EXPECT_NEAR(459.2457, radiosity, 1e-5);

    auto hc = aOutdoor->getHc();
    EXPECT_NEAR(14.895502, hc, 1e-5);

    auto outIR = aOutdoor->getRadiationFlow();
    EXPECT_NEAR(-7.777658, outIR, 1e-5);

    auto outConvection = aOutdoor->getConvectionConductionFlow();
    EXPECT_NEAR(-22.696083, outConvection, 1e-5);

    auto totalHeatFlow = aOutdoor->getHeatFlow();
    EXPECT_NEAR(-30.473740, totalHeatFlow, 1e-5);
}
