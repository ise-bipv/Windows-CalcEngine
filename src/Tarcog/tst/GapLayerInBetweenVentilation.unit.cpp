#include <memory>
#include <gtest/gtest.h>

#include "WCETarcog.hpp"

class TestGapLayerInBetweenVentilation : public testing::Test
{
private:
    std::shared_ptr<Tarcog::ISO15099::CSingleSystem> m_TarcogSystem;

protected:
    void SetUp() override
    {
        /////////////////////////////////////////////////////////
        // Outdoor
        /////////////////////////////////////////////////////////
        auto airTemperature = 255.15;   // Kelvins
        auto airSpeed = 5.5;            // meters per second
        auto tSky = 255.15;             // Kelvins
        auto solarRadiation = 0.0;

        auto Outdoor = Tarcog::ISO15099::Environments::outdoor(
          airTemperature, airSpeed, solarRadiation, tSky, Tarcog::ISO15099::SkyModel::AllSpecified);
        ASSERT_TRUE(Outdoor != nullptr);
        Outdoor->setHCoeffModel(Tarcog::ISO15099::BoundaryConditionsCoeffModel::CalculateH);

        /////////////////////////////////////////////////////////
        /// Indoor
        /////////////////////////////////////////////////////////

        auto roomTemperature = 295.15;

        auto Indoor = Tarcog::ISO15099::Environments::indoor(roomTemperature);
        ASSERT_TRUE(Indoor != nullptr);

        // IGU
        auto solidLayerThickness = 0.005715;   // [m]
        auto solidLayerConductance = 1.0;

        auto SolidLayer1 =
          Tarcog::ISO15099::Layers::solid(solidLayerThickness, solidLayerConductance);
        ASSERT_TRUE(SolidLayer1 != nullptr);

        auto SolidLayer2 =
          Tarcog::ISO15099::Layers::solid(solidLayerThickness, solidLayerConductance);
        ASSERT_TRUE(SolidLayer2 != nullptr);

        auto shadeLayerThickness = 0.01;
        auto shadeLayerConductance = 160.0;
        auto Atop = 0.1;
        auto Abot = 0.1;
        auto Aleft = 0.1;
        auto Aright = 0.1;
        auto Afront = 0.2;

        auto shadeLayer = Tarcog::ISO15099::Layers::shading(
          shadeLayerThickness, shadeLayerConductance, Atop, Abot, Aleft, Aright, Afront);

        ASSERT_TRUE(shadeLayer != nullptr);

        auto gapThickness = 0.0127;
        auto gapPressure = 101325.0;
        auto gapLayer1 = Tarcog::ISO15099::Layers::gap(gapThickness, gapPressure);
        ASSERT_TRUE(gapLayer1 != nullptr);

        auto gapLayer2 = Tarcog::ISO15099::Layers::gap(gapThickness, gapPressure);
        ASSERT_TRUE(gapLayer2 != nullptr);

        double windowWidth = 1;
        double windowHeight = 1;
        Tarcog::ISO15099::CIGU aIGU(windowWidth, windowHeight);
        aIGU.addLayer(SolidLayer1);
        aIGU.addLayer(gapLayer1);
        aIGU.addLayer(shadeLayer);
        aIGU.addLayer(gapLayer2);
        aIGU.addLayer(SolidLayer2);

        /////////////////////////////////////////////////////////
        /// System
        /////////////////////////////////////////////////////////
        m_TarcogSystem = std::make_shared<Tarcog::ISO15099::CSingleSystem>(aIGU, Indoor, Outdoor);
        ASSERT_TRUE(m_TarcogSystem != nullptr);
    }

public:
    std::shared_ptr<Tarcog::ISO15099::CBaseIGULayer> GetGap1() const
    {
        return m_TarcogSystem->getGapLayers()[0];
    };

    std::shared_ptr<Tarcog::ISO15099::CBaseIGULayer> GetGap2() const
    {
        return m_TarcogSystem->getGapLayers()[1];
    };
};

TEST_F(TestGapLayerInBetweenVentilation, VentilationFlow)
{
    SCOPED_TRACE("Begin Test: Test Ventilated Gap Layer - Intial Airflow");

    auto aLayer = GetGap1();

    // Airflow iterations are set to 1e-4 and it cannot exceed that precision

    ASSERT_TRUE(aLayer != nullptr);
    auto gainEnergy = aLayer->getGainFlow();
    EXPECT_NEAR(32.414571203538848, gainEnergy, 1e-4);

    aLayer = GetGap2();
    ASSERT_TRUE(aLayer != nullptr);
    gainEnergy = aLayer->getGainFlow();
    EXPECT_NEAR(-32.414571203538848, gainEnergy, 1e-4);
}
