
#include <cmath>
#include <cassert>
#include <stdexcept>

#include "IGUVentilatedGapLayer.hpp"
#include "WCEGases.hpp"

namespace Tarcog
{
    namespace ISO15099
    {
        CIGUVentilatedGapLayer::CIGUVentilatedGapLayer(
          std::shared_ptr<CIGUGapLayer> const & t_Layer) :
            CIGUGapLayer(*t_Layer),
            m_Layer(t_Layer),
            m_inTemperature(Gases::DefaultTemperature),
            m_outTemperature(Gases::DefaultTemperature),
            m_Zin(0),
            m_Zout(0)
        {
            m_ReferenceGas = m_Gas;
            m_ReferenceGas.setTemperatureAndPressure(ReferenceTemperature, m_Pressure);
        }

        double CIGUVentilatedGapLayer::layerTemperature()
        {
            assert(m_Height != 0);
            const auto cHeight = characteristicHeight();
            const auto avTemp = averageTemperature();
            return avTemp - (cHeight / m_Height) * (m_outTemperature - m_inTemperature);
        }

        void CIGUVentilatedGapLayer::setFlowGeometry(double const t_Ain, double const t_Aout)
        {
            m_Ain = t_Aout;
            m_Aout = t_Ain;
        }

        void CIGUVentilatedGapLayer::setFlowDirection(const AirVerticalDirection t_Direction)
        {
            m_AirVerticalDirection = t_Direction;

            m_Zin = calcImpedance(m_Ain);
            m_Zout = calcImpedance(m_Aout);

            resetCalculated();
        }

        void CIGUVentilatedGapLayer::setInletTemperature(double inletTemperature)
        {
            m_InletTemperature = inletTemperature;
            calculateImpedances(inletTemperature);
        }

        void CIGUVentilatedGapLayer::calculateImpedances(double inletTemperature)
        {
            m_AirVerticalDirection = layerTemperature() > inletTemperature
                                       ? AirVerticalDirection::Up
                                       : AirVerticalDirection::Down;

            m_Zin = calcImpedance(m_Ain);
            m_Zout = calcImpedance(m_Aout);

            resetCalculated();
        }

        void CIGUVentilatedGapLayer::setFlowTemperatures(double const t_topTemp,
                                                         double const t_botTemp,
                                                         AirVerticalDirection const & t_Direction)
        {
            m_AirVerticalDirection = t_Direction;
            switch(m_AirVerticalDirection)
            {
                case AirVerticalDirection::None:
                    break;
                case AirVerticalDirection::Up:
                    m_inTemperature = t_botTemp;
                    m_outTemperature = t_topTemp;
                    break;
                case AirVerticalDirection::Down:
                    m_inTemperature = t_topTemp;
                    m_outTemperature = t_botTemp;
                    break;
                default:
                    throw std::runtime_error("Incorrect argument for airflow direction.");
                    break;
            }

            resetCalculated();
        }

        void CIGUVentilatedGapLayer::setFlowTemperatures(double t_inTemperature,
                                                         double t_outTemperature)
        {
            m_inTemperature = t_inTemperature;
            m_outTemperature = t_outTemperature;
            resetCalculated();
        }

        void CIGUVentilatedGapLayer::setFlowSpeed(double const t_speed)
        {
            m_AirSpeed = t_speed;
            resetCalculated();
        }

        double CIGUVentilatedGapLayer::getDrivingPressure()
        {
            using ConstantsData::GRAVITYCONSTANT;
            using ConstantsData::WCE_PI;

            const auto tiltAngle = WCE_PI / 180 * (m_Tilt - 90);
            const auto gapTemperature = layerTemperature();
            const auto aProperties = m_ReferenceGas.getGasProperties();
            const auto temperatureMultiplier =
              std::abs(gapTemperature - m_InletTemperature) / (gapTemperature * m_InletTemperature);
            return aProperties.m_Density * ReferenceTemperature * GRAVITYCONSTANT * m_Height
                   * std::abs(cos(tiltAngle)) * temperatureMultiplier;
        }

        double CIGUVentilatedGapLayer::bernoullyPressureTerm()
        {
            const auto aGasProperties = m_Gas.getGasProperties();
            return 0.5 * aGasProperties.m_Density;
        }

        double CIGUVentilatedGapLayer::hagenPressureTerm()
        {
            const auto aGasProperties = m_Gas.getGasProperties();
            return 12 * aGasProperties.m_Viscosity * m_Height / pow(getThickness(), 2);
        }

        double CIGUVentilatedGapLayer::pressureLossTerm()
        {
            calculateImpedances(m_InletTemperature);
            const auto aGasProperties = m_Gas.getGasProperties();
            return 0.5 * aGasProperties.m_Density * (m_Zin + m_Zout);
        }

        double CIGUVentilatedGapLayer::betaCoeff()
        {
            calculateLayerHeatFlow();
            return exp(-m_Height / characteristicHeight());
        }

        void CIGUVentilatedGapLayer::smoothEnergyGain(double const qv1, double const qv2)
        {
            const auto smooth = (std::abs(qv1) + std::abs(qv2)) / 2;
            m_LayerGainFlow = smooth;
            if(m_inTemperature < m_outTemperature)
            {
                m_LayerGainFlow = -m_LayerGainFlow;
            }
        }

        void CIGUVentilatedGapLayer::calculateConvectionOrConductionFlow()
        {
            CIGUGapLayer::calculateConvectionOrConductionFlow();
            if(!isCalculated())
            {
                ventilatedFlow();
            }
        }

        double CIGUVentilatedGapLayer::characteristicHeight()
        {
            const auto aProperties = m_Gas.getGasProperties();
            double cHeight = 0;
            // Characteristic height can only be calculated after initialization is performed
            if(!FenestrationCommon::isEqual(m_ConductiveConvectiveCoeff, 0))
            {
                cHeight = aProperties.m_Density * aProperties.m_SpecificHeat * getThickness()
                          * m_AirSpeed / (4 * m_ConductiveConvectiveCoeff);
            }
            return cHeight;
        }

        double CIGUVentilatedGapLayer::calcImpedance(double const t_A) const
        {
            auto impedance = 0.0;

            if(!FenestrationCommon::isEqual(t_A, 0))
            {
                impedance = pow(m_Width * getThickness() / (0.6 * t_A) - 1, 2);
            }

            return impedance;
        }

        void CIGUVentilatedGapLayer::ventilatedFlow()
        {
            const auto aProperties = m_Gas.getGasProperties();
            m_LayerGainFlow = aProperties.m_Density * aProperties.m_SpecificHeat * m_AirSpeed
                              * getThickness() * (m_inTemperature - m_outTemperature) / m_Height;
        }

        double CIGUVentilatedGapLayer::calculateThermallyDrivenSpeed()
        {
            double drivingPressure = getDrivingPressure();
            double A = bernoullyPressureTerm() + pressureLossTerm();
            double B = hagenPressureTerm();
            return (sqrt(std::abs(pow(B, 2) + 4 * A * drivingPressure)) - B) / (2 * A);
        }

        double CIGUVentilatedGapLayer::calculateOutletTemperatureFromAirFlow()
        {
            if(!isVentilationForced())
            {
                m_AirSpeed = calculateThermallyDrivenSpeed();
            }
            double beta = betaCoeff();
            double alpha = 1 - beta;

            return alpha * averageTemperature() + beta * m_InletTemperature;
        }

    }   // namespace ISO15099

}   // namespace Tarcog
