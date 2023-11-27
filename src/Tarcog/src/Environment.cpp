#include "Environment.hpp"
#include "TarcogConstants.hpp"


namespace Tarcog::ISO15099
{
    CEnvironment::CEnvironment(double t_Pressure,
                               double t_AirSpeed,
                               AirHorizontalDirection t_AirDirection) :
        m_DirectSolarRadiation(0),
        m_Emissivity(TarcogConstants::DEFAULT_ENV_EMISSIVITY),
        m_HInput(0),
        m_HCoefficientModel(BoundaryConditionsCoeffModel::CalculateH),
        m_IRCalculatedOutside(false),
        m_Pressure(t_Pressure),
        m_AirflowProperties(t_AirSpeed, AirVerticalDirection::None, t_AirDirection, false)
    {}

    CEnvironment::~CEnvironment()
    {
        tearDownConnections();
    }

    void CEnvironment::setHCoeffModel(BoundaryConditionsCoeffModel const t_BCModel,
                                      double const t_HCoeff)
    {
        m_HCoefficientModel = t_BCModel;
        m_HInput = t_HCoeff;
        resetCalculated();
        m_Gas.setTemperatureAndPressure(getGasTemperature(), m_Pressure);
    }

    void CEnvironment::setEnvironmentIR(double const t_InfraRed)
    {
        setIRFromEnvironment(t_InfraRed);
        m_IRCalculatedOutside = true;
        resetCalculated();
        m_Gas.setTemperatureAndPressure(getGasTemperature(), m_Pressure);
    }

    void CEnvironment::setEmissivity(double const t_Emissivity)
    {
        m_Emissivity = t_Emissivity;
        resetCalculated();
        m_Gas.setTemperatureAndPressure(getGasTemperature(), m_Pressure);
    }

    double CEnvironment::getEnvironmentIR()
    {
        calculateLayerHeatFlow();
        return getIRFromEnvironment();
    }

    double CEnvironment::getHc()
    {
        return getConductionConvectionCoefficient();
    }

    double CEnvironment::getAirTemperature()
    {
        return getGasTemperature();
    }

    double CEnvironment::getAmbientTemperature()
    {
        double hc = getHc();
        double hr = getHr();
        return (hc * getAirTemperature() + hr * getRadiationTemperature()) / (hc + hr);
    }

    double CEnvironment::getDirectSolarRadiation() const
    {
        return m_DirectSolarRadiation;
    }

    void CEnvironment::connectToIGULayer(std::shared_ptr<CBaseLayer> const &)
    {
        //
    }

    void CEnvironment::calculateRadiationFlow()
    {
        // In case of environments, there is no need to calculate radiation
        // if radiation is provided from outside calculations
        if(!m_IRCalculatedOutside)
        {
            setIRFromEnvironment(calculateIRFromVariables());
        }
    }

    double CEnvironment::getPressure() const
    {
        return m_Pressure;
    }

}   // namespace Tarcog::ISO15099
