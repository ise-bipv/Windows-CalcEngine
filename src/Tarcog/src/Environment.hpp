#ifndef TARENVIRONMENT_H
#define TARENVIRONMENT_H

#include <memory>

#include "BaseLayer.hpp"

namespace Tarcog
{

    namespace ISO15099
    {
        enum class BoundaryConditionsCoeffModel
        {
            CalculateH,
            HPrescribed,
            HcPrescribed
        };

        enum class Environment
        {
            Indoor,
            Outdoor
        };

    class CEnvironment : public Tarcog::CBaseLayer, public CGasLayer
        {
        public:
            CEnvironment(double t_Pressure,
                         double t_AirSpeed,
                         AirHorizontalDirection t_AirDirection);
            CEnvironment(const CEnvironment & t_Environment);
            CEnvironment & operator=(const CEnvironment & t_Environment);

            ~CEnvironment();

            void setHCoeffModel(BoundaryConditionsCoeffModel t_BCModel, double t_HCoeff = 0);
            void setForcedVentilation(const ForcedVentilation & t_ForcedVentilation);
            void setEnvironmentIR(double t_InfraRed);
            void setEmissivity(double t_Emissivity);

            double getDirectSolarRadiation() const;
            double getEnvironmentIR();
            double getHc();

            double getAirTemperature();
            double getAmbientTemperature();

            virtual void connectToIGULayer(const std::shared_ptr<CBaseLayer> & t_IGULayer);

            virtual std::shared_ptr<CEnvironment> cloneEnvironment() const = 0;

        protected:
            void initializeStateVariables() override;
            void calculateRadiationFlow() override;
            virtual double calculateIRFromVariables() = 0;
            virtual double getHr() = 0;
            virtual void setIRFromEnvironment(double t_IR) = 0;
            virtual double getIRFromEnvironment() const = 0;
            virtual double getRadiationTemperature() const = 0;

            double m_DirectSolarRadiation;
            double m_Emissivity;   // Emissivity from the environment
            // double m_InfraredRadiation; // Infrared radiation from environemnt [W/m2]
            double
              m_HInput;   // Input convection coefficient which type depends on selected BC model
            // [W/m2*K]
            BoundaryConditionsCoeffModel
              m_HCoefficientModel;   // Model used to calculate BC coefficient
            bool
              m_IRCalculatedOutside;   // Keep info if IR radiation is provided (calculated) outside
        };

    }   // namespace ISO15099

}   // namespace Tarcog

#endif
