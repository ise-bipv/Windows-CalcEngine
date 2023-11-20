#ifndef BASEIGUTARCOGLAYER_H
#define BASEIGUTARCOGLAYER_H

#include <memory>
#include "BaseLayer.hpp"

namespace FenestrationCommon
{
    enum class Side;
}

namespace Tarcog::ISO15099
{
    class CBaseIGULayer : public Tarcog::ISO15099::CBaseLayer
    {
    public:
        explicit CBaseIGULayer(double t_Thickness);

        double getThickness() const override;
        double getTemperature(FenestrationCommon::Side t_Position) const;
        double J(FenestrationCommon::Side t_Position) const;
        virtual double getMaxDeflection() const = 0;
        virtual double getMeanDeflection() const = 0;

        double getConductivity();

        double getEffectiveThermalConductivity();

    protected:
        virtual double layerTemperature();

        double m_Thickness;
    };
}   // namespace Tarcog::ISO15099


#endif
