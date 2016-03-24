#ifndef TARIGUSOLIDLAYER_H
#define TARIGUSOLIDLAYER_H

#include <memory>
#include "BaseIGUTarcogLayer.hpp"

namespace Tarcog {

  class CTarSurface;
  enum class SurfacePosition;

  class CTarIGUSolidLayer :
    public CBaseIGUTarcogLayer {
  public:
    CTarIGUSolidLayer( double const t_Thickness, double const t_Conductivity, 
      std::shared_ptr< CTarSurface > t_FrontSurface = nullptr, std::shared_ptr< CTarSurface > t_BackSurface=nullptr );

    void connectToBackSide( std::shared_ptr< CBaseTarcogLayer > t_Layer );

    void setLayerState( double const t_Tf, double const t_Tb, double const t_Jf, double const t_Jb );
    void setSolarRadiation( double const t_SolarRadiation );
    void setSolarAbsorptance( double const t_SolarAbsorptance );

  protected:
    virtual void calculateConvectionConductionState();

  private:
    void setSurfaceState( double const t_Temperature, double const t_J, SurfacePosition const t_Position );
    void initializeStateVariables();

    double m_SolarAbsorptance;
    double m_Conductivity;
  };

}

#endif