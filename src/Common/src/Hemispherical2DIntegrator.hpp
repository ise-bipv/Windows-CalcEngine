#ifndef HEMISPHERICAL2DINTEGRATOR_H
#define HEMISPHERICAL2DINTEGRATOR_H

#include< memory >

namespace FenestrationCommon {

  class CSeries;
  enum class IntegrationType;

  // Used to calculated hemispherical values in 2D. If for example some optical property is calculated 
  // for different incident angles, then this integrator will calculate hemispherical to hemispherical
  // value
  class CHemispherical2DIntegrator {
  public:
    CHemispherical2DIntegrator( std::shared_ptr< CSeries > t_AngularProperties, 
      const IntegrationType t_IntegrationType );

    double value() const;

  private: 
    double m_Value;
  
  };

}

#endif