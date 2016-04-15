#include "BSDFThetaLimits.hpp"

using namespace std;

namespace LayerOptics {

  CThetaLimits::CThetaLimits( const vector< double >& t_ThetaAngles ) {
    if( t_ThetaAngles.size() == 0 ) {
      throw runtime_error("Error in definition of theta angles. Cannot form theta definitions.");
    }
    m_ThetaLimits = make_shared< vector< double > >();
    createLimits( t_ThetaAngles );
  }

  shared_ptr< vector< double > > CThetaLimits::getThetaLimits() const {
    return m_ThetaLimits;
  }

  void CThetaLimits::createLimits( const vector< double >& t_ThetaAngles ) {
    vector< double >::const_reverse_iterator it;
    double previousAngle = 90;
    m_ThetaLimits->push_back( previousAngle );

    for( it = t_ThetaAngles.rbegin(); it < t_ThetaAngles.rend(); ++it ) {
      double currentAngle = ( *it );
      double delta = 2 * ( previousAngle - currentAngle );
      double limit = previousAngle - delta;
      if( limit < 0 ) {
        limit = 0;
      }
      m_ThetaLimits->insert( m_ThetaLimits->begin(), limit );
      previousAngle = limit;
    }
  }
}