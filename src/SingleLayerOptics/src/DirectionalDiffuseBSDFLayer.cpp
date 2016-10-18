#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

#include "DirectionalDiffuseBSDFLayer.hpp"
#include "DirectionalDiffuseCell.hpp"
#include "BSDFResults.hpp"
#include "SquareMatrix.hpp"
#include "BSDFDirections.hpp"
#include "BeamDirection.hpp"
#include "BSDFPatch.hpp"
#include "FenestrationCommon.hpp"

using namespace std;
using namespace FenestrationCommon;

namespace SingleLayerOptics {

  CDirectionalDiffuseBSDFLayer::CDirectionalDiffuseBSDFLayer( const shared_ptr< CDirectionalDiffuseCell >& t_Cell, 
    const shared_ptr< const CBSDFHemisphere >& t_Hemisphere ) :
    CBSDFLayer( t_Cell, t_Hemisphere ) {
    
  }

  shared_ptr< CDirectionalDiffuseCell > CDirectionalDiffuseBSDFLayer::cellAsDirectionalDiffuse() const {
    shared_ptr< CDirectionalDiffuseCell > aCell = dynamic_pointer_cast< CDirectionalDiffuseCell >( m_Cell );
    assert( aCell != nullptr );
    return aCell;
  }

  void CDirectionalDiffuseBSDFLayer::calcDiffuseDistribution( const Side aSide, 
    const CBeamDirection& t_Direction,
    const size_t t_DirectionIndex ) {

    shared_ptr< CDirectionalDiffuseCell > aCell = cellAsDirectionalDiffuse();
    
    shared_ptr< CSquareMatrix > Tau = m_Results->Tau( aSide );
    shared_ptr< CSquareMatrix > Rho = m_Results->Rho( aSide );

    shared_ptr< CBSDFDirections > jDirections = m_BSDFHemisphere->getDirections( BSDFHemisphere::Outgoing );

    size_t size = jDirections->size();

    for( size_t j = 0; j < size; ++j ) {

      const CBeamDirection jDirection = *( *jDirections )[ j ]->centerPoint();

      double aTau = aCell->T_dir_dif( aSide, t_Direction, jDirection );
      double Ref = aCell->R_dir_dif( aSide, t_Direction, jDirection );

      //( *Tau )[ t_DirectionIndex ][ j ] += aTau / M_PI;
      //( *Rho )[ t_DirectionIndex ][ j ] += Ref / M_PI;
      ( *Tau )[ j ][ t_DirectionIndex ] += aTau / M_PI;
      ( *Rho )[ j ][ t_DirectionIndex ] += Ref / M_PI;
    }

  }

  void CDirectionalDiffuseBSDFLayer::calcDiffuseDistribution_wv( const Side aSide, 
    const CBeamDirection& t_Direction,
    const size_t t_DirectionIndex ) {

    shared_ptr< CDirectionalDiffuseCell > aCell = cellAsDirectionalDiffuse();

    CBSDFDirections iDirections = *m_BSDFHemisphere->getDirections( BSDFHemisphere::Outgoing );

    size_t size = iDirections.size();

    for( size_t i = 0; i < size; ++i ) {

      const CBeamDirection iDirection = *iDirections[ i ]->centerPoint();

      shared_ptr< vector< double > > aTau = aCell->T_dir_dif_band( aSide, t_Direction, iDirection );
      shared_ptr< vector< double > > Ref = aCell->R_dir_dif_band( aSide, t_Direction, iDirection );

      size_t numWV = aTau->size();
      for( size_t j = 0; j < numWV; ++j ) {
        shared_ptr< CBSDFResults > aResults = nullptr;
        aResults = ( *m_WVResults )[ j ];
        assert( aResults != nullptr );
        shared_ptr< CSquareMatrix > Tau = aResults->Tau( aSide );
        shared_ptr< CSquareMatrix > Rho = aResults->Rho( aSide );
        //( *Tau )[ t_DirectionIndex ][ i ] += ( *aTau )[ j ] / M_PI;
        //( *Rho )[ t_DirectionIndex ][ i ] += ( *Ref )[ j ] / M_PI;
        ( *Tau )[ i ][ t_DirectionIndex ] += ( *aTau )[ j ] / M_PI;
        ( *Rho )[ i ][ t_DirectionIndex ] += ( *Ref )[ j ] / M_PI;
      }
      
    }

  }

}