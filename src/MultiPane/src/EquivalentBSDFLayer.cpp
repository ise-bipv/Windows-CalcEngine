#include <algorithm>
#include <iterator>
#include <assert.h>
#include <stdexcept>


#include "EquivalentBSDFLayer.hpp"
#include "EquivalentBSDFLayerSingleBand.hpp"
#include "BSDFLayer.hpp"
#include "SpecularBSDFLayer.hpp"
#include "Series.hpp"
#include "IntegratorStrategy.hpp"
#include "BSDFResults.hpp"
#include "SquareMatrix.hpp"
#include "BSDFDirections.hpp"
#include "FenestrationCommon.hpp"

using namespace std;
using namespace FenestrationCommon;
using namespace LayerOptics;

namespace MultiPane {

  CEquivalentBSDFLayer::CEquivalentBSDFLayer( const shared_ptr< vector< double > >& t_CommonWavelengths,
    const shared_ptr< CSeries >& t_SolarRadiation, const shared_ptr< CBSDFLayer >& t_Layer ) : 
    m_SolarRadiation( t_SolarRadiation ), m_CombinedLayerWavelengths( t_CommonWavelengths ), m_Calculated( false ) {
    if( t_Layer == nullptr ) {
      throw runtime_error("Equivalent BSDF Layer must contain valid layer.");
    }

    m_LayersWL = make_shared< vector< shared_ptr< CEquivalentBSDFLayerSingleBand > > >();

    // Lambda matrix from spectral results. Same lambda is valid for any wavelength
    m_Lambda = t_Layer->getResults()->lambdaMatrix();

    shared_ptr< vector< shared_ptr < CBSDFResults > > > aResults = nullptr;

    aResults = t_Layer->getWavelengthResults();
    size_t size = m_CombinedLayerWavelengths->size();
    for( size_t i = 0; i < size; ++i ) {
      double curWL = ( *m_CombinedLayerWavelengths )[ i ];
      int index = t_Layer->getBandIndex( curWL );
      assert( index > -1 );

      shared_ptr< CBSDFResults > currentLayer = ( *aResults )[ size_t( index ) ];
      shared_ptr< CEquivalentBSDFLayerSingleBand > aEquivalentLayer = 
        make_shared< CEquivalentBSDFLayerSingleBand >( currentLayer );

      m_LayersWL->push_back( aEquivalentLayer );

    }

    m_Results = make_shared< CBSDFResults >( t_Layer->m_BSDFHemisphere->getDirections( BSDFHemisphere::Incoming ) );

  }

  void CEquivalentBSDFLayer::addLayer( const shared_ptr< CBSDFLayer >& t_Layer ) {

    shared_ptr< vector< shared_ptr < CBSDFResults > > > aResults = nullptr;

    aResults = t_Layer->getWavelengthResults();
    size_t size = m_CombinedLayerWavelengths->size();
    for( size_t i = 0; i < size; ++i ) {
      double curWL = ( *m_CombinedLayerWavelengths )[ i ];
      int index = t_Layer->getBandIndex( curWL );
      assert( index > -1 );
      shared_ptr< CBSDFResults > currentLayer = ( *aResults )[ size_t( index ) ];
      shared_ptr< CEquivalentBSDFLayerSingleBand > currentEqLayer = ( *m_LayersWL )[ i ];
      currentEqLayer->addLayer( currentLayer );
    }

  }

  shared_ptr< CSquareMatrix > CEquivalentBSDFLayer::Tau( const double minLambda, 
    const double maxLambda, Side t_Side ) {
    if( !m_Calculated ) {
      calculate( minLambda, maxLambda );
    }

    return m_Results->Tau( t_Side );
  }

  shared_ptr< CSquareMatrix > CEquivalentBSDFLayer::Rho( const double minLambda, 
    const double maxLambda, Side t_Side ) {
    if( !m_Calculated ) {
      calculate( minLambda, maxLambda );
    }

    return m_Results->Rho( t_Side );
  }

  shared_ptr< vector< double > > CEquivalentBSDFLayer::Abs( const double minLambda, const double maxLambda, 
    const Side t_Side, const size_t Index ) {
    if( !m_Calculated ) {
      calculate( minLambda, maxLambda );
    }
    return ( *m_Abs.at( t_Side ) )[ Index - 1 ];
  }

  shared_ptr< vector< double > > CEquivalentBSDFLayer::TauHem( const double minLambda, const double maxLambda, 
    const Side t_Side ) {
    if( !m_Calculated ) {
      calculate( minLambda, maxLambda );
    }
    return m_Results->TauHem( t_Side );
  }

  shared_ptr< vector< double > > CEquivalentBSDFLayer::RhoHem( const double minLambda, const double maxLambda, 
    const Side t_Side ) {
    if( !m_Calculated ) {
      calculate( minLambda, maxLambda );
    }
    return m_Results->RhoHem( t_Side );
  }

  double CEquivalentBSDFLayer::TauHem( const double minLambda, const double maxLambda, 
    const Side t_Side, const double t_Theta, const double t_Phi ) {
    auto aIndex = m_Results->getDirections()->getNearestBeamIndex( t_Theta, t_Phi );
    return ( *TauHem( minLambda, maxLambda, t_Side ) )[ aIndex ];
  }

  double CEquivalentBSDFLayer::RhoHem( const double minLambda, const double maxLambda, 
    const Side t_Side, const double t_Theta, const double t_Phi ) {
    auto aIndex = m_Results->getDirections()->getNearestBeamIndex( t_Theta, t_Phi );
    return ( *RhoHem( minLambda, maxLambda, t_Side ) )[ aIndex ];
  }

  double CEquivalentBSDFLayer::Abs( const double minLambda, const double maxLambda, 
    const Side t_Side, const size_t Index, const double t_Theta, const double t_Phi ) {
    auto aIndex = m_Results->getDirections()->getNearestBeamIndex( t_Theta, t_Phi );
    return ( *Abs( minLambda, maxLambda, t_Side, Index ) )[ aIndex ];
  }

  void CEquivalentBSDFLayer::calculate( const double minLambda, const double maxLambda ) {
    size_t matrixSize = m_Lambda->getSize();

    map< Side, shared_ptr< CSquareMatrix > > aTau;
    map< Side, shared_ptr< CSquareMatrix > > aRho;

    aTau[ Side::Front ] = make_shared< CSquareMatrix >( matrixSize );
    aTau[ Side::Back ] = make_shared< CSquareMatrix >( matrixSize );
    aRho[ Side::Front ] = make_shared< CSquareMatrix >( matrixSize );
    aRho[ Side::Back ] = make_shared< CSquareMatrix >( matrixSize );

    size_t numberOfLayers = ( *m_LayersWL )[ 0 ]->getNumberOfLayers();

    m_Abs[ Side::Front ] = make_shared< vector< shared_ptr< vector< double > > > >( numberOfLayers );
    m_Abs[ Side::Back ] = make_shared< vector< shared_ptr< vector< double > > > >( numberOfLayers );

    for( size_t i = 0; i < numberOfLayers; ++i ) {
      for(Side t_Side : EnumSide()) {
        ( *m_Abs.at( t_Side ) )[ i ] = make_shared< vector< double > >( matrixSize );
      }
    }

    shared_ptr< CSeries > iTotalSolar = m_SolarRadiation->integrate( IntegrationType::Trapezoidal );
    double incomingSolar = iTotalSolar->sum( minLambda, maxLambda );

    shared_ptr< CSeries > interpolatedSolar = m_SolarRadiation->interpolate( *m_CombinedLayerWavelengths );

    size_t size = m_CombinedLayerWavelengths->size();

    // Total matrices for every property
    map< Side, vector< vector< shared_ptr< CSeries > > > > aTotalT;
    map< Side, vector< vector< shared_ptr< CSeries > > > > aTotalR;

    map< Side, vector< vector< shared_ptr< CSeries > > > > aTotalA;
    for(Side t_Side : EnumSide()) {
      aTotalA[ t_Side ] = vector< vector< shared_ptr< CSeries > > >( numberOfLayers );
    }
    for( size_t i = 0; i < numberOfLayers; ++i ) {
      for(Side t_Side : EnumSide()) {
        aTotalA.at( t_Side )[ i ].resize( matrixSize );
      }
    }

    for(Side t_Side : EnumSide()) {
      aTotalT[ t_Side ] = vector< vector< shared_ptr< CSeries > > >( matrixSize );
      aTotalR[ t_Side ] = vector< vector< shared_ptr< CSeries > > >( matrixSize );
    }

    for( size_t i = 0; i < matrixSize; ++i ) {
      for(Side t_Side : EnumSide()) {
        aTotalT.at( t_Side )[ i ].resize( matrixSize );
        aTotalR.at( t_Side )[ i ].resize( matrixSize );
      }
    }

    // Calculate total transmitted solar per matrix and perform integration
    for( size_t i = 0; i < size; ++i ) {
      // First need to select correct side
      double curWL = ( *m_CombinedLayerWavelengths )[ i ];
      shared_ptr< CEquivalentBSDFLayerSingleBand > curLayer = ( *m_LayersWL )[ i ];

      for( size_t j = 0; j < matrixSize; ++j ) {
        for( size_t k = 0; k < numberOfLayers; ++k ) {
          if( i == 0 ) {
            for(Side t_Side : EnumSide()) {
              aTotalA.at( t_Side )[ k ][ j ] = make_shared< CSeries >();
            }
          }
          for(Side t_Side : EnumSide()) {
            aTotalA.at( t_Side )[ k ][ j ]->addProperty( curWL, ( *curLayer->getLayerAbsorptances( k + 1, t_Side) )[ j ] );
          }
        }
        
        for( size_t k = 0; k < matrixSize; ++k ) {
          if( i == 0 ) {
            for(Side t_Side : EnumSide()) {
              aTotalT.at( t_Side )[ j ][ k ] = make_shared< CSeries >();
              aTotalR.at( t_Side )[ j ][ k ] = make_shared< CSeries >();
            }
          }
          
          for(Side t_Side : EnumSide()) {
            aTotalT.at( t_Side )[ j ][ k ]->addProperty( curWL, ( *curLayer->Tau( t_Side ) )[ j ][ k ] );
            aTotalR.at( t_Side )[ j ][ k ]->addProperty( curWL, ( *curLayer->Rho( t_Side ) )[ j ][ k ] );
          }
        }
      }
    }

    for( size_t j = 0; j < matrixSize; ++j ) {
      for( size_t k = 0; k < numberOfLayers; ++k ) {
        for(Side t_Side : EnumSide()) {
          aTotalA.at( t_Side )[ k ][ j ] = aTotalA.at( t_Side )[ k ][ j ]->mMult( interpolatedSolar );
          aTotalA.at( t_Side )[ k ][ j ] = aTotalA.at( t_Side )[ k ][ j ]->integrate( IntegrationType::Trapezoidal );
          ( *( *m_Abs.at( t_Side ) )[ k ] )[ j ] = aTotalA.at( t_Side )[ k ][ j ]->sum( minLambda, maxLambda );
          ( *( *m_Abs.at( t_Side ) )[ k ] )[ j ] = ( *( *m_Abs.at( t_Side ) )[ k ] )[ j ] / incomingSolar;
        }
      }

      for( size_t k = 0; k < matrixSize; ++k ) {
        // Transmittance
        for(Side t_Side : EnumSide()) {
          // Transmittance
          aTotalT.at( t_Side )[ j ][ k ] = aTotalT.at( t_Side )[ j ][ k ]->mMult( interpolatedSolar );
          aTotalT.at( t_Side )[ j ][ k ] = aTotalT.at( t_Side )[ j ][ k ]->integrate( IntegrationType::Trapezoidal );
          ( *aTau.at( t_Side ) )[ j ][ k ] = aTotalT.at( t_Side )[ j ][ k ]->sum( minLambda, maxLambda );
          ( *aTau.at( t_Side ) )[ j ][ k ] = ( *aTau.at( t_Side ) )[ j ][ k ] / incomingSolar;

          // Reflectance
          aTotalR.at( t_Side )[ j ][ k ] = aTotalR.at( t_Side )[ j ][ k ]->mMult( interpolatedSolar );
          aTotalR.at( t_Side )[ j ][ k ] = aTotalR.at( t_Side )[ j ][ k ]->integrate( IntegrationType::Trapezoidal );
          ( *aRho.at( t_Side ) )[ j ][ k ] = aTotalR.at( t_Side )[ j ][ k ]->sum( minLambda, maxLambda );
          ( *aRho.at( t_Side ) )[ j ][ k ] = ( *aRho.at( t_Side ) )[ j ][ k ] / incomingSolar;
        }
      }
    }

    for( Side t_Side : EnumSide() ) {
      m_Results->setResultMatrices( aTau.at( t_Side ), aRho.at( t_Side ), t_Side );
    }

    m_Calculated = true;

  }

}