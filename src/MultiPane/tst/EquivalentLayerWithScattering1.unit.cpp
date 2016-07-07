#include <memory>
#include <gtest/gtest.h>

#include "EquivalentLayer.hpp"
#include "OpticalSurface.hpp"
#include "OpticalLayer.hpp"
#include "FenestrationCommon.hpp"

using namespace std;
using namespace LayerOptics;
using namespace MultiPane;
using namespace FenestrationCommon;

// Test equivalent properties of double layer with direct-direct, direct-diffuse and diffuse-diffuse components
// Tests include adding layer on back and front sides
class TestEquivalentLayerWithScattering1 : public testing::Test {

private:
  // Additional layer added to the back side
  shared_ptr< CEquivalentLayer > m_DoubleBack;

  // Additional layer added to the front side
  shared_ptr< CEquivalentLayer > m_DoubleFront;

protected:
  virtual void SetUp() {
    shared_ptr< CScatteringSurface > f1 = make_shared< CScatteringSurface >( 0.08, 0.05, 0.46, 0.23, 0.46, 0.52 );
    shared_ptr< CScatteringSurface > b1 = make_shared< CScatteringSurface >( 0.13, 0.25, 0.38, 0.19, 0.64, 0.22 );
    shared_ptr< CLayer > aLayer1 = make_shared< CLayer >( f1, b1 );
    
    shared_ptr< CScatteringSurface > f2 = make_shared< CScatteringSurface >( 0.1, 0.05, 0.48, 0.26, 0.56, 0.34 );
    shared_ptr< CScatteringSurface > b2 = make_shared< CScatteringSurface >( 0.15, 0.0, 0.38, 0.19, 0.49, 0.39 );
    shared_ptr< CLayer > aLayer2 = make_shared< CLayer >( f2, b2 );
    
    m_DoubleBack = make_shared< CEquivalentLayer >( aLayer1 );
    m_DoubleBack->addLayer( aLayer2, Side::Back );

    m_DoubleFront = make_shared< CEquivalentLayer >( aLayer1 );
    m_DoubleFront->addLayer( aLayer2, Side::Front );
  
  }

public:
  shared_ptr< CEquivalentLayer > getDoubleBack() { return m_DoubleBack; };
  shared_ptr< CEquivalentLayer > getDoubleFront() { return m_DoubleFront; };

};

TEST_F( TestEquivalentLayerWithScattering1, TestLayerAtBackSide ) {
  SCOPED_TRACE( "Begin Test: Equivalent layer transmittance and reflectances (direct-direct, direct-diffuse and diffuse-diffuse" );
  
  shared_ptr< CEquivalentLayer > doubleLayer = getDoubleBack();

  ///////////////////////////////////////////////
  // Direct-Direct
  ///////////////////////////////////////////////
  double Tf = doubleLayer->getPropertySimple( PropertySimple::T, Side::Front, Scattering::DirectDirect );
  EXPECT_NEAR( 0.008101266, Tf, 1e-6 );

  double Rf = doubleLayer->getPropertySimple( PropertySimple::R, Side::Front, Scattering::DirectDirect );
  EXPECT_NEAR( 0.050526582, Rf, 1e-6 );

  double Tb = doubleLayer->getPropertySimple( PropertySimple::T, Side::Back, Scattering::DirectDirect );
  EXPECT_NEAR( 0.019746835, Tb, 1e-6 );

  double Rb = doubleLayer->getPropertySimple( PropertySimple::R, Side::Back, Scattering::DirectDirect );
  EXPECT_NEAR( 0.003797468, Rb, 1e-6 );

  ///////////////////////////////////////////////
  // Diffuse-Diffuse
  ///////////////////////////////////////////////
  Tf = doubleLayer->getPropertySimple( PropertySimple::T, Side::Front, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.278426286, Tf, 1e-6 );

  Rf = doubleLayer->getPropertySimple( PropertySimple::R, Side::Front, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.6281885, Rf, 1e-6 );

  Tb = doubleLayer->getPropertySimple( PropertySimple::T, Side::Back, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.33895374, Tb, 1e-6 );

  Rb = doubleLayer->getPropertySimple( PropertySimple::R, Side::Back, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.455248595, Rb, 1e-6 );

  ///////////////////////////////////////////////
  // Direct-Diffuse
  ///////////////////////////////////////////////
  Tf = doubleLayer->getPropertySimple( PropertySimple::T, Side::Front, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.32058299, Tf, 1e-6 );

  Rf = doubleLayer->getPropertySimple( PropertySimple::R, Side::Front, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.354479119, Rf, 1e-6 );

  Tb = doubleLayer->getPropertySimple( PropertySimple::T, Side::Back, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.334201295, Tb, 1e-6 );

  Rb = doubleLayer->getPropertySimple( PropertySimple::R, Side::Back, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.27761223, Rb, 1e-6 );
}

TEST_F( TestEquivalentLayerWithScattering1, TestLayerAtFrontSide ) {
  SCOPED_TRACE( "Begin Test: Equivalent layer transmittance and reflectances (direct-direct, direct-diffuse and diffuse-diffuse" );

  shared_ptr< CEquivalentLayer > doubleLayer = getDoubleFront();

  ///////////////////////////////////////////////
  // Direct-Direct
  ///////////////////////////////////////////////
  double Tf = doubleLayer->getPropertySimple( PropertySimple::T, Side::Front, Scattering::DirectDirect );
  EXPECT_NEAR( 0.008, Tf, 1e-6 );

  double Rf = doubleLayer->getPropertySimple( PropertySimple::R, Side::Front, Scattering::DirectDirect );
  EXPECT_NEAR( 0.05075, Rf, 1e-6 );

  double Tb = doubleLayer->getPropertySimple( PropertySimple::T, Side::Back, Scattering::DirectDirect );
  EXPECT_NEAR( 0.0195, Tb, 1e-6 );

  double Rb = doubleLayer->getPropertySimple( PropertySimple::R, Side::Back, Scattering::DirectDirect );
  EXPECT_NEAR( 0.25, Rb, 1e-6 );

  ///////////////////////////////////////////////
  // Diffuse-Diffuse
  ///////////////////////////////////////////////
  Tf = doubleLayer->getPropertySimple( PropertySimple::T, Side::Front, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.323130958, Tf, 1e-6 );

  Rf = doubleLayer->getPropertySimple( PropertySimple::R, Side::Front, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.518986453, Rf, 1e-6 );

  Tb = doubleLayer->getPropertySimple( PropertySimple::T, Side::Back, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.393376819, Tb, 1e-6 );

  Rb = doubleLayer->getPropertySimple( PropertySimple::R, Side::Back, Scattering::DiffuseDiffuse );
  EXPECT_NEAR( 0.364024084, Rb, 1e-6 );

  ///////////////////////////////////////////////
  // Direct-Diffuse
  ///////////////////////////////////////////////
  Tf = doubleLayer->getPropertySimple( PropertySimple::T, Side::Front, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.328693427, Tf, 1e-6 );

  Rf = doubleLayer->getPropertySimple( PropertySimple::R, Side::Front, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.429757577, Rf, 1e-6 );

  Tb = doubleLayer->getPropertySimple( PropertySimple::T, Side::Back, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.290862067, Tb, 1e-6 );

  Rb = doubleLayer->getPropertySimple( PropertySimple::R, Side::Back, Scattering::DirectDiffuse );
  EXPECT_NEAR( 0.289766683, Rb, 1e-6 );
}