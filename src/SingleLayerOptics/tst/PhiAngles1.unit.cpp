#include <memory>
#include <gtest/gtest.h>

#include <memory>

#include "WCESingleLayerOptics.hpp"

using namespace std;
using namespace SingleLayerOptics;

class TestPhisAngles1 : public testing::Test {

private:
	std::shared_ptr< CBSDFPhiAngles > m_BasisRing;

protected:
	virtual void SetUp() {
		m_BasisRing = make_shared< CBSDFPhiAngles >( 8 );
	}

public:
	std::shared_ptr< CBSDFPhiAngles > GetRing() {
		return m_BasisRing;
	};

};

TEST_F( TestPhisAngles1, TestBSDFRingCreation ) {
	SCOPED_TRACE( "Begin Test: Phi angles creation." );

	std::shared_ptr< CBSDFPhiAngles > aRing = GetRing();

	vector< double > results = *( aRing->phiAngles() );

	vector< double > correctResults = { 0, 45, 90, 135, 180, 225, 270, 315 };
	EXPECT_EQ( results.size(), correctResults.size() );

	for ( size_t i = 0; i < results.size(); ++i ) {
		EXPECT_NEAR( results[i], correctResults[i], 1e-6 );
	}

}
