#ifndef GEOMETRY2D_H
#define GEOMETRY2D_H

#include <vector>
#include <memory>

namespace FenestrationCommon {

  class CSquareMatrix;

}

namespace Viewer {

  class CViewSegment2D;
  class CPoint2D;
  class CSegment2D;

  class CGeometry2D {
  public:
    CGeometry2D();

    void appendSegment( const std::shared_ptr< CViewSegment2D >& t_Segment );
    void appendGeometry2D( const std::shared_ptr< CGeometry2D >& t_Geometry2D );
    std::shared_ptr< FenestrationCommon::CSquareMatrix > viewFactors();

    // Shifts all segments for given coordinates
    std::shared_ptr< CGeometry2D > Translate( const double t_x, const double t_y );

    // First and last point function are necessary to make connection between different enclosures
    std::shared_ptr< const CPoint2D > firstPoint() const;
    std::shared_ptr< const CPoint2D > lastPoint() const;
    
    // Entry point of the enclosure. Important since enclosure can be entered in reverse way.
    std::shared_ptr< const CPoint2D > entryPoint() const;

    // Exit point of the enclosure. Important since enclosure can be entered in reverse way.
    std::shared_ptr< const CPoint2D > exitPoint() const;

    std::shared_ptr< std::vector< std::shared_ptr< CViewSegment2D > > > segments() const;

  private:
    // Determines if point is laying inside rectangle made by two segments. That would mean that 
    // point is in blocking view of two segments
    bool pointInSegmentsView( const std::shared_ptr< const CViewSegment2D >& t_Segment1, 
      const std::shared_ptr< const CViewSegment2D >& t_Segment2, 
      const std::shared_ptr< const CPoint2D >& t_Point ) const;

    // Calculates third surface blocking.
    bool thirdSurfaceShadowing( const std::shared_ptr< const CViewSegment2D >& t_Segment1, 
      const std::shared_ptr< const CViewSegment2D >& t_Segment2 ) const;

    // Simplified version for fast calculation of third surface shadowing. Used for subsurfaces
    bool thirdSurfaceShadowingSimple( const std::shared_ptr< const CViewSegment2D >& t_Segment1, 
      const std::shared_ptr< const CViewSegment2D >& t_Segment2 ) const;

    // Calculate view factor between two segments. This routine will check third surface shadowing as well. 
    // It will divide segments into subsurfaces by default
    double viewFactorCoeff( const std::shared_ptr< const CViewSegment2D >& t_Segment1,
      const std::shared_ptr< const CViewSegment2D >& t_Segment2 ) const;

    // returns Y coordinate of intersection with Y axis for line going through t_Point and for given
    // tangens of profile angle
    double intersectionWithYAxis( const double tanPhi, const CPoint2D& t_Point ) const;

    // Checks if view factors are valid for current geometry
    void checkViewFactors();

    std::shared_ptr< std::vector< std::shared_ptr< CViewSegment2D > > > m_Segments;

    // Holds state for the view factors. No need to recalculate them every time since it is
    // time consuming operation.
    std::shared_ptr< FenestrationCommon::CSquareMatrix > m_ViewFactors;
    bool m_ViewFactorsCalculated;

  };

}

#endif