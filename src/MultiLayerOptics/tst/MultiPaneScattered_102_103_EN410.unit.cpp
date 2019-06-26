#include <memory>
#include <gtest/gtest.h>

#include "WCESpectralAveraging.hpp"
#include "WCEMultiLayerOptics.hpp"
#include "WCESingleLayerOptics.hpp"
#include "WCECommon.hpp"


using namespace SingleLayerOptics;
using namespace FenestrationCommon;
using namespace SpectralAveraging;
using namespace MultiLayerOptics;

// Example on how to create scattered multilayer.

class MultiPaneScattered_102_103_EN410 : public testing::Test {
private:
    std::unique_ptr<CMultiLayerScattered> m_Layer;

    std::shared_ptr<CSeries> loadSolarRadiationFile() {
        // Full prEN 410 Table 2 AM1_0 (Solar radiation)
        auto aSolarRadiation = CSeries::create(
                {{0.300, 0.0005},
                 {0.320, 0.0069},
                 {0.340, 0.0122},
                 {0.360, 0.0145},
                 {0.380, 0.0177},
                 {0.400, 0.0235},
                 {0.420, 0.0268},
                 {0.440, 0.0294},
                 {0.460, 0.0343},
                 {0.480, 0.0339},
                 {0.500, 0.0326},
                 {0.520, 0.0318},
                 {0.540, 0.0321},
                 {0.560, 0.0312},
                 {0.580, 0.0294},
                 {0.600, 0.0289},
                 {0.620, 0.0289},
                 {0.640, 0.0280},
                 {0.660, 0.0273},
                 {0.680, 0.0246},
                 {0.700, 0.0237},
                 {0.720, 0.0220},
                 {0.740, 0.0230},
                 {0.760, 0.0199},
                 {0.780, 0.0211},
                 {0.800, 0.0330},
                 {0.850, 0.0453},
                 {0.900, 0.0381},
                 {0.950, 0.0220},
                 {1.000, 0.0329},
                 {1.050, 0.0306},
                 {1.100, 0.0185},
                 {1.150, 0.0136},
                 {1.200, 0.0210},
                 {1.250, 0.0211},
                 {1.300, 0.0166},
                 {1.350, 0.0042},
                 {1.400, 0.0010},
                 {1.450, 0.0044},
                 {1.500, 0.0095},
                 {1.550, 0.0123},
                 {1.600, 0.0110},
                 {1.650, 0.0106},
                 {1.700, 0.0093},
                 {1.750, 0.0068},
                 {1.800, 0.0024},
                 {1.850, 0.0005},
                 {1.900, 0.0002},
                 {1.950, 0.0012},
                 {2.000, 0.0030},
                 {2.050, 0.0037},
                 {2.100, 0.0057},
                 {2.200, 0.0066},
                 {2.300, 0.0060},
                 {2.400, 0.0041},
                 {2.500, 0.0006}

                });

        return aSolarRadiation;
    }

    std::shared_ptr<CSpectralSampleData> loadSampleData_NFRC_102() {
        auto aMeasurements_102 = CSpectralSampleData::create(
                {{0.300, 0.0020, 0.0470, 0.0480},
                 {0.305, 0.0030, 0.0470, 0.0480},
                 {0.310, 0.0090, 0.0470, 0.0480},
                 {0.315, 0.0350, 0.0470, 0.0480},
                 {0.320, 0.1000, 0.0470, 0.0480},
                 {0.325, 0.2180, 0.0490, 0.0500},
                 {0.330, 0.3560, 0.0530, 0.0540},
                 {0.335, 0.4980, 0.0600, 0.0610},
                 {0.340, 0.6160, 0.0670, 0.0670},
                 {0.345, 0.7090, 0.0730, 0.0740},
                 {0.350, 0.7740, 0.0780, 0.0790},
                 {0.355, 0.8180, 0.0820, 0.0820},
                 {0.360, 0.8470, 0.0840, 0.0840},
                 {0.365, 0.8630, 0.0850, 0.0850},
                 {0.370, 0.8690, 0.0850, 0.0860},
                 {0.375, 0.8610, 0.0850, 0.0850},
                 {0.380, 0.8560, 0.0840, 0.0840},
                 {0.385, 0.8660, 0.0850, 0.0850},
                 {0.390, 0.8810, 0.0860, 0.0860},
                 {0.395, 0.8890, 0.0860, 0.0860},
                 {0.400, 0.8930, 0.0860, 0.0860},
                 {0.410, 0.8930, 0.0860, 0.0860},
                 {0.420, 0.8920, 0.0860, 0.0860},
                 {0.430, 0.8920, 0.0850, 0.0850},
                 {0.440, 0.8920, 0.0850, 0.0850},
                 {0.450, 0.8960, 0.0850, 0.0850},
                 {0.460, 0.9000, 0.0850, 0.0850},
                 {0.470, 0.9020, 0.0840, 0.0840},
                 {0.480, 0.9030, 0.0840, 0.0840},
                 {0.490, 0.9040, 0.0850, 0.0850},
                 {0.500, 0.9050, 0.0840, 0.0840},
                 {0.510, 0.9050, 0.0840, 0.0840},
                 {0.520, 0.9050, 0.0840, 0.0840},
                 {0.530, 0.9040, 0.0840, 0.0840},
                 {0.540, 0.9040, 0.0830, 0.0830},
                 {0.550, 0.9030, 0.0830, 0.0830},
                 {0.560, 0.9020, 0.0830, 0.0830},
                 {0.570, 0.9000, 0.0820, 0.0820},
                 {0.580, 0.8980, 0.0820, 0.0820},
                 {0.590, 0.8960, 0.0810, 0.0810},
                 {0.600, 0.8930, 0.0810, 0.0810},
                 {0.610, 0.8900, 0.0810, 0.0810},
                 {0.620, 0.8860, 0.0800, 0.0800},
                 {0.630, 0.8830, 0.0800, 0.0800},
                 {0.640, 0.8790, 0.0790, 0.0790},
                 {0.650, 0.8750, 0.0790, 0.0790},
                 {0.660, 0.8720, 0.0790, 0.0790},
                 {0.670, 0.8680, 0.0780, 0.0780},
                 {0.680, 0.8630, 0.0780, 0.0780},
                 {0.690, 0.8590, 0.0770, 0.0770},
                 {0.700, 0.8540, 0.0760, 0.0770},
                 {0.710, 0.8500, 0.0760, 0.0760},
                 {0.720, 0.8450, 0.0750, 0.0760},
                 {0.730, 0.8400, 0.0750, 0.0750},
                 {0.740, 0.8350, 0.0750, 0.0750},
                 {0.750, 0.8310, 0.0740, 0.0740},
                 {0.760, 0.8260, 0.0740, 0.0740},
                 {0.770, 0.8210, 0.0740, 0.0740},
                 {0.780, 0.8160, 0.0730, 0.0730},
                 {0.790, 0.8120, 0.0730, 0.0730},
                 {0.800, 0.8080, 0.0720, 0.0720},
                 {0.810, 0.8030, 0.0720, 0.0720},
                 {0.820, 0.8000, 0.0720, 0.0720},
                 {0.830, 0.7960, 0.0710, 0.0710},
                 {0.840, 0.7930, 0.0700, 0.0710},
                 {0.850, 0.7880, 0.0700, 0.0710},
                 {0.860, 0.7860, 0.0700, 0.0700},
                 {0.870, 0.7820, 0.0740, 0.0740},
                 {0.880, 0.7800, 0.0720, 0.0720},
                 {0.890, 0.7770, 0.0730, 0.0740},
                 {0.900, 0.7760, 0.0720, 0.0720},
                 {0.910, 0.7730, 0.0720, 0.0720},
                 {0.920, 0.7710, 0.0710, 0.0710},
                 {0.930, 0.7700, 0.0700, 0.0700},
                 {0.940, 0.7680, 0.0690, 0.0690},
                 {0.950, 0.7660, 0.0680, 0.0680},
                 {0.960, 0.7660, 0.0670, 0.0680},
                 {0.970, 0.7640, 0.0680, 0.0680},
                 {0.980, 0.7630, 0.0680, 0.0680},
                 {0.990, 0.7620, 0.0670, 0.0670},
                 {1.000, 0.7620, 0.0660, 0.0670},
                 {1.050, 0.7600, 0.0660, 0.0660},
                 {1.100, 0.7590, 0.0660, 0.0660},
                 {1.150, 0.7610, 0.0660, 0.0660},
                 {1.200, 0.7650, 0.0660, 0.0660},
                 {1.250, 0.7700, 0.0650, 0.0650},
                 {1.300, 0.7770, 0.0670, 0.0670},
                 {1.350, 0.7860, 0.0660, 0.0670},
                 {1.400, 0.7950, 0.0670, 0.0680},
                 {1.450, 0.8080, 0.0670, 0.0670},
                 {1.500, 0.8190, 0.0690, 0.0690},
                 {1.550, 0.8290, 0.0690, 0.0690},
                 {1.600, 0.8360, 0.0700, 0.0700},
                 {1.650, 0.8400, 0.0700, 0.0700},
                 {1.700, 0.8420, 0.0690, 0.0700},
                 {1.750, 0.8420, 0.0690, 0.0700},
                 {1.800, 0.8410, 0.0700, 0.0700},
                 {1.850, 0.8400, 0.0690, 0.0690},
                 {1.900, 0.8390, 0.0680, 0.0680},
                 {1.950, 0.8390, 0.0710, 0.0710},
                 {2.000, 0.8390, 0.0690, 0.0690},
                 {2.050, 0.8400, 0.0680, 0.0680},
                 {2.100, 0.8410, 0.0680, 0.0680},
                 {2.150, 0.8390, 0.0690, 0.0690},
                 {2.200, 0.8300, 0.0700, 0.0700},
                 {2.250, 0.8300, 0.0700, 0.0700},
                 {2.300, 0.8320, 0.0690, 0.0690},
                 {2.350, 0.8320, 0.0690, 0.0700},
                 {2.400, 0.8320, 0.0700, 0.0700},
                 {2.450, 0.8260, 0.0690, 0.0690},
                 {2.500, 0.8220, 0.0680, 0.0680}});

        return aMeasurements_102;
    }

    std::shared_ptr<CSpectralSampleData> loadSampleData_NFRC_103() {
        auto aMeasurements_103 = CSpectralSampleData::create(
                {{0.300, 0.0000, 0.0470, 0.0490},
                 {0.305, 0.0050, 0.0470, 0.0490},
                 {0.310, 0.0000, 0.0470, 0.0480},
                 {0.315, 0.0030, 0.0460, 0.0480},
                 {0.320, 0.0190, 0.0460, 0.0480},
                 {0.325, 0.0660, 0.0450, 0.0460},
                 {0.330, 0.1600, 0.0450, 0.0470},
                 {0.335, 0.2940, 0.0490, 0.0500},
                 {0.340, 0.4370, 0.0550, 0.0560},
                 {0.345, 0.5660, 0.0620, 0.0620},
                 {0.350, 0.6710, 0.0690, 0.0690},
                 {0.355, 0.7440, 0.0740, 0.0740},
                 {0.360, 0.7930, 0.0780, 0.0780},
                 {0.365, 0.8220, 0.0800, 0.0800},
                 {0.370, 0.8320, 0.0810, 0.0810},
                 {0.375, 0.8190, 0.0800, 0.0800},
                 {0.380, 0.8090, 0.0790, 0.0790},
                 {0.385, 0.8290, 0.0800, 0.0800},
                 {0.390, 0.8530, 0.0820, 0.0820},
                 {0.395, 0.8680, 0.0830, 0.0830},
                 {0.400, 0.8750, 0.0830, 0.0830},
                 {0.410, 0.8750, 0.0830, 0.0830},
                 {0.420, 0.8730, 0.0830, 0.0830},
                 {0.430, 0.8730, 0.0820, 0.0820},
                 {0.440, 0.8730, 0.0820, 0.0820},
                 {0.450, 0.8800, 0.0820, 0.0820},
                 {0.460, 0.8870, 0.0820, 0.0820},
                 {0.470, 0.8900, 0.0820, 0.0820},
                 {0.480, 0.8920, 0.0830, 0.0830},
                 {0.490, 0.8930, 0.0820, 0.0820},
                 {0.500, 0.8940, 0.0820, 0.0820},
                 {0.510, 0.8950, 0.0820, 0.0820},
                 {0.520, 0.8950, 0.0820, 0.0820},
                 {0.530, 0.8940, 0.0820, 0.0820},
                 {0.540, 0.8930, 0.0810, 0.0810},
                 {0.550, 0.8910, 0.0810, 0.0810},
                 {0.560, 0.8880, 0.0810, 0.0810},
                 {0.570, 0.8840, 0.0800, 0.0800},
                 {0.580, 0.8810, 0.0800, 0.0800},
                 {0.590, 0.8760, 0.0790, 0.0790},
                 {0.600, 0.8710, 0.0790, 0.0790},
                 {0.610, 0.8650, 0.0780, 0.0780},
                 {0.620, 0.8590, 0.0770, 0.0770},
                 {0.630, 0.8530, 0.0770, 0.0770},
                 {0.640, 0.8470, 0.0760, 0.0760},
                 {0.650, 0.8400, 0.0750, 0.0750},
                 {0.660, 0.8330, 0.0750, 0.0750},
                 {0.670, 0.8260, 0.0740, 0.0740},
                 {0.680, 0.8180, 0.0730, 0.0730},
                 {0.690, 0.8100, 0.0730, 0.0730},
                 {0.700, 0.8020, 0.0720, 0.0720},
                 {0.710, 0.7940, 0.0710, 0.0720},
                 {0.720, 0.7860, 0.0710, 0.0710},
                 {0.730, 0.7770, 0.0700, 0.0700},
                 {0.740, 0.7690, 0.0690, 0.0700},
                 {0.750, 0.7610, 0.0690, 0.0690},
                 {0.760, 0.7520, 0.0680, 0.0680},
                 {0.770, 0.7440, 0.0670, 0.0680},
                 {0.780, 0.7360, 0.0670, 0.0670},
                 {0.790, 0.7290, 0.0660, 0.0660},
                 {0.800, 0.7220, 0.0660, 0.0660},
                 {0.810, 0.7150, 0.0650, 0.0660},
                 {0.820, 0.7100, 0.0650, 0.0650},
                 {0.830, 0.7020, 0.0640, 0.0650},
                 {0.840, 0.6980, 0.0640, 0.0640},
                 {0.850, 0.6900, 0.0630, 0.0640},
                 {0.860, 0.6870, 0.0650, 0.0650},
                 {0.870, 0.6810, 0.0670, 0.0670},
                 {0.880, 0.6770, 0.0650, 0.0660},
                 {0.890, 0.6730, 0.0660, 0.0660},
                 {0.900, 0.6700, 0.0650, 0.0660},
                 {0.910, 0.6670, 0.0650, 0.0650},
                 {0.920, 0.6640, 0.0640, 0.0640},
                 {0.930, 0.6600, 0.0630, 0.0630},
                 {0.940, 0.6580, 0.0640, 0.0640},
                 {0.950, 0.6560, 0.0630, 0.0630},
                 {0.960, 0.6540, 0.0610, 0.0610},
                 {0.970, 0.6530, 0.0620, 0.0620},
                 {0.980, 0.6510, 0.0610, 0.0620},
                 {0.990, 0.6490, 0.0610, 0.0620},
                 {1.000, 0.6480, 0.0590, 0.0600},
                 {1.050, 0.6450, 0.0590, 0.0600},
                 {1.100, 0.6450, 0.0580, 0.0590},
                 {1.150, 0.6470, 0.0590, 0.0590},
                 {1.200, 0.6530, 0.0590, 0.0590},
                 {1.250, 0.6610, 0.0580, 0.0590},
                 {1.300, 0.6730, 0.0600, 0.0600},
                 {1.350, 0.6870, 0.0600, 0.0600},
                 {1.400, 0.7020, 0.0610, 0.0610},
                 {1.450, 0.7220, 0.0610, 0.0620},
                 {1.500, 0.7410, 0.0630, 0.0640},
                 {1.550, 0.7570, 0.0630, 0.0640},
                 {1.600, 0.7690, 0.0650, 0.0650},
                 {1.650, 0.7750, 0.0650, 0.0640},
                 {1.700, 0.7790, 0.0640, 0.0650},
                 {1.750, 0.7790, 0.0650, 0.0650},
                 {1.800, 0.7770, 0.0650, 0.0650},
                 {1.850, 0.7760, 0.0650, 0.0630},
                 {1.900, 0.7730, 0.0620, 0.0620},
                 {1.950, 0.7730, 0.0650, 0.0650},
                 {2.000, 0.7720, 0.0650, 0.0650},
                 {2.050, 0.7740, 0.0640, 0.0640},
                 {2.100, 0.7750, 0.0640, 0.0650},
                 {2.150, 0.7730, 0.0650, 0.0650},
                 {2.200, 0.7580, 0.0640, 0.0650},
                 {2.250, 0.7590, 0.0640, 0.0640},
                 {2.300, 0.7660, 0.0650, 0.0650},
                 {2.350, 0.7670, 0.0640, 0.0650},
                 {2.400, 0.7660, 0.0640, 0.0640},
                 {2.450, 0.7570, 0.0640, 0.0640},
                 {2.500, 0.7500, 0.0630, 0.0630}});

        return aMeasurements_103;
    }

protected:
    void SetUp() override {
        const auto solarRadiationSpectrum{loadSolarRadiationFile()};
        const std::vector<double> wavelengthSet = solarRadiationSpectrum->getXArray();

        // Create material from samples
        auto thickness = 3.048e-3;   // [m]
        auto aMaterial_102 = SingleLayerOptics::Material::nBandMaterial(
                loadSampleData_NFRC_102(), thickness, MaterialType::Monolithic, WavelengthRange::Solar,
                IntegrationType::PreWeighted);
        thickness = 5.715e-3;   // [m]

        aMaterial_102->setBandWavelengths(wavelengthSet);

        auto aMaterial_103 = SingleLayerOptics::Material::nBandMaterial(
                loadSampleData_NFRC_103(), thickness, MaterialType::Monolithic, WavelengthRange::Solar,
                IntegrationType::PreWeighted);

        aMaterial_103->setBandWavelengths(wavelengthSet);

        const CScatteringLayer Layer102 = CScatteringLayer::createSpecularLayer(aMaterial_102);
        const CScatteringLayer Layer103 = CScatteringLayer::createSpecularLayer(aMaterial_103);

        // Equivalent BSDF layer
        m_Layer = CMultiLayerScattered::create(Layer102);
        m_Layer->addLayer(Layer103);

        m_Layer->setSourceData(solarRadiationSpectrum);
    }

public:
    CMultiLayerScattered &getLayer() const {
        return *m_Layer;
    };
};

TEST_F(MultiPaneScattered_102_103_EN410, TestSpecular1) {
    SCOPED_TRACE("Begin Test: Specular layer - Scattering model front side (normal incidence).");

    auto &aLayer = getLayer();

    auto aSide = Side::Front;
    double theta = 0;
    double phi = 0;

    auto T_dir_dir =
            aLayer.getPropertySimple(PropertySimple::T, aSide, Scattering::DirectDirect, theta, phi);
    EXPECT_NEAR(0.652175, T_dir_dir, 1e-6);

    auto T_dir_dif =
            aLayer.getPropertySimple(PropertySimple::T, aSide, Scattering::DirectDiffuse, theta, phi);
    EXPECT_NEAR(0, T_dir_dif, 1e-6);

    auto T_dif_dif =
            aLayer.getPropertySimple(PropertySimple::T, aSide, Scattering::DiffuseDiffuse, theta, phi);
    EXPECT_NEAR(0.535661, T_dif_dif, 1e-6);

    auto R_dir_dir =
            aLayer.getPropertySimple(PropertySimple::R, aSide, Scattering::DirectDirect, theta, phi);
    EXPECT_NEAR(0.125666, R_dir_dir, 1e-6);

    auto R_dir_dif =
            aLayer.getPropertySimple(PropertySimple::R, aSide, Scattering::DirectDiffuse, theta, phi);
    EXPECT_NEAR(0, R_dir_dif, 1e-6);

    auto R_dif_dif =
            aLayer.getPropertySimple(PropertySimple::R, aSide, Scattering::DiffuseDiffuse, theta, phi);
    EXPECT_NEAR(0.228589, R_dif_dif, 1e-6);

    auto A_dir1 = aLayer.getAbsorptanceLayer(1, aSide, ScatteringSimple::Direct, theta, phi);
    EXPECT_NEAR(0.094341, A_dir1, 1e-6);

    auto A_dir2 = aLayer.getAbsorptanceLayer(2, aSide, ScatteringSimple::Direct, theta, phi);
    EXPECT_NEAR(0.127819, A_dir2, 1e-6);

    auto A_dif1 = aLayer.getAbsorptanceLayer(1, aSide, ScatteringSimple::Diffuse, theta, phi);
    EXPECT_NEAR(0.108470, A_dif1, 1e-6);

    auto A_dif2 = aLayer.getAbsorptanceLayer(2, aSide, ScatteringSimple::Diffuse, theta, phi);
    EXPECT_NEAR(0.127279, A_dif2, 1e-6);
}