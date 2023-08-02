#include <memory>
#include <gtest/gtest.h>

#include "WCESpectralAveraging.hpp"
#include "WCEMultiLayerOptics.hpp"

using namespace SingleLayerOptics;
using namespace FenestrationCommon;
using namespace SpectralAveraging;
using namespace MultiLayerOptics;

// Example/test case on multlayer specular
// Difference from BSDF layer is that properties can be calculated at any custom angle

class MultiPaneSpecular_2191_103_FullSpectrum : public testing::Test
{
private:
    std::shared_ptr<CMultiPaneSpecular> m_Layer;

    static CSeries loadSolarRadiationFile()
    {
        // Full ASTM E891-87 Table 1 (Solar radiation)
        CSeries aSolarRadiation(
          {{0.3000, 0.0},    {0.3050, 3.4},    {0.3100, 15.6},   {0.3150, 41.1},   {0.3200, 71.2},
           {0.3250, 100.2},  {0.3300, 152.4},  {0.3350, 155.6},  {0.3400, 179.4},  {0.3450, 186.7},
           {0.3500, 212.0},  {0.3600, 240.5},  {0.3700, 324.0},  {0.3800, 362.4},  {0.3900, 381.7},
           {0.4000, 556.0},  {0.4100, 656.3},  {0.4200, 690.8},  {0.4300, 641.9},  {0.4400, 798.5},
           {0.4500, 956.6},  {0.4600, 990.0},  {0.4700, 998.0},  {0.4800, 1046.1}, {0.4900, 1005.1},
           {0.5000, 1026.7}, {0.5100, 1066.7}, {0.5200, 1011.5}, {0.5300, 1084.9}, {0.5400, 1082.4},
           {0.5500, 1102.2}, {0.5700, 1087.4}, {0.5900, 1024.3}, {0.6100, 1088.8}, {0.6300, 1062.1},
           {0.6500, 1061.7}, {0.6700, 1046.2}, {0.6900, 859.2},  {0.7100, 1002.4}, {0.7180, 816.9},
           {0.7244, 842.8},  {0.7400, 971.0},  {0.7525, 956.3},  {0.7575, 942.2},  {0.7625, 524.8},
           {0.7675, 830.7},  {0.7800, 908.9},  {0.8000, 873.4},  {0.8160, 712.0},  {0.8237, 660.2},
           {0.8315, 765.5},  {0.8400, 799.8},  {0.8600, 815.2},  {0.8800, 778.3},  {0.9050, 630.4},
           {0.9150, 565.2},  {0.9250, 586.4},  {0.9300, 348.1},  {0.9370, 224.2},  {0.9480, 271.4},
           {0.9650, 451.2},  {0.9800, 549.7},  {0.9935, 630.1},  {1.0400, 582.9},  {1.0700, 539.7},
           {1.1000, 366.2},  {1.1200, 98.1},   {1.1300, 169.5},  {1.1370, 118.7},  {1.1610, 301.9},
           {1.1800, 406.8},  {1.2000, 375.2},  {1.2350, 423.6},  {1.2900, 365.7},  {1.3200, 223.4},
           {1.3500, 30.1},   {1.3950, 1.4},    {1.4425, 51.6},   {1.4625, 97.0},   {1.4770, 97.3},
           {1.4970, 167.1},  {1.5200, 239.3},  {1.5390, 248.8},  {1.5580, 249.3},  {1.5780, 222.3},
           {1.5920, 227.3},  {1.6100, 210.5},  {1.6300, 224.7},  {1.6460, 215.9},  {1.6780, 202.8},
           {1.7400, 158.2},  {1.8000, 28.6},   {1.8600, 1.8},    {1.9200, 1.1},    {1.9600, 19.7},
           {1.9850, 84.9},   {2.0050, 25.0},   {2.0350, 92.5},   {2.0650, 56.3},   {2.1000, 82.7},
           {2.1480, 76.2},   {2.1980, 66.4},   {2.2700, 65.0},   {2.3600, 57.6},   {2.4500, 19.8},
           {2.4940, 17.0},   {2.5370, 3.0},    {2.9410, 4.0},    {2.9730, 7.0},    {3.0050, 6.0},
           {3.0560, 3.0},    {3.1320, 5.0},    {3.1560, 18.0},   {3.2040, 1.2},    {3.2450, 3.0},
           {3.3170, 12.0},   {3.3440, 3.0},    {3.4500, 12.2},   {3.5730, 11.0},   {3.7650, 9.0},
           {4.0450, 6.9}

          });

        return aSolarRadiation;
    }

    static std::shared_ptr<CSpectralSampleData> loadSampleData_NFRC_102()
    {
        auto aMeasurements_102 = CSpectralSampleData::create(
          {{0.300, 0.0020, 0.0470, 0.0480}, {0.305, 0.0030, 0.0470, 0.0480},
           {0.310, 0.0090, 0.0470, 0.0480}, {0.315, 0.0350, 0.0470, 0.0480},
           {0.320, 0.1000, 0.0470, 0.0480}, {0.325, 0.2180, 0.0490, 0.0500},
           {0.330, 0.3560, 0.0530, 0.0540}, {0.335, 0.4980, 0.0600, 0.0610},
           {0.340, 0.6160, 0.0670, 0.0670}, {0.345, 0.7090, 0.0730, 0.0740},
           {0.350, 0.7740, 0.0780, 0.0790}, {0.355, 0.8180, 0.0820, 0.0820},
           {0.360, 0.8470, 0.0840, 0.0840}, {0.365, 0.8630, 0.0850, 0.0850},
           {0.370, 0.8690, 0.0850, 0.0860}, {0.375, 0.8610, 0.0850, 0.0850},
           {0.380, 0.8560, 0.0840, 0.0840}, {0.385, 0.8660, 0.0850, 0.0850},
           {0.390, 0.8810, 0.0860, 0.0860}, {0.395, 0.8890, 0.0860, 0.0860},
           {0.400, 0.8930, 0.0860, 0.0860}, {0.410, 0.8930, 0.0860, 0.0860},
           {0.420, 0.8920, 0.0860, 0.0860}, {0.430, 0.8920, 0.0850, 0.0850},
           {0.440, 0.8920, 0.0850, 0.0850}, {0.450, 0.8960, 0.0850, 0.0850},
           {0.460, 0.9000, 0.0850, 0.0850}, {0.470, 0.9020, 0.0840, 0.0840},
           {0.480, 0.9030, 0.0840, 0.0840}, {0.490, 0.9040, 0.0850, 0.0850},
           {0.500, 0.9050, 0.0840, 0.0840}, {0.510, 0.9050, 0.0840, 0.0840},
           {0.520, 0.9050, 0.0840, 0.0840}, {0.530, 0.9040, 0.0840, 0.0840},
           {0.540, 0.9040, 0.0830, 0.0830}, {0.550, 0.9030, 0.0830, 0.0830},
           {0.560, 0.9020, 0.0830, 0.0830}, {0.570, 0.9000, 0.0820, 0.0820},
           {0.580, 0.8980, 0.0820, 0.0820}, {0.590, 0.8960, 0.0810, 0.0810},
           {0.600, 0.8930, 0.0810, 0.0810}, {0.610, 0.8900, 0.0810, 0.0810},
           {0.620, 0.8860, 0.0800, 0.0800}, {0.630, 0.8830, 0.0800, 0.0800},
           {0.640, 0.8790, 0.0790, 0.0790}, {0.650, 0.8750, 0.0790, 0.0790},
           {0.660, 0.8720, 0.0790, 0.0790}, {0.670, 0.8680, 0.0780, 0.0780},
           {0.680, 0.8630, 0.0780, 0.0780}, {0.690, 0.8590, 0.0770, 0.0770},
           {0.700, 0.8540, 0.0760, 0.0770}, {0.710, 0.8500, 0.0760, 0.0760},
           {0.720, 0.8450, 0.0750, 0.0760}, {0.730, 0.8400, 0.0750, 0.0750},
           {0.740, 0.8350, 0.0750, 0.0750}, {0.750, 0.8310, 0.0740, 0.0740},
           {0.760, 0.8260, 0.0740, 0.0740}, {0.770, 0.8210, 0.0740, 0.0740},
           {0.780, 0.8160, 0.0730, 0.0730}, {0.790, 0.8120, 0.0730, 0.0730},
           {0.800, 0.8080, 0.0720, 0.0720}, {0.810, 0.8030, 0.0720, 0.0720},
           {0.820, 0.8000, 0.0720, 0.0720}, {0.830, 0.7960, 0.0710, 0.0710},
           {0.840, 0.7930, 0.0700, 0.0710}, {0.850, 0.7880, 0.0700, 0.0710},
           {0.860, 0.7860, 0.0700, 0.0700}, {0.870, 0.7820, 0.0740, 0.0740},
           {0.880, 0.7800, 0.0720, 0.0720}, {0.890, 0.7770, 0.0730, 0.0740},
           {0.900, 0.7760, 0.0720, 0.0720}, {0.910, 0.7730, 0.0720, 0.0720},
           {0.920, 0.7710, 0.0710, 0.0710}, {0.930, 0.7700, 0.0700, 0.0700},
           {0.940, 0.7680, 0.0690, 0.0690}, {0.950, 0.7660, 0.0680, 0.0680},
           {0.960, 0.7660, 0.0670, 0.0680}, {0.970, 0.7640, 0.0680, 0.0680},
           {0.980, 0.7630, 0.0680, 0.0680}, {0.990, 0.7620, 0.0670, 0.0670},
           {1.000, 0.7620, 0.0660, 0.0670}, {1.050, 0.7600, 0.0660, 0.0660},
           {1.100, 0.7590, 0.0660, 0.0660}, {1.150, 0.7610, 0.0660, 0.0660},
           {1.200, 0.7650, 0.0660, 0.0660}, {1.250, 0.7700, 0.0650, 0.0650},
           {1.300, 0.7770, 0.0670, 0.0670}, {1.350, 0.7860, 0.0660, 0.0670},
           {1.400, 0.7950, 0.0670, 0.0680}, {1.450, 0.8080, 0.0670, 0.0670},
           {1.500, 0.8190, 0.0690, 0.0690}, {1.550, 0.8290, 0.0690, 0.0690},
           {1.600, 0.8360, 0.0700, 0.0700}, {1.650, 0.8400, 0.0700, 0.0700},
           {1.700, 0.8420, 0.0690, 0.0700}, {1.750, 0.8420, 0.0690, 0.0700},
           {1.800, 0.8410, 0.0700, 0.0700}, {1.850, 0.8400, 0.0690, 0.0690},
           {1.900, 0.8390, 0.0680, 0.0680}, {1.950, 0.8390, 0.0710, 0.0710},
           {2.000, 0.8390, 0.0690, 0.0690}, {2.050, 0.8400, 0.0680, 0.0680},
           {2.100, 0.8410, 0.0680, 0.0680}, {2.150, 0.8390, 0.0690, 0.0690},
           {2.200, 0.8300, 0.0700, 0.0700}, {2.250, 0.8300, 0.0700, 0.0700},
           {2.300, 0.8320, 0.0690, 0.0690}, {2.350, 0.8320, 0.0690, 0.0700},
           {2.400, 0.8320, 0.0700, 0.0700}, {2.450, 0.8260, 0.0690, 0.0690},
           {2.500, 0.8220, 0.0680, 0.0680}});

        return aMeasurements_102;
    }

    static std::shared_ptr<CSpectralSampleData> loadSampleData_NFRC_2191()
    {
        auto aMeasurements_103 = CSpectralSampleData::create(
          {{0.300, 0.0007, 0.0565, 0.1042},  {0.305, 0.0006, 0.0557, 0.1146},
           {0.310, 0.0019, 0.0552, 0.1321},  {0.315, 0.0083, 0.0550, 0.1537},
           {0.320, 0.0258, 0.0578, 0.1805},  {0.325, 0.0595, 0.0652, 0.2041},
           {0.330, 0.1079, 0.0832, 0.2254},  {0.335, 0.1638, 0.1121, 0.2413},
           {0.340, 0.2200, 0.1470, 0.2528},  {0.345, 0.2722, 0.1796, 0.2591},
           {0.350, 0.3182, 0.2066, 0.2626},  {0.355, 0.3592, 0.2261, 0.2640},
           {0.360, 0.3977, 0.2380, 0.2629},  {0.365, 0.4343, 0.2440, 0.2611},
           {0.370, 0.4695, 0.2437, 0.2580},  {0.375, 0.5015, 0.2364, 0.2528},
           {0.380, 0.5355, 0.2272, 0.2452},  {0.385, 0.5783, 0.2224, 0.2369},
           {0.390, 0.6243, 0.2181, 0.2273},  {0.395, 0.6656, 0.2101, 0.2157},
           {0.400, 0.7002, 0.1991, 0.2026},  {0.410, 0.7509, 0.1753, 0.1755},
           {0.420, 0.7853, 0.1540, 0.1510},  {0.430, 0.8099, 0.1375, 0.1308},
           {0.440, 0.8258, 0.1253, 0.1155},  {0.450, 0.8395, 0.1170, 0.1044},
           {0.460, 0.8496, 0.1113, 0.0964},  {0.470, 0.8566, 0.1072, 0.0912},
           {0.480, 0.8611, 0.1043, 0.0879},  {0.490, 0.8652, 0.1022, 0.0860},
           {0.500, 0.8682, 0.1007, 0.0850},  {0.510, 0.8709, 0.0994, 0.0844},
           {0.520, 0.8721, 0.0982, 0.0837},  {0.530, 0.8730, 0.0967, 0.0828},
           {0.540, 0.8738, 0.0953, 0.0818},  {0.550, 0.8738, 0.0938, 0.0805},
           {0.560, 0.8738, 0.0918, 0.0789},  {0.570, 0.8736, 0.0896, 0.0770},
           {0.580, 0.8729, 0.0875, 0.0751},  {0.590, 0.8720, 0.0848, 0.0726},
           {0.600, 0.8710, 0.0823, 0.0703},  {0.610, 0.8697, 0.0796, 0.0678},
           {0.620, 0.8682, 0.0767, 0.0649},  {0.630, 0.8665, 0.0738, 0.0621},
           {0.640, 0.8650, 0.0709, 0.0593},  {0.650, 0.8626, 0.0683, 0.0566},
           {0.660, 0.8609, 0.0656, 0.0539},  {0.670, 0.8583, 0.0629, 0.0511},
           {0.680, 0.8555, 0.0604, 0.0487},  {0.690, 0.8518, 0.0581, 0.0463},
           {0.700, 0.8481, 0.0562, 0.0443},  {0.710, 0.8448, 0.0543, 0.0423},
           {0.720, 0.8406, 0.0528, 0.0408},  {0.730, 0.8361, 0.0515, 0.0395},
           {0.740, 0.8312, 0.0506, 0.0386},  {0.750, 0.8257, 0.0499, 0.0380},
           {0.760, 0.8209, 0.0496, 0.0379},  {0.770, 0.8150, 0.0496, 0.0380},
           {0.780, 0.8096, 0.0499, 0.0386},  {0.790, 0.8028, 0.0505, 0.0398},
           {0.800, 0.7961, 0.0515, 0.0412},  {0.810, 0.7899, 0.0528, 0.0432},
           {0.820, 0.7840, 0.0544, 0.0457},  {0.830, 0.7775, 0.0561, 0.0486},
           {0.840, 0.7716, 0.0583, 0.0520},  {0.850, 0.7634, 0.0608, 0.0557},
           {0.860, 0.7576, 0.0638, 0.0601},  {0.870, 0.7515, 0.0667, 0.0652},
           {0.880, 0.7460, 0.0714, 0.0702},  {0.890, 0.7385, 0.0745, 0.0754},
           {0.900, 0.7307, 0.0770, 0.0814},  {0.910, 0.7227, 0.0815, 0.0877},
           {0.920, 0.7147, 0.0851, 0.0944},  {0.930, 0.7070, 0.0891, 0.1006},
           {0.940, 0.6988, 0.0937, 0.1080},  {0.950, 0.6905, 0.0980, 0.1150},
           {0.960, 0.6825, 0.1027, 0.1223},  {0.970, 0.6744, 0.1075, 0.1298},
           {0.980, 0.6664, 0.1123, 0.1378},  {0.990, 0.6584, 0.1174, 0.1457},
           {1.000, 0.6502, 0.1226, 0.1543},  {1.050, 0.6099, 0.1500, 0.1973},
           {1.100, 0.5703, 0.1793, 0.2425},  {1.150, 0.5325, 0.2093, 0.2881},
           {1.200, 0.4967, 0.2401, 0.3329},  {1.250, 0.4639, 0.2712, 0.3757},
           {1.300, 0.4340, 0.3024, 0.4157},  {1.350, 0.4065, 0.3338, 0.4527},
           {1.400, 0.3820, 0.3658, 0.4883},  {1.450, 0.3597, 0.3974, 0.5199},
           {1.500, 0.3393, 0.4292, 0.5490},  {1.550, 0.3198, 0.4585, 0.5758},
           {1.600, 0.3008, 0.4845, 0.6005},  {1.650, 0.2824, 0.5071, 0.6227},
           {1.700, 0.2649, 0.5256, 0.6433},  {1.750, 0.2483, 0.5415, 0.6628},
           {1.800, 0.2319, 0.5535, 0.6794},  {1.850, 0.2174, 0.5647, 0.6944},
           {1.900, 0.2049, 0.5755, 0.7092},  {1.950, 0.1934, 0.5863, 0.7233},
           {2.000, 0.1822, 0.5942, 0.7339},  {2.050, 0.1731, 0.6030, 0.7451},
           {2.100, 0.1624, 0.6110, 0.7554},  {2.150, 0.1550, 0.6157, 0.7629},
           {2.200, 0.1444, 0.6105, 0.7740},  {2.250, 0.1362, 0.6137, 0.7805},
           {2.300, 0.1307, 0.6260, 0.7899},  {2.350, 0.1264, 0.6335, 0.7956},
           {2.400, 0.1202, 0.6325, 0.8019},  {2.450, 0.1130, 0.6304, 0.8067},
           {2.500, 0.1078, 0.6227, 0.8021},  {5.000, 0.0000, 0.0310, 0.9310},
           {6.000, 0.0000, 0.0250, 0.9374},  {7.000, 0.0000, 0.0130, 0.9393},
           {8.000, 0.0000, 0.0030, 0.9411},  {9.000, 0.0000, 0.2100, 0.9439},
           {10.000, 0.0000, 0.2460, 0.9444}, {11.000, 0.0000, 0.1510, 0.9432},
           {12.000, 0.0000, 0.0710, 0.9437}, {13.000, 0.0000, 0.0800, 0.9448},
           {14.000, 0.0000, 0.0670, 0.9455}, {15.000, 0.0000, 0.0540, 0.9452},
           {16.000, 0.0000, 0.0450, 0.9454}, {17.000, 0.0000, 0.0310, 0.9445},
           {18.000, 0.0000, 0.0260, 0.9457}, {19.000, 0.0000, 0.0770, 0.9483},
           {20.000, 0.0000, 0.1830, 0.9489}, {21.000, 0.0000, 0.2360, 0.9490},
           {22.000, 0.0000, 0.2350, 0.9503}, {23.000, 0.0000, 0.2100, 0.9524},
           {24.000, 0.0000, 0.1870, 0.9508}, {25.000, 0.0000, 0.1700, 0.9497}});

        return aMeasurements_103;
    }

protected:
    virtual void SetUp()
    {
        double thickness = 3.048e-3;   // [m]
        const auto aMaterial_102 =
          Material::nBandMaterial(loadSampleData_NFRC_102(), thickness, MaterialType::Monolithic);

        thickness = 3e-3;   // [m]
        const auto aMaterial_2191 =
          Material::nBandMaterial(loadSampleData_NFRC_2191(), thickness, MaterialType::Monolithic);


        auto layer102 = SpecularLayer::createLayer(aMaterial_102);
        auto layer2191 = SpecularLayer::createLayer(aMaterial_2191);
        layer2191->Flipped(true);

        m_Layer = CMultiPaneSpecular::create({layer2191, layer102});

        const CalculationProperties input{loadSolarRadiationFile(),
                                          loadSolarRadiationFile().getXArray()};
        m_Layer->setCalculationProperties(input);
    }

public:
    [[nodiscard]] std::shared_ptr<CMultiPaneSpecular> getLayer() const
    {
        return m_Layer;
    };
};

TEST_F(MultiPaneSpecular_2191_103_FullSpectrum, TestAngle0)
{
    SCOPED_TRACE("Begin Test: Specular MultiLayerOptics layer - angle = 0 deg.");

    const double angle = 0;

    const double minLambda = 0.3;
    const double maxLambda = 2.5;

    CMultiPaneSpecular aLayer = *getLayer();

    const double T = aLayer.getPropertySimple(
      minLambda, maxLambda, PropertySimple::T, Side::Front, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.5862113966042467, T, 1e-6);

    const double Rf = aLayer.getPropertySimple(
      minLambda, maxLambda, PropertySimple::R, Side::Front, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.23011267434056742, Rf, 1e-6);

    const double Rb = aLayer.getPropertySimple(
      minLambda, maxLambda, PropertySimple::R, Side::Back, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.1903911215406871, Rb, 1e-6);

    const std::vector<double> AbsorptanceFront{aLayer.getAbsorptanceLayers(
      minLambda, maxLambda, Side::Front, ScatteringSimple::Direct, angle, 0)};

    const double AbsFront1 = aLayer.getAbsorptanceLayer(
      minLambda, maxLambda, 1, Side::Front, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.12766749504232652, AbsFront1, 1e-6);
    EXPECT_NEAR(0.12766749504232652, AbsorptanceFront[0], 1e-6);

    const double AbsFront2 = aLayer.getAbsorptanceLayer(
      minLambda, maxLambda, 2, Side::Front, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.056008434012859425, AbsFront2, 1e-6);
    EXPECT_NEAR(0.056008434012859425, AbsorptanceFront[1], 1e-6);

    const std::vector<double> AbsorptanceBack{aLayer.getAbsorptanceLayers(
      minLambda, maxLambda, Side::Back, ScatteringSimple::Direct, angle, 0)};

    const double AbsBack1 = aLayer.getAbsorptanceLayer(
      minLambda, maxLambda, 1, Side::Back, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.10474173607311184, AbsBack1, 1e-6);
    EXPECT_NEAR(0.10474173607311184, AbsorptanceBack[0], 1e-6);

    const double AbsBack2 = aLayer.getAbsorptanceLayer(
      minLambda, maxLambda, 2, Side::Back, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.1186557457819544, AbsBack2, 1e-6);
    EXPECT_NEAR(0.1186557457819544, AbsorptanceBack[1], 1e-6);

    const double Them = aLayer.getPropertySimple(
      minLambda, maxLambda, PropertySimple::T, Side::Front, Scattering::DiffuseDiffuse);
    EXPECT_NEAR(0.5092302931977486, Them, 1e-6);

    const double Rfhem = aLayer.getPropertySimple(
      minLambda, maxLambda, PropertySimple::R, Side::Front, Scattering::DiffuseDiffuse);
    EXPECT_NEAR(0.28196889594945301, Rfhem, 1e-6);

    const double Rbhem = aLayer.getPropertySimple(
      minLambda, maxLambda, PropertySimple::R, Side::Back, Scattering::DiffuseDiffuse);
    EXPECT_NEAR(0.25153667149679543, Rbhem, 1e-6);
}
