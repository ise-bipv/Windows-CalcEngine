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

// Example/test case on multlayer specular
// Difference from BSDF layer is that properties can be calculated at any custom angle

class EquivalentSpecularLayer_102_103_Visible_Range : public testing::Test
{
private:
    std::shared_ptr<CMultiPaneSpecular> m_Layer;

    std::vector<double> getWavelengths() const
    {
        std::vector<double> aWavelengths{
                0.380, 0.385, 0.390, 0.395, 0.400, 0.405, 0.410, 0.415, 0.420, 0.425, 0.430, 0.435,
                0.440, 0.445, 0.450, 0.455, 0.460, 0.465, 0.470, 0.475, 0.480, 0.485, 0.490, 0.495,
                0.500, 0.505, 0.510, 0.515, 0.520, 0.525, 0.530, 0.535, 0.540, 0.545, 0.550, 0.555,
                0.560, 0.565, 0.570, 0.575, 0.580, 0.585, 0.590, 0.595, 0.600, 0.605, 0.610, 0.615,
                0.620, 0.625, 0.630, 0.635, 0.640, 0.645, 0.650, 0.655, 0.660, 0.665, 0.670, 0.675,
                0.680, 0.685, 0.690, 0.695, 0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735,
                0.740, 0.745, 0.750, 0.755, 0.760, 0.765, 0.770, 0.775, 0.780};

        return aWavelengths;
    }

    std::shared_ptr<CSeries> getSourceSpectrum() const
    {
        auto solarRadiation =
          std::make_shared<CSeries>(std::initializer_list<std::pair<double, double>>({

            // Full CIE Illuminant D651 nm ssp table (used for PHOTOPIC properties)
            {0.3000, 0.034100},  {0.3010, 0.360140},  {0.3020, 0.686180},  {0.3030, 1.012220},
            {0.3040, 1.338260},  {0.3050, 1.664300},  {0.3060, 1.990340},  {0.3070, 2.316380},
            {0.3080, 2.642420},  {0.3090, 2.968460},  {0.3100, 3.294500},  {0.3110, 4.988650},
            {0.3120, 6.682800},  {0.3130, 8.376950},  {0.3140, 10.071100}, {0.3150, 11.765200},
            {0.3160, 13.459400}, {0.3170, 15.153500}, {0.3180, 16.847700}, {0.3190, 18.541800},
            {0.3200, 20.236000}, {0.3210, 21.917700}, {0.3220, 23.599500}, {0.3230, 25.281200},
            {0.3240, 26.963000}, {0.3250, 28.644700}, {0.3260, 30.326500}, {0.3270, 32.008200},
            {0.3280, 33.690000}, {0.3290, 35.371700}, {0.3300, 37.053500}, {0.3310, 37.343000},
            {0.3320, 37.632600}, {0.3330, 37.922100}, {0.3340, 38.211600}, {0.3350, 38.501100},
            {0.3360, 38.790700}, {0.3370, 39.080200}, {0.3380, 39.369700}, {0.3390, 39.659300},
            {0.3400, 39.948800}, {0.3410, 40.445100}, {0.3420, 40.941400}, {0.3430, 41.437700},
            {0.3440, 41.934000}, {0.3450, 42.430200}, {0.3460, 42.926500}, {0.3470, 43.422800},
            {0.3480, 43.919100}, {0.3490, 44.415400}, {0.3500, 44.911700}, {0.3510, 45.084400},
            {0.3520, 45.257000}, {0.3530, 45.429700}, {0.3540, 45.602300}, {0.3550, 45.775000},
            {0.3560, 45.947700}, {0.3570, 46.120300}, {0.3580, 46.293000}, {0.3590, 46.465600},
            {0.3600, 46.638300}, {0.3610, 47.183400}, {0.3620, 47.728500}, {0.3630, 48.273500},
            {0.3640, 48.818600}, {0.3650, 49.363700}, {0.3660, 49.908800}, {0.3670, 50.453900},
            {0.3680, 50.998900}, {0.3690, 51.544000}, {0.3700, 52.089100}, {0.3710, 51.877700},
            {0.3720, 51.666400}, {0.3730, 51.455000}, {0.3740, 51.243700}, {0.3750, 51.032300},
            {0.3760, 50.820900}, {0.3770, 50.609600}, {0.3780, 50.398200}, {0.3790, 50.186900},
            {0.3800, 49.975500}, {0.3810, 50.442800}, {0.3820, 50.910000}, {0.3830, 51.377300},
            {0.3840, 51.844600}, {0.3850, 52.311800}, {0.3860, 52.779100}, {0.3870, 53.246400},
            {0.3880, 53.713700}, {0.3890, 54.180900}, {0.3900, 54.648200}, {0.3910, 57.458900},
            {0.3920, 60.269500}, {0.3930, 63.080200}, {0.3940, 65.890900}, {0.3950, 68.701500},
            {0.3960, 71.512200}, {0.3970, 74.322900}, {0.3980, 77.133600}, {0.3990, 79.944200},
            {0.4000, 82.754900}, {0.4010, 83.628000}, {0.4020, 84.501100}, {0.4030, 85.374200},
            {0.4040, 86.247300}, {0.4050, 87.120400}, {0.4060, 87.993600}, {0.4070, 88.866700},
            {0.4080, 89.739800}, {0.4090, 90.612900}, {0.4100, 91.486000}, {0.4110, 91.680600},
            {0.4120, 91.875200}, {0.4130, 92.069700}, {0.4140, 92.264300}, {0.4150, 92.458900},
            {0.4160, 92.653500}, {0.4170, 92.848100}, {0.4180, 93.042600}, {0.4190, 93.237200},
            {0.4200, 93.431800}, {0.4210, 92.756800}, {0.4220, 92.081900}, {0.4230, 91.406900},
            {0.4240, 90.732000}, {0.4250, 90.057000}, {0.4260, 89.382100}, {0.4270, 88.707100},
            {0.4280, 88.032200}, {0.4290, 87.357200}, {0.4300, 86.682300}, {0.4310, 88.500600},
            {0.4320, 90.318800}, {0.4330, 92.137100}, {0.4340, 93.955400}, {0.4350, 95.773600},
            {0.4360, 97.591900}, {0.4370, 99.410200}, {0.4380, 101.22800}, {0.4390, 103.04700},
            {0.4400, 104.86500}, {0.4410, 106.07900}, {0.4420, 107.29400}, {0.4430, 108.50800},
            {0.4440, 109.72200}, {0.4450, 110.93600}, {0.4460, 112.15100}, {0.4470, 113.36500},
            {0.4480, 114.57900}, {0.4490, 115.79400}, {0.4500, 117.00800}, {0.4510, 117.08800},
            {0.4520, 117.16900}, {0.4530, 117.24900}, {0.4540, 117.33000}, {0.4550, 117.41000},
            {0.4560, 117.49000}, {0.4570, 117.57100}, {0.4580, 117.65100}, {0.4590, 117.73200},
            {0.4600, 117.81200}, {0.4610, 117.51700}, {0.4620, 117.22200}, {0.4630, 116.92700},
            {0.4640, 116.63200}, {0.4650, 116.33600}, {0.4660, 116.04100}, {0.4670, 115.74600},
            {0.4680, 115.45100}, {0.4690, 115.15600}, {0.4700, 114.86100}, {0.4710, 114.96700},
            {0.4720, 115.07300}, {0.4730, 115.18000}, {0.4740, 115.28600}, {0.4750, 115.39200},
            {0.4760, 115.49800}, {0.4770, 115.60400}, {0.4780, 115.71100}, {0.4790, 115.81700},
            {0.4800, 115.92300}, {0.4810, 115.21200}, {0.4820, 114.50100}, {0.4830, 113.78900},
            {0.4840, 113.07800}, {0.4850, 112.36700}, {0.4860, 111.65600}, {0.4870, 110.94500},
            {0.4880, 110.23300}, {0.4890, 109.52200}, {0.4900, 108.81100}, {0.4910, 108.86500},
            {0.4920, 108.92000}, {0.4930, 108.97400}, {0.4940, 109.02800}, {0.4950, 109.08200},
            {0.4960, 109.13700}, {0.4970, 109.19100}, {0.4980, 109.24500}, {0.4990, 109.30000},
            {0.5000, 109.35400}, {0.5010, 109.19900}, {0.5020, 109.04400}, {0.5030, 108.88800},
            {0.5040, 108.73300}, {0.5050, 108.57800}, {0.5060, 108.42300}, {0.5070, 108.26800},
            {0.5080, 108.11200}, {0.5090, 107.95700}, {0.5100, 107.80200}, {0.5110, 107.50100},
            {0.5120, 107.20000}, {0.5130, 106.89800}, {0.5140, 106.59700}, {0.5150, 106.29600},
            {0.5160, 105.99500}, {0.5170, 105.69400}, {0.5180, 105.39200}, {0.5190, 105.09100},
            {0.5200, 104.79000}, {0.5210, 105.08000}, {0.5220, 105.37000}, {0.5230, 105.66000},
            {0.5240, 105.95000}, {0.5250, 106.23900}, {0.5260, 106.52900}, {0.5270, 106.81900},
            {0.5280, 107.10900}, {0.5290, 107.39900}, {0.5300, 107.68900}, {0.5310, 107.36100},
            {0.5320, 107.03200}, {0.5330, 106.70400}, {0.5340, 106.37500}, {0.5350, 106.04700},
            {0.5360, 105.71900}, {0.5370, 105.39000}, {0.5380, 105.06200}, {0.5390, 104.73300},
            {0.5400, 104.40500}, {0.5410, 104.36900}, {0.5420, 104.33300}, {0.5430, 104.29700},
            {0.5440, 104.26100}, {0.5450, 104.22500}, {0.5460, 104.19000}, {0.5470, 104.15400},
            {0.5480, 104.11800}, {0.5490, 104.08200}, {0.5500, 104.04600}, {0.5510, 103.64100},
            {0.5520, 103.23700}, {0.5530, 102.83200}, {0.5540, 102.42800}, {0.5550, 102.02300},
            {0.5560, 101.61800}, {0.5570, 101.21400}, {0.5580, 100.80900}, {0.5590, 100.40500},
            {0.5600, 100.00000}, {0.5610, 99.633400}, {0.5620, 99.266800}, {0.5630, 98.900300},
            {0.5640, 98.533700}, {0.5650, 98.167100}, {0.5660, 97.800500}, {0.5670, 97.433900},
            {0.5680, 97.067400}, {0.5690, 96.700800}, {0.5700, 96.334200}, {0.5710, 96.279600},
            {0.5720, 96.225000}, {0.5730, 96.170300}, {0.5740, 96.115700}, {0.5750, 96.061100},
            {0.5760, 96.006500}, {0.5770, 95.951900}, {0.5780, 95.897200}, {0.5790, 95.842600},
            {0.5800, 95.788000}, {0.5810, 95.077800}, {0.5820, 94.367500}, {0.5830, 93.657300},
            {0.5840, 92.947000}, {0.5850, 92.236800}, {0.5860, 91.526600}, {0.5870, 90.816300},
            {0.5880, 90.106100}, {0.5890, 89.395800}, {0.5900, 88.685600}, {0.5910, 88.817700},
            {0.5920, 88.949700}, {0.5930, 89.081800}, {0.5940, 89.213800}, {0.5950, 89.345900},
            {0.5960, 89.478000}, {0.5970, 89.610000}, {0.5980, 89.742100}, {0.5990, 89.874100},
            {0.6000, 90.006200}, {0.6010, 89.965500}, {0.6020, 89.924800}, {0.6030, 89.884100},
            {0.6040, 89.843400}, {0.6050, 89.802600}, {0.6060, 89.761900}, {0.6070, 89.721200},
            {0.6080, 89.680500}, {0.6090, 89.639800}, {0.6100, 89.599100}, {0.6110, 89.409100},
            {0.6120, 89.219000}, {0.6130, 89.029000}, {0.6140, 88.838900}, {0.6150, 88.648900},
            {0.6160, 88.458900}, {0.6170, 88.268800}, {0.6180, 88.078800}, {0.6190, 87.888700},
            {0.6200, 87.698700}, {0.6210, 87.257700}, {0.6220, 86.816700}, {0.6230, 86.375700},
            {0.6240, 85.934700}, {0.6250, 85.493600}, {0.6260, 85.052600}, {0.6270, 84.611600},
            {0.6280, 84.170600}, {0.6290, 83.729600}, {0.6300, 83.288600}, {0.6310, 83.329700},
            {0.6320, 83.370700}, {0.6330, 83.411800}, {0.6340, 83.452800}, {0.6350, 83.493900},
            {0.6360, 83.535000}, {0.6370, 83.576000}, {0.6380, 83.617100}, {0.6390, 83.658100},
            {0.6400, 83.699200}, {0.6410, 83.332000}, {0.6420, 82.964700}, {0.6430, 82.597500},
            {0.6440, 82.230200}, {0.6450, 81.863000}, {0.6460, 81.495800}, {0.6470, 81.128500},
            {0.6480, 80.761300}, {0.6490, 80.394000}, {0.6500, 80.026800}, {0.6510, 80.045600},
            {0.6520, 80.064400}, {0.6530, 80.083100}, {0.6540, 80.101900}, {0.6550, 80.120700},
            {0.6560, 80.139500}, {0.6570, 80.158300}, {0.6580, 80.177000}, {0.6590, 80.195800},
            {0.6600, 80.214600}, {0.6610, 80.420900}, {0.6620, 80.627200}, {0.6630, 80.833600},
            {0.6640, 81.039900}, {0.6650, 81.246200}, {0.6660, 81.452500}, {0.6670, 81.658800},
            {0.6680, 81.865200}, {0.6690, 82.071500}, {0.6700, 82.277800}, {0.6710, 81.878400},
            {0.6720, 81.479100}, {0.6730, 81.079700}, {0.6740, 80.680400}, {0.6750, 80.281000},
            {0.6760, 79.881600}, {0.6770, 79.482300}, {0.6780, 79.082900}, {0.6790, 78.683600},
            {0.6800, 78.284200}, {0.6810, 77.427900}, {0.6820, 76.571600}, {0.6830, 75.715300},
            {0.6840, 74.859000}, {0.6850, 74.002700}, {0.6860, 73.146500}, {0.6870, 72.290200},
            {0.6880, 71.433900}, {0.6890, 70.577600}, {0.6900, 69.721300}, {0.6910, 69.910100},
            {0.6920, 70.098900}, {0.6930, 70.287600}, {0.6940, 70.476400}, {0.6950, 70.665200},
            {0.6960, 70.854000}, {0.6970, 71.042800}, {0.6980, 71.231500}, {0.6990, 71.420300},
            {0.7000, 71.609100}, {0.7010, 71.883100}, {0.7020, 72.157100}, {0.7030, 72.431100},
            {0.7040, 72.705100}, {0.7050, 72.979000}, {0.7060, 73.253000}, {0.7070, 73.527000},
            {0.7080, 73.801000}, {0.7090, 74.075000}, {0.7100, 74.349000}, {0.7110, 73.074500},
            {0.7120, 71.800000}, {0.7130, 70.525500}, {0.7140, 69.251000}, {0.7150, 67.976500},
            {0.7160, 66.702000}, {0.7170, 65.427500}, {0.7180, 64.153000}, {0.7190, 62.878500},
            {0.7200, 61.604000}, {0.7210, 62.432200}, {0.7220, 63.260300}, {0.7230, 64.088500},
            {0.7240, 64.916600}, {0.7250, 65.744800}, {0.7260, 66.573000}, {0.7270, 67.401100},
            {0.7280, 68.229300}, {0.7290, 69.057400}, {0.7300, 69.885600}, {0.7310, 70.405700},
            {0.7320, 70.925900}, {0.7330, 71.446000}, {0.7340, 71.966200}, {0.7350, 72.486300},
            {0.7360, 73.006400}, {0.7370, 73.526600}, {0.7380, 74.046700}, {0.7390, 74.566900},
            {0.7400, 75.087000}, {0.7410, 73.937600}, {0.7420, 72.788100}, {0.7430, 71.638700},
            {0.7440, 70.489300}, {0.7450, 69.339800}, {0.7460, 68.190400}, {0.7470, 67.041000},
            {0.7480, 65.891600}, {0.7490, 64.742100}, {0.7500, 63.592700}, {0.7510, 61.875200},
            {0.7520, 60.157800}, {0.7530, 58.440300}, {0.7540, 56.722900}, {0.7550, 55.005400},
            {0.7560, 53.288000}, {0.7570, 51.570500}, {0.7580, 49.853100}, {0.7590, 48.135600},
            {0.7600, 46.418200}, {0.7610, 48.456900}, {0.7620, 50.495600}, {0.7630, 52.534400},
            {0.7640, 54.573100}, {0.7650, 56.611800}, {0.7660, 58.650500}, {0.7670, 60.689200},
            {0.7680, 62.728000}, {0.7690, 64.766700}, {0.7700, 66.805400}, {0.7710, 66.463100},
            {0.7720, 66.120900}, {0.7730, 65.778600}, {0.7740, 65.436400}, {0.7750, 65.094100},
            {0.7760, 64.751800}, {0.7770, 64.409600}, {0.7780, 64.067300}, {0.7790, 63.725100},
            {0.7800, 63.382800}, {0.7810, 63.474900}, {0.7820, 63.567000}, {0.7830, 63.659200},
            {0.7840, 63.751300}, {0.7850, 63.843400}, {0.7860, 63.935500}, {0.7870, 64.027600},
            {0.7880, 64.119800}, {0.7890, 64.211900}, {0.7900, 64.304000}, {0.7910, 63.818800},
            {0.7920, 63.333600}, {0.7930, 62.848400}, {0.7940, 62.363200}, {0.7950, 61.877900},
            {0.7960, 61.392700}, {0.7970, 60.907500}, {0.7980, 60.422300}, {0.7990, 59.937100},
            {0.8000, 59.451900}, {0.8010, 58.702600}, {0.8020, 57.953300}, {0.8030, 57.204000},
            {0.8040, 56.454700}, {0.8050, 55.705400}, {0.8060, 54.956200}, {0.8070, 54.206900},
            {0.8080, 53.457600}, {0.8090, 52.708300}, {0.8100, 51.959000}, {0.8110, 52.507200},
            {0.8120, 53.055300}, {0.8130, 53.603500}, {0.8140, 54.151600}, {0.8150, 54.699800},
            {0.8160, 55.248000}, {0.8170, 55.796100}, {0.8180, 56.344300}, {0.8190, 56.892400},
            {0.8200, 57.440600}, {0.8210, 57.727800}, {0.8220, 58.015000}, {0.8230, 58.302200},
            {0.8240, 58.589400}, {0.8250, 58.876500}, {0.8260, 59.163700}, {0.8270, 59.450900},
            {0.8280, 59.738100}, {0.8290, 60.025300}, {0.8300, 60.312500}}));

        return solarRadiation;
    }

    std::shared_ptr<CSeries> getDetectorData() const
    {
        auto detectorData =
          std::make_shared<CSeries>(std::initializer_list<std::pair<double, double>>({

            {0.380, 0.0000}, {0.385, 0.0001}, {0.390, 0.0001}, {0.395, 0.0002}, {0.400, 0.0004},
            {0.405, 0.0006}, {0.410, 0.0012}, {0.415, 0.0022}, {0.420, 0.0040}, {0.425, 0.0073},
            {0.430, 0.0116}, {0.435, 0.0168}, {0.440, 0.0230}, {0.445, 0.0298}, {0.450, 0.0380},
            {0.455, 0.0480}, {0.460, 0.0600}, {0.465, 0.0739}, {0.470, 0.0910}, {0.475, 0.1126},
            {0.480, 0.1390}, {0.485, 0.1693}, {0.490, 0.2080}, {0.495, 0.2586}, {0.500, 0.3230},
            {0.505, 0.4073}, {0.510, 0.5030}, {0.515, 0.6082}, {0.520, 0.7100}, {0.525, 0.7932},
            {0.530, 0.8620}, {0.535, 0.9149}, {0.540, 0.9540}, {0.545, 0.9803}, {0.550, 0.9950},
            {0.555, 1.0000}, {0.560, 0.9950}, {0.565, 0.9786}, {0.570, 0.9520}, {0.575, 0.9154},
            {0.580, 0.8700}, {0.585, 0.8163}, {0.590, 0.7570}, {0.595, 0.6949}, {0.600, 0.6310},
            {0.605, 0.5668}, {0.610, 0.5030}, {0.615, 0.4412}, {0.620, 0.3810}, {0.625, 0.3210},
            {0.630, 0.2650}, {0.635, 0.2170}, {0.640, 0.1750}, {0.645, 0.1382}, {0.650, 0.1070},
            {0.655, 0.0816}, {0.660, 0.0610}, {0.665, 0.0446}, {0.670, 0.0320}, {0.675, 0.0232},
            {0.680, 0.0170}, {0.685, 0.0119}, {0.690, 0.0082}, {0.695, 0.0057}, {0.700, 0.0041},
            {0.705, 0.0029}, {0.710, 0.0021}, {0.715, 0.0015}, {0.720, 0.0010}, {0.725, 0.0007},
            {0.730, 0.0005}, {0.735, 0.0004}, {0.740, 0.0002}, {0.745, 0.0002}, {0.750, 0.0001},
            {0.755, 0.0001}, {0.760, 0.0001}, {0.765, 0.0000}, {0.770, 0.0000}, {0.775, 0.0000},
            {0.780, 0.0000}}));

        return detectorData;
    }

    std::shared_ptr<CSpectralSampleData> loadSampleData_NFRC_102() const {
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

    std::shared_ptr<CSpectralSampleData> loadSampleData_NFRC_103() const {
        auto aMeasurements_103 = CSpectralSampleData::create(
          {{0.300, 0.0000, 0.0470, 0.0490}, {0.305, 0.0050, 0.0470, 0.0490},
           {0.310, 0.0000, 0.0470, 0.0480}, {0.315, 0.0030, 0.0460, 0.0480},
           {0.320, 0.0190, 0.0460, 0.0480}, {0.325, 0.0660, 0.0450, 0.0460},
           {0.330, 0.1600, 0.0450, 0.0470}, {0.335, 0.2940, 0.0490, 0.0500},
           {0.340, 0.4370, 0.0550, 0.0560}, {0.345, 0.5660, 0.0620, 0.0620},
           {0.350, 0.6710, 0.0690, 0.0690}, {0.355, 0.7440, 0.0740, 0.0740},
           {0.360, 0.7930, 0.0780, 0.0780}, {0.365, 0.8220, 0.0800, 0.0800},
           {0.370, 0.8320, 0.0810, 0.0810}, {0.375, 0.8190, 0.0800, 0.0800},
           {0.380, 0.8090, 0.0790, 0.0790}, {0.385, 0.8290, 0.0800, 0.0800},
           {0.390, 0.8530, 0.0820, 0.0820}, {0.395, 0.8680, 0.0830, 0.0830},
           {0.400, 0.8750, 0.0830, 0.0830}, {0.410, 0.8750, 0.0830, 0.0830},
           {0.420, 0.8730, 0.0830, 0.0830}, {0.430, 0.8730, 0.0820, 0.0820},
           {0.440, 0.8730, 0.0820, 0.0820}, {0.450, 0.8800, 0.0820, 0.0820},
           {0.460, 0.8870, 0.0820, 0.0820}, {0.470, 0.8900, 0.0820, 0.0820},
           {0.480, 0.8920, 0.0830, 0.0830}, {0.490, 0.8930, 0.0820, 0.0820},
           {0.500, 0.8940, 0.0820, 0.0820}, {0.510, 0.8950, 0.0820, 0.0820},
           {0.520, 0.8950, 0.0820, 0.0820}, {0.530, 0.8940, 0.0820, 0.0820},
           {0.540, 0.8930, 0.0810, 0.0810}, {0.550, 0.8910, 0.0810, 0.0810},
           {0.560, 0.8880, 0.0810, 0.0810}, {0.570, 0.8840, 0.0800, 0.0800},
           {0.580, 0.8810, 0.0800, 0.0800}, {0.590, 0.8760, 0.0790, 0.0790},
           {0.600, 0.8710, 0.0790, 0.0790}, {0.610, 0.8650, 0.0780, 0.0780},
           {0.620, 0.8590, 0.0770, 0.0770}, {0.630, 0.8530, 0.0770, 0.0770},
           {0.640, 0.8470, 0.0760, 0.0760}, {0.650, 0.8400, 0.0750, 0.0750},
           {0.660, 0.8330, 0.0750, 0.0750}, {0.670, 0.8260, 0.0740, 0.0740},
           {0.680, 0.8180, 0.0730, 0.0730}, {0.690, 0.8100, 0.0730, 0.0730},
           {0.700, 0.8020, 0.0720, 0.0720}, {0.710, 0.7940, 0.0710, 0.0720},
           {0.720, 0.7860, 0.0710, 0.0710}, {0.730, 0.7770, 0.0700, 0.0700},
           {0.740, 0.7690, 0.0690, 0.0700}, {0.750, 0.7610, 0.0690, 0.0690},
           {0.760, 0.7520, 0.0680, 0.0680}, {0.770, 0.7440, 0.0670, 0.0680},
           {0.780, 0.7360, 0.0670, 0.0670}, {0.790, 0.7290, 0.0660, 0.0660},
           {0.800, 0.7220, 0.0660, 0.0660}, {0.810, 0.7150, 0.0650, 0.0660},
           {0.820, 0.7100, 0.0650, 0.0650}, {0.830, 0.7020, 0.0640, 0.0650},
           {0.840, 0.6980, 0.0640, 0.0640}, {0.850, 0.6900, 0.0630, 0.0640},
           {0.860, 0.6870, 0.0650, 0.0650}, {0.870, 0.6810, 0.0670, 0.0670},
           {0.880, 0.6770, 0.0650, 0.0660}, {0.890, 0.6730, 0.0660, 0.0660},
           {0.900, 0.6700, 0.0650, 0.0660}, {0.910, 0.6670, 0.0650, 0.0650},
           {0.920, 0.6640, 0.0640, 0.0640}, {0.930, 0.6600, 0.0630, 0.0630},
           {0.940, 0.6580, 0.0640, 0.0640}, {0.950, 0.6560, 0.0630, 0.0630},
           {0.960, 0.6540, 0.0610, 0.0610}, {0.970, 0.6530, 0.0620, 0.0620},
           {0.980, 0.6510, 0.0610, 0.0620}, {0.990, 0.6490, 0.0610, 0.0620},
           {1.000, 0.6480, 0.0590, 0.0600}, {1.050, 0.6450, 0.0590, 0.0600},
           {1.100, 0.6450, 0.0580, 0.0590}, {1.150, 0.6470, 0.0590, 0.0590},
           {1.200, 0.6530, 0.0590, 0.0590}, {1.250, 0.6610, 0.0580, 0.0590},
           {1.300, 0.6730, 0.0600, 0.0600}, {1.350, 0.6870, 0.0600, 0.0600},
           {1.400, 0.7020, 0.0610, 0.0610}, {1.450, 0.7220, 0.0610, 0.0620},
           {1.500, 0.7410, 0.0630, 0.0640}, {1.550, 0.7570, 0.0630, 0.0640},
           {1.600, 0.7690, 0.0650, 0.0650}, {1.650, 0.7750, 0.0650, 0.0640},
           {1.700, 0.7790, 0.0640, 0.0650}, {1.750, 0.7790, 0.0650, 0.0650},
           {1.800, 0.7770, 0.0650, 0.0650}, {1.850, 0.7760, 0.0650, 0.0630},
           {1.900, 0.7730, 0.0620, 0.0620}, {1.950, 0.7730, 0.0650, 0.0650},
           {2.000, 0.7720, 0.0650, 0.0650}, {2.050, 0.7740, 0.0640, 0.0640},
           {2.100, 0.7750, 0.0640, 0.0650}, {2.150, 0.7730, 0.0650, 0.0650},
           {2.200, 0.7580, 0.0640, 0.0650}, {2.250, 0.7590, 0.0640, 0.0640},
           {2.300, 0.7660, 0.0650, 0.0650}, {2.350, 0.7670, 0.0640, 0.0650},
           {2.400, 0.7660, 0.0640, 0.0640}, {2.450, 0.7570, 0.0640, 0.0640},
           {2.500, 0.7500, 0.0630, 0.0630}});

        return aMeasurements_103;
    }

protected:
    virtual void SetUp()
    {
        const auto minLambda = 0.38;
        const auto maxLambda = 0.78;

        double thickness = 3.048e-3;   // [m]
        const auto aMaterial_102 = Material::nBandMaterial(loadSampleData_NFRC_102(),
                                                           thickness,
                                                           MaterialType::Monolithic,
                                                           minLambda,
                                                           maxLambda);

        aMaterial_102->setBandWavelengths(getWavelengths());

        thickness = 5.715e-3;   // [m]
        const auto aMaterial_103 = Material::nBandMaterial(loadSampleData_NFRC_103(),
                                                           thickness,
                                                           MaterialType::Monolithic,
                                                           minLambda,
                                                           maxLambda);

        aMaterial_103->setBandWavelengths(getWavelengths());

        const auto layer102 = SpecularLayer::createLayer(aMaterial_102);
        const auto layer103 = SpecularLayer::createLayer(aMaterial_103);

        m_Layer = CMultiPaneSpecular::create({layer102, layer103}, getSourceSpectrum(), getDetectorData());

    }

public:
    std::shared_ptr<CMultiPaneSpecular> getLayer() const
    {
        return m_Layer;
    };

};
TEST_F(EquivalentSpecularLayer_102_103_Visible_Range, TestAngle0)
{
    SCOPED_TRACE("Begin Test: Specular MultiLayerOptics layer - angle = 0 deg.");

    const double angle = 0;

    CMultiPaneSpecular aLayer = *getLayer();

    const double T =
      aLayer.getPropertySimple(PropertySimple::T, Side::Front, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.800040, T, 1e-6);

    const double Rf =
      aLayer.getPropertySimple(PropertySimple::R, Side::Front, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.148036, Rf, 1e-6);

    const double Rb =
      aLayer.getPropertySimple(PropertySimple::R, Side::Back, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.145341, Rb, 1e-6);

    const double Abs1 =
      aLayer.getAbsorptanceLayer(1, Side::Front, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.036833, Abs1, 1e-6);

    const double Abs2 =
      aLayer.getAbsorptanceLayer(2, Side::Front, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.056681, Abs2, 1e-6);

    const double Them =
      aLayer.getPropertySimple(PropertySimple::T, Side::Front, Scattering::DiffuseDiffuse);
    EXPECT_NEAR(0.697288, Them, 1e-6);

    const double Rfhem =
      aLayer.getPropertySimple(PropertySimple::R, Side::Front, Scattering::DiffuseDiffuse);
    EXPECT_NEAR(0.235641, Rfhem, 1e-6);

    const double Rbhem =
      aLayer.getPropertySimple(PropertySimple::R, Side::Back, Scattering::DiffuseDiffuse);
    EXPECT_NEAR(0.230712, Rbhem, 1e-6);
}

TEST_F(EquivalentSpecularLayer_102_103_Visible_Range, TestAngle10)
{
    SCOPED_TRACE("Begin Test: Specular MultiLayerOptics layer - angle = 10 deg.");

    const double angle = 10;

    CMultiPaneSpecular aLayer = *getLayer();

    const double T =
      aLayer.getPropertySimple(PropertySimple::T, Side::Front, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.799712, T, 1e-6);

    const double Rf =
      aLayer.getPropertySimple(PropertySimple::R, Side::Front, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.148037, Rf, 1e-6);

    const double Rb =
      aLayer.getPropertySimple(PropertySimple::R, Side::Back, Scattering::DirectDirect, angle, 0);
    EXPECT_NEAR(0.145324, Rb, 1e-6);

    const double Abs1 =
      aLayer.getAbsorptanceLayer(1, Side::Front, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.037065, Abs1, 1e-6);

    const double Abs2 =
      aLayer.getAbsorptanceLayer(2, Side::Front, ScatteringSimple::Direct, angle, 0);
    EXPECT_NEAR(0.057009, Abs2, 1e-6);
}

TEST_F(EquivalentSpecularLayer_102_103_Visible_Range, TestAngleHemispherical10)
{
    SCOPED_TRACE("Begin Test: Hemispherical to hemispherical with ten integration points.");

    const double minLambda = 0.3;
    const double maxLambda = 2.5;
    std::vector<double> aAngles{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

    CMultiPaneSpecular aLayer = *getLayer();

    double Tfhem =
      aLayer.getHemisphericalProperty(Side::Front, Property::T, aAngles, minLambda, maxLambda);
    EXPECT_NEAR(0.697288, Tfhem, 1e-6);

    double Tbhem =
      aLayer.getHemisphericalProperty(Side::Back, Property::T, aAngles, minLambda, maxLambda);
    EXPECT_NEAR(0.697288, Tbhem, 1e-6);

    double Rfhem =
      aLayer.getHemisphericalProperty(Side::Front, Property::R, aAngles, minLambda, maxLambda);
    EXPECT_NEAR(0.235641, Rfhem, 1e-6);

    double Rbhem =
      aLayer.getHemisphericalProperty(Side::Back, Property::R, aAngles, minLambda, maxLambda);
    EXPECT_NEAR(0.230712, Rbhem, 1e-6);

    double Abs1 = aLayer.AbsHemispherical(1, aAngles, minLambda, maxLambda);
    EXPECT_NEAR(0.042672, Abs1, 1e-6);

    double Abs2 = aLayer.AbsHemispherical(2, aAngles, minLambda, maxLambda);
    EXPECT_NEAR(0.058583, Abs2, 1e-6);
}