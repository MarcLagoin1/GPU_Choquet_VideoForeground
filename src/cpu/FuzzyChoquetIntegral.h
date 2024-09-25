//
// Created by Pierre-Louis Delcroix on 06/06/2023.
//

#ifndef GPU_FUZZYCHOQUETINTEGRAL_H
#define GPU_FUZZYCHOQUETINTEGRAL_H

#include "ChoquetIntegral.h"
#include "FeaturesExtraction.h"

class FuzzyChoquetIntegral {
public:
    static void process(const Image *fg, const Image *bg, Image *output);
};


#endif//GPU_FUZZYCHOQUETINTEGRAL_H
