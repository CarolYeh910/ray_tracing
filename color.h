#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

__device__ void write_color(int *ib, color pixel_color, int samples_per_pixel) {
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    float scale = 1.0f / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0,255] value of each color component.
    ib[0] = static_cast<int>(256 * clamp(r, 0.0f, 0.999f));
    ib[1] = static_cast<int>(256 * clamp(g, 0.0f, 0.999f));
    ib[2] = static_cast<int>(256 * clamp(b, 0.0f, 0.999f));
}

#endif