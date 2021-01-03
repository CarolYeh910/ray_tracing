#ifndef TEXTURE_H
#define TEXTURE_H

#include "rtweekend.h"
#include <iostream>


class my_texture  {
    public:
        __device__ virtual color value(float u, float v, const vec3& p) const = 0;
};


class solid_color : public my_texture {
    public:
        __host__ __device__ solid_color() {}
        __host__ __device__ solid_color(color c) : color_value(c) {}

        __host__ __device__ solid_color(float red, float green, float blue)
          : solid_color(color(red,green,blue)) {}

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            return color_value;
        }

    private:
        color color_value;
};


class checker_texture : public my_texture {
    public:
        __host__ __device__ checker_texture() {}

        __host__ __device__ checker_texture(my_texture* _even, my_texture* _odd)
            : even(_even), odd(_odd) {}

        __host__ __device__ checker_texture(color c1, color c2)
            : even(new solid_color(c1)) , odd(new solid_color(c2)) {}

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            float sines = sin(10.0f *p.x())*sin(10.0f *p.y())*sin(10.0f *p.z());
            if (sines < 0.0f)
                return odd->value(u, v, p);
            else
                return even->value(u, v, p);
        }

    public:
        my_texture* odd;
        my_texture* even;
};


class image_texture : public my_texture {
    public:
        const static int bytes_per_pixel = 3;

        __device__ image_texture()
          : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

        __device__ image_texture(unsigned char *p, int w, int h) {
            data = p;
            width = w;
            height = h;
            bytes_per_scanline = bytes_per_pixel * width;
        }

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            // If we have no texture data, then return solid cyan as a debugging aid.
            if (data == nullptr)
                return color(0,1,1);

            // Clamp input texture coordinates to [0,1] x [1,0]
            u = clamp(u, 0.0f, 1.0f);
            v = 1.0 - clamp(v, 0.0f, 1.0f);  // Flip V to image coordinates

            int i = static_cast<int>(u * width);
            int j = static_cast<int>(v * height);

            // Clamp integer mapping, since actual coordinates should be less than 1.0
            if (i >= width)  i = width-1;
            if (j >= height) j = height-1;

            const float color_scale = 1.0f / 255.0f;
            unsigned char* pixel = data + j*bytes_per_scanline + i*bytes_per_pixel;

            return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
        }

    private:
        unsigned char *data;
        int width, height;
        int bytes_per_scanline;
};

#endif