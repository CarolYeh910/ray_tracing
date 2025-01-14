#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"

class camera {
    public:
        __device__ camera(
            point3 lookfrom,
            point3 lookat,
            vec3   vup,
            float vfov, // vertical field-of-view in degrees
            float aspect_ratio,
            float aperture,
            float focus_dist,
            float _time0 = 0,
            float _time1 = 0
        ) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta/2.0f);
            float viewport_height = 2.0f * h;
            float viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - focus_dist*w;

            lens_radius = aperture / 2.0f;
            time0 = _time0;
            time1 = _time1;
        }

        __device__ ray get_ray(float s, float t, curandState* local_rand_state) const {
            vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            vec3 offset = u * rd.x() + v * rd.y();

            return ray(
                origin + offset,
                lower_left_corner + s*horizontal + t*vertical - origin - offset,
                random_float(time0, time1, local_rand_state)
            );
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lens_radius;
        float time0, time1;  // shutter open/close times
};
#endif