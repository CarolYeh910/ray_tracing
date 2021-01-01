#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "check_cuda.h"
#include "aabb.h"

class hittable_list : public hittable {
    public:
        __device__ hittable_list() {}
        __device__ hittable_list(hittable** list, size_t len) : objects(list), size(len) { }
        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;
        __device__ virtual bool bounding_box(
            float time0, float time1, aabb& output_box) const override;

    public:
        hittable** objects;
        size_t size;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (size_t i = 0; i < size; i++) {
        hittable* object = objects[i];
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ bool hittable_list::bounding_box(float time0, float time1, aabb& output_box) const {
	if (size == 0) return false;

    aabb temp_box;
    bool first_box = true;

    for (size_t i = 0; i < size; i++) {
        hittable* object = objects[i];
        if (!object->bounding_box(time0, time1, temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}

#endif