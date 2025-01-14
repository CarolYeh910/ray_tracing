#ifndef BVH_H
#define BVH_H

#include "rtweekend.h"

#include "hittable.h"
#include "hittable_list.h"

#include <thrust/sort.h>

__device__ inline bool box_compare(hittable* a, hittable* b, int axis) {
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b)) {
        //std::cerr << "No bounding box in bvh_node constructor.\n";
    }

    return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ bool box_x_compare (hittable* a, hittable* b) {
    return box_compare(a, b, 0);
}

__device__ bool box_y_compare (hittable* a, hittable* b) {
    return box_compare(a, b, 1);
}

__device__ bool box_z_compare (hittable* a, hittable* b) {
    return box_compare(a, b, 2);
}

class bvh_node : public hittable {
    public:
        __device__ bvh_node();

        __device__ bvh_node(hittable_list* list, float time0, float time1, curandState* local_rand_state)
            : bvh_node(list->objects, 0, list->size, time0, time1, local_rand_state)
        {}

        __device__ bvh_node(
			hittable** (&src_objects),
            size_t start, size_t end, float time0, float time1,
            curandState* local_rand_state);

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

    public:
        hittable* left;
        hittable* right;
        aabb box;
};

__device__ bool bvh_node::bounding_box(float time0, float time1, aabb& output_box) const {
    output_box = box;
    return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ bvh_node::bvh_node(
    hittable** (&src_objects),
    size_t start, size_t end, float time0, float time1,
    curandState* local_rand_state
) {
    hittable** objects = src_objects; // Create a modifiable array of the source scene objects

	int axis = random_int(0, 2, local_rand_state);
    auto comparator = (axis == 0) ? box_x_compare
                    : (axis == 1) ? box_y_compare
                                  : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    } else if (object_span == 2) {
        if (comparator(objects[start], objects[start+1])) {
            left = objects[start];
            right = objects[start+1];
        } else {
            left = objects[start+1];
            right = objects[start];
        }
    } else {
        thrust::sort(objects + start, objects + end, comparator);

        auto mid = start + object_span/2;
        left = new bvh_node(objects, start, mid, time0, time1, local_rand_state);
        right = new bvh_node(objects, mid, end, time0, time1, local_rand_state);
    }

    aabb box_left, box_right;

    if (!left->bounding_box(time0, time1, box_left)
        || !right->bounding_box(time0, time1, box_right)
        ) {
        //std::cerr << "No bounding box in bvh_node constructor.\n";
    }

    box = surrounding_box(box_left, box_right);
}

#endif