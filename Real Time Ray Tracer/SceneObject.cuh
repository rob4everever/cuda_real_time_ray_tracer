/*
* author: 100086865
*
* This is the base class for all objects within a scene. Every object
* has a method to deterine if it has been intersected by a ray.
*/

#pragma once

#include "Ray.cuh"

class Material;

/*
* Objects hit record
* t - parameter t
* point - point along the ray
* normal - object normal
*/
struct SceneObjectRecord {
	float t;
	float u;
	float v;
	glm::vec3 point;
	glm::vec3 normal;
	Material *material;
};

class SceneObject {
public:
	/*
	* Virtual method that determines if a ray intersects with a scene object
	* @param: The ray
	* @param: Smallest t
	* @param: Largest t
	* @param: Objects hit record
	* @return: If an intersection occurs between a scene object and a ray
	*/
	__device__ virtual bool intersect(const Ray &ray, float t_min, float t_max, SceneObjectRecord &record) const = 0;
};

