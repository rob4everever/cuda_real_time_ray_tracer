/*
* author: 100086865
*
* SceneObject that defines a sphere and provides a function to 
* determine if it is intersected by a ray.
*/

#pragma once

#include "SceneObject.cuh"

class Sphere : public SceneObject {

public:

	//Sphere state
	glm::vec3 center;
	float radius;
	Material *material;

	/*
	* Creates a new Sphere object
	* @param: Center of the sphere
	* @param: Sphere radius
	*/
	__device__ Sphere(glm::vec3 center, float radius, Material *material) : center(center), radius(radius), material(material) {};

	/*
	* Virtual method to determine if a Sphere intersects with a ray
	* @param: The ray
	* @param: Smallest t
	* @param: Largest t
	* @param: Objects hit record
	* @return: If an intersection occurs between a Sphere and a ray
	*/
	__device__ virtual bool intersect(const Ray &ray, float t_min, float t_max, SceneObjectRecord &record) const;
};

__device__ bool Sphere::intersect(const Ray &ray, float t_min, float t_max, SceneObjectRecord &record) const {

	//Sphere equation
	glm::vec3 oc = ray.origin - center;
	float a = glm::dot(ray.direction, ray.direction);
	float b = glm::dot(oc, ray.direction);
	float c = glm::dot(oc, oc) - radius * radius;
	float dis = b * b - a * c;

	//if ray intersects
	if (dis > 0) {
	
		//sphere intersection
		float temp = (-b - sqrt(dis)) / a;
		//beteen t min and t max
		if (temp < t_max && temp > t_min) {
			record.t = temp;
			//calculate point of intersection
			record.point = ray.pointOnRay(record.t);
			//calculate normal direction
			record.normal = (record.point - center) / radius;
			record.material = material;
			return true;
		}
		temp = (-b + sqrt(dis)) / a;
		if (temp < t_max && temp > t_min) {
			record.t = temp;
			//calculate point of intersection
			record.point = ray.pointOnRay(record.t);
			//calculate normal direction
			record.normal = (record.point - center) / radius;
			record.material = material;
			return true;
		}
	}
	return false;
}

