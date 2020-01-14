/*
* author: 100086865
*
* In this file the material classes are defined, Diffuse, Specular
* and Fresnel which all inherit a base material class. Each material 
* has a function for dealing with incoming light to create the different 
* materials.
*/

#pragma once

#include <curand_kernel.h>
#include <glm/glm.hpp>
#include<glm/gtx/norm.hpp>

#include"Ray.cuh"
#include"Texture.cuh"
#include"SceneObject.cuh"

//random vector
#define RANDVEC3 glm::vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

//forward decleration
__device__ glm::vec3 randomUnitInSphere(curandState *local_rand_state);
__device__ glm::vec3 reflect(const glm::vec3 &v, const glm::vec3 &n);
__device__ bool refract(const glm::vec3 &incidentDirection, const glm::vec3 &surfaceNormal, float niNT, glm::vec3 &refracted);
__device__ float schlick(float cosine, float refIdx);

/*
* Material base class. Defines a virtual function that handles
* incoming light based on the material.
*/
class Material {

public:

	/*
	* Virtual function that determines if a ray is scattered
	* @param: Ray
	* @param: Objects hit record
	* @param: Emmited light 
	* @param: 
	* @param: RandomState
	*/
	__device__ virtual bool scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const = 0;
	
	__device__ virtual glm::vec3 emitted(float u, float v, const glm::vec3 &p) const {
		return glm::vec3(0, 0, 0);
	}
};

/*
* Represents a diffuse texture and provides functionality
* to handle incoming light rays by scattering them randomly
*/
class Diffuse : public Material {

public:

	//Diffuse state
	Texture *albedo;

	/*
	*
	*/
	__device__ Diffuse(Texture *a) : albedo(a) {};
	
	/*
	*
	*/
	__device__ virtual bool scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const;
};
__device__ bool Diffuse::scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const {
	//randomly scatter a ray using a random point in sphere (hit point - randompoint)
	glm::vec3 target = record.point + record.normal + randomUnitInSphere(local_rand_state);
	//createnew ray heading from intersection point in the direction of the new random target
	scattered = Ray(record.point, target - record.point);
	attenuation = albedo->value(0, 0, record.point); //if scattered how much should the ray be attenuated(reduced) when it scatters
	return true;
}

/*
* Represents a specular texture and provides functionality
* to handle incoming light rays by scattering them perpendicular
* to the hit point.
*/
class Specular : public Material {

public:

	//specular state
	glm::vec3 albedo;

	/*
	*
	*/
	__device__ Specular(const glm::vec3 &a) : albedo(a) {};

	/*	
	*
	*/
	__device__ virtual bool scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const;
};
__device__ bool Specular::scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const {
	//calculate reflected direction of intersection
	glm::vec3 reflected = reflect(glm::normalize(ray.direction), record.normal);
	//create new ray from intersection point, heading in the direction of reflection ray
	scattered = Ray(record.point, reflected);
	attenuation = albedo;	//if scattered how much should the ray be attenuated(reduced) when it scatters
	//return true if ray scatters (if > 0) otherwise ray is dead 
	return (glm::dot(scattered.direction, record.normal) > 0.0f);
}

/*
* Represents a frensel texture and provides functionality
* to handle incoming light rays by splitting them randomly
* into reflective or refractive rays
*/
class Fresnel : public Material {

public:

	//Fresnel state
	float ref_idx;
	
	/*
	*
	*/
	__device__ Fresnel(float ri) : ref_idx(ri) {}
	
	/*
	*
	*/
	__device__ virtual bool scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const;
};
__device__ bool Fresnel::scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const {
	glm::vec3 outNormal;
	glm::vec3 reflected = reflect(ray.direction, record.normal);
	float niNt;
	attenuation = glm::vec3(1.0, 1.0, 1.0);
	glm::vec3 refracted;
	float reflectProb;
	float cosine;

	if (glm::dot(ray.direction, record.normal) > 0.0f) {
		outNormal = -record.normal;
		niNt = ref_idx;
		cosine = glm::dot(ray.direction, record.normal) / ray.direction.length();
		cosine = sqrt(1.0f - ref_idx * ref_idx*(1 - cosine * cosine));
	}
	else {
		outNormal = record.normal;
		niNt = 1.0f / ref_idx;
		cosine = -glm::dot(ray.direction, record.normal) / ray.direction.length();
	}
	if (refract(ray.direction, outNormal, niNt, refracted)) {
		reflectProb = schlick(cosine, ref_idx);
	}
	else {
		reflectProb = 1.0f;
	}
	//Reflect/Refract split
	if (curand_uniform(local_rand_state) < reflectProb) {
		scattered = Ray(record.point, reflected);
	}
	else {
		scattered = Ray(record.point, refracted);
	}
	return true;
}

/*
* Models Schlick's approximation
* @param: cosine
* @param: refractive ID (glass is around 1.5)
*/
__device__ float schlick(float cosine, float refIdx) {
	float r0 = (1.0f - refIdx) / (1.0f + refIdx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0)*pow((1.0f - cosine), 5.0f);
}

/*
* Determines if an object can refact
* @param: incident ray direction
* @param: surface normal
* @param: ni/nt
* @param: refracted ray
*/
__device__ bool refract(const glm::vec3 &incidentDirection, const glm::vec3 &surfaceNormal, float niNT, glm::vec3 &refracted) {
	glm::vec3 uv = glm::normalize(incidentDirection);
	float dt = glm::dot(uv, surfaceNormal);
	float dis = 1.0f - niNT * niNT*(1.0f - dt * dt);
	if (dis > 0) {
		refracted = niNT * (uv - surfaceNormal * dt) - surfaceNormal * sqrt(dis);
		return true;
	}
	else {
		return false;
	}
}

/*
* Calculates the direction of a reflected ray
* @param: direction of the incident ray
* @param: objects surface normal 
*/
__device__ glm::vec3 reflect(const glm::vec3 &incidentDirection, const glm::vec3 &surfaceNormal) {
	return incidentDirection - 2.0f * glm::dot(incidentDirection, surfaceNormal) * surfaceNormal;
}

/*
* finds a random point within the objects radius
* @param: RandomState
*/
__device__ glm::vec3 randomUnitInSphere(curandState *local_rand_state) {
	glm::vec3 p;
	do {
		p = 2.0f * RANDVEC3 - glm::vec3(1, 1, 1);
	} while (glm::length2(p) >= 1.0f);
	return p;
}

class Light : public Material {
public:
	Texture *emittedLight;
	__device__ Light(Texture *texture) : emittedLight(texture) {}

	__device__ virtual bool scatter(const Ray &ray, const SceneObjectRecord &record, glm::vec3 &attenuation, Ray &scattered, curandState *local_rand_state) const {
		return false;
	}
	__device__ virtual glm::vec3 emitted(float u, float v, const glm::vec3 &p) const {
		return emittedLight->value(u, v, p);
	}
};