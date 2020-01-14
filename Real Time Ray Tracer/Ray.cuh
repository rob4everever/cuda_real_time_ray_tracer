/*
* author: 100086865
*
* A ray is created here and functionallity to return a ray and 
* determine a point along a ray is also defined here.
*/

#pragma once

#include "cuda_runtime.h"
#include <glm/glm.hpp>

class Ray {

public:

	//Ray state
	glm::vec3 origin;
	glm::vec3 direction;

	/*
	* Default constructor
	*/
	__device__ Ray();

	/*
	* Constructs a new ray
	* @param: the origin of the ray
	* @param: the direction the ray is firing in
	*/
	__device__ Ray(glm::vec3 origin, glm::vec3 direction);

	/*
	* Returns the position of a point along the line given t
	* @param: increment along the line
	* @return: position along the way
	*/
	__device__ glm::vec3 pointOnRay(float t) const;
};













__device__ Ray::Ray() {

}

__device__ Ray::Ray(glm::vec3 origin, glm::vec3 direction) {
	this->origin = origin;
	this->direction = direction;
}

__device__ glm::vec3 Ray::pointOnRay(float t) const {
	return this->origin + t * this->direction;
}