/*
* author: 100086865
*
* The camera class. Here a new camera is created to look in a specified direction
*/

#pragma once
#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/glm.hpp>

#include "Ray.cuh"

class Camera {

public:

	//camera state
	glm::vec3 origin;
	glm::vec3 left;
	glm::vec3 horizontal;
	glm::vec3 vertical;

	/*
	* Creates a new camera based on the provided parameters
	* @param: camera origin
	* @param: camera vector
	* @param: up vector
	* @param: field of view
	* @param: aspect ratio
	*/
	__device__ Camera::Camera(glm::vec3 lookFrom, glm::vec3 lookAt, glm::vec3 up, float vfov, float aspect);

	/*
	* Returns the position of a ray
	* @param: u
	* @param: v
	*/
	__device__ Ray getRay(float u, float v);
};

__device__ Camera::Camera(glm::vec3 lookFrom, glm::vec3 lookAt, glm::vec3 up, float vfov, float aspect ) {
	glm::vec3 u, v, w;
	float theta = vfov * M_PI / 180.0f;
	float half_height = tan(theta / 2);
	float half_width = aspect * half_height;
	origin = lookFrom;
	w = glm::normalize(lookFrom - lookAt);
	u = glm::normalize(glm::cross(up, w));
	v = glm::cross(w, u);
	left = origin - half_width * u - half_height * v - w;
	horizontal = 2 * half_width * u;
	vertical = 2 * half_height * v;
}

__device__ Ray Camera::getRay(float u, float v) {
	return Ray(origin, left + u * horizontal + v * vertical - origin);
}