/*
* author: 100086865
*
* This class represents an entire scene. A scene is essentially a list
* of objects. A method is defined that determines if a ray intersects with
* any object within a scene.
*/

#pragma once

#include "SceneObject.cuh"

class Scene : public SceneObject {
public:

	//Scene state
	SceneObject **list;
	int listSize;

	/*
	* Creates a new scene with n objects
	* @param: List of scene objects
	* @param: Number of scene objects
	*/
	__device__ Scene(SceneObject **list, int n);

	/*
	* Virtual method to determine if a Scene Object intersects with a ray
	* @param: The ray
	* @param: Smallest t
	* @param: Largest t
	* @param: Objects hit record
	* @return: If an intersection occurs between a Scene Object and a ray
	*/
	__device__ virtual bool intersect(const Ray &ray, float t_min, float t_max, SceneObjectRecord &record) const;

};

__device__ Scene::Scene(SceneObject **list, int n) {
	this->list = list;
	this->listSize = n;
}

__device__ bool Scene::intersect(const Ray &ray, float t_min, float t_max, SceneObjectRecord &record) const {
	SceneObjectRecord _record;
	bool isIntersect = false;
	float closest = t_max;
	for (int i = 0; i < listSize; i++) {
		if (list[i]->intersect(ray, t_min, closest, _record)) {
			isIntersect = true;
			closest = _record.t;
			record = _record;
		}
	}

	return isIntersect;
}
 