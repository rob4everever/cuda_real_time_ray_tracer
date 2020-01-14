/*
* author: 100086865
*
* This class defines the 3 rectangles needed to create Cornell's box
* On top of that, a function to flip the normals of the rectangles is
* also provided.
*/

#pragma once

#include "SceneObject.cuh" 

/*
* Represents an XY rectangle of the Cornell's box
*/
class XY_Rectangle : public SceneObject {

public:
	float x0, x1, y0, y1, k;
	Material *material;
	
	__device__ XY_Rectangle() {}
	__device__ XY_Rectangle(float x0, float x1, float y0, float y1, float k, Material *material) : x0(x0), x1(x1), y0(y0), y1(y1), k(k), material(material){}
	__device__ virtual bool intersect(const Ray &ray, float t_min, float t_max, SceneObjectRecord &record) const {
		
		float t = (k - ray.origin.z) / ray.direction.z;
		
		if (t < t_min || t > t_max) {
			return false;
		}

		float x = ray.origin.x + t * ray.direction.x;
		float y = ray.origin.y + t * ray.direction.y;

		if (x < x0 || x > x1 || y < y0 || y > y1) {
			return false;
		}

		record.u = (x - x0) / (x1 - x0);
		record.v = (y - y0) / (y1 - y0);
		record.t = t;
		record.material = material;
		record.point = ray.pointOnRay(t);
		record.normal = glm::vec3(0, 0, 1);
		return true;
	}
};

/*
* Represents an XZ rectangle of the Cornell's box
*/
class XZ_Rectangle : public SceneObject {
public:
	__device__ XZ_Rectangle() {}
	__device__ XZ_Rectangle(float _x0, float _x1, float _z0, float _z1, float _k, Material *mat) : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {};
	__device__ virtual bool intersect(const Ray& ray, float t0, float t1, SceneObjectRecord& rec) const {
		float t = (k - ray.origin.y) / ray.direction.y;
		if (t < t0 || t > t1)
			return false;
		float x = ray.origin.x + t * ray.direction.x;
		float z = ray.origin.z + t * ray.direction.z;
		if (x < x0 || x > x1 || z < z0 || z > z1)
			return false;
		rec.u = (x - x0) / (x1 - x0);
		rec.v = (z - z0) / (z1 - z0);
		rec.t = t;
		rec.material = mp;
		rec.point = ray.pointOnRay(t);
		rec.normal = glm::vec3(0, 1, 0);
		return true;
	}
	
	Material  *mp;
	float x0, x1, z0, z1, k;
};

/*
* Represents an YZ rectangle of the Cornell's box
*/
class YZ_Rectangle : public SceneObject {
public:
	__device__ YZ_Rectangle() {}
	__device__ YZ_Rectangle(float _y0, float _y1, float _z0, float _z1, float _k, Material *mat) : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {};
	__device__ virtual bool intersect(const Ray& ray, float t0, float t1, SceneObjectRecord& rec) const {
		float t = (k - ray.origin.x) / ray.direction.x;
		if (t < t0 || t > t1)
			return false;
		float y = ray.origin.y + t * ray.direction.y;
		float z = ray.origin.z + t * ray.direction.z;
		if (y < y0 || y > y1 || z < z0 || z > z1)
			return false;
		rec.u = (y - y0) / (y1 - y0);
		rec.v = (z - z0) / (z1 - z0);
		rec.t = t;
		rec.material = mp;
		rec.point = ray.pointOnRay(t);
		rec.normal = glm::vec3(1, 0, 0);
		return true;
	}
	
	Material  *mp;
	float y0, y1, z0, z1, k;
};

/*
* A class that flips the normal vector of a rectangle
*/
class FlipNormals : public SceneObject {
public:
	
	SceneObject *p;

	__device__ FlipNormals(SceneObject *obj) : p(obj){}
	__device__ virtual bool intersect(const Ray &ray, float t_min, float t_max, SceneObjectRecord &record) const {
		if (p->intersect(ray, t_min, t_max, record)) {
			record.normal = -record.normal;
			return true;
		}
		else {
			return false;	
		}
	}
};