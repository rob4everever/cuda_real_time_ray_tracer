/*
* author: 100086865
*
* Various textures are defined here inclusing sold colours and 
* a checkered texture.
*/

#pragma once

#include <glm/glm.hpp>

/*
* Texture base class
*/
class Texture {

public:
	/*
	* virtual method that calculates the colour value of a texture
	*/
	__device__ virtual glm::vec3 value(float u, float v, const glm::vec3 &p) const = 0;
};

/*
* Represents a solid coloured texture
*/
class ColouredTexture : public Texture {

public:

	glm::vec3 textureColour;

	/*
	* Default ColouredTexture constructor
	*/
	__device__ ColouredTexture() { }

	/*
	* Creates a new ColouredTexture
	* @param: texture colour
	*/
	__device__ ColouredTexture(glm::vec3 c) : textureColour(c) { }
	
	/*
	* Returns a textures colour
	* @param: u
	* @param: v
	* @param: point
	*/
	__device__ virtual glm::vec3 value(float u, float v, const glm::vec3& p) const {
		return textureColour;
	}
};

/*
* Represents a checkered texture which is used as the 
* floor of one of my scenes
*/
class CheckeredTexture : public Texture {

public:

	Texture *odd;
	Texture *even;

	/*
	* Default CheckeredTexture constructor
	*/
	__device__ CheckeredTexture() { }

	/*
	* Creates a new ColouredTexture
	* @param: texture 1
	* @param: texture 2
	*/
	__device__ CheckeredTexture(Texture *t0, Texture *t1) : even(t0), odd(t1) { }

	/*
	* Creates a checked texture by alternating between two colours
	* @param: u
	* @param: v
	* @param: point
	*/
	__device__ virtual glm::vec3 value(float u, float v, const glm::vec3& p) const {
		float sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
		if (sines < 0) {
			return odd->value(u, v, p);
		}
		else {
			return even->value(u, v, p);
		}
	}
};