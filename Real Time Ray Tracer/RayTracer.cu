/*
* author: 100086865
*
* The main file of the ray tracer. GLFW, OpenGL and CUDA are initialised here.
* The ray tracer is configured, executed and the results are rendered to the
* window.
*/

#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include "cuda_gl_interop.h"

#include <fstream>
#include <iostream>
#include <time.h>

#include "Camera.cuh"
#include "Material.cuh"
#include "Ray.cuh"
#include "Rectangle.cuh"
#include "Scene.cuh"
#include "SceneObject.cuh"
#include "Sphere.cuh"
#include "Texture.cuh"

//Screen dimensions
const unsigned int SCR_WIDTH = 512;
const unsigned int SCR_HEIGHT = 256;

//CUDA/OpenGL interop resources
GLuint viewGLTexture;
cudaGraphicsResource_t viewCudaResource;
float* deviceRes = NULL;

//Callbacks
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
GLFWwindow* initGLFW();
void initGL();
void initCuda();
void callCudaKernel(cudaSurfaceObject_t image);

//Camera settings
glm::vec3 camLookFrom = glm::vec3(-8, 4, 4);
glm::vec3 camLookAt = glm::vec3(0, 0, -1);

//Current scene
int sceneNumber = 2;

//macro to output cudaError_t result to stdout
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
static void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << " " << cudaGetErrorString(result) << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

/*
* Determines the colour value of a pixel
* @param: Ray at which to determine the pixel colour
* @param: Scene to fire the ray into
* @param: RandomState
* @return: RGB colour value
*/
__device__ glm::vec3 calculateColour(const Ray& ray, SceneObject** world, curandState* local_rand_state) {

	Ray currentRay = ray;
	glm::vec3 currentAttenuation = glm::vec3(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 50; i++) {

		SceneObjectRecord record;
		//intersect
		if ((*world)->intersect(currentRay, 0.001f, FLT_MAX, record)) {
			Ray scattered;
			glm::vec3 attenuation;
			if (record.material->scatter(currentRay, record, attenuation, scattered, local_rand_state)) {
				currentAttenuation *= attenuation;
				currentRay = scattered;
			}
			else {
				return glm::vec3(0, 0, 0);
			}
		}
		//background colour
		else {
			glm::vec3 unitDir = glm::normalize(currentRay.direction);
			float t = 0.5f * (unitDir.y + 1.0f);
			glm::vec3 x = (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
			return currentAttenuation * x;
		}
	}
}

/*
* Initialises the random state for each pixel
* @param: Max x coordinate
* @param: Max y coordinate
* @param: RandomState
*/
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	const int pixelIndex = (j * max_x + i);
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixelIndex, 0, &rand_state[pixelIndex]);
}

/*
* Caclautes an image using the ray tracer and writes the output to a cuda surface
* @param: Cuda surface object to be written to
* @param: Image x
* @param: Image y
* @param: Number of samples to do
* @param: Scene camera
* @param: Scene
* @param: RandomState
* @param: device Res
*/
__global__ void render(cudaSurfaceObject_t image, int x, int y, int samples, Camera** camera, SceneObject** scene, curandState* rand_state, float* deviceRes) {

	//coordinates of each thread
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= x) || (j >= y)) return;

	//current pixel
	int pixelIndex = j * x + i;
	curandState local_rand_state = rand_state[pixelIndex];

	//samples the colour
	glm::vec3 colour = glm::vec3(0.0f, 0.0f, 0.0f);
	for (int s = 0; s < samples; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(y);
		//Cast a ray into the center of the pixel
		Ray ray = (*camera)->getRay(u, v);
		colour += calculateColour(ray, scene, &local_rand_state);
	}

	//average the colour
	rand_state[pixelIndex] = local_rand_state;
	colour.x = colour.x / samples;
	colour.y = colour.y / samples;
	colour.z = colour.z / samples;

	//square rooting the result to solve gamma correction issues
	colour = glm::vec3(sqrt(colour.x), sqrt(colour.y), sqrt(colour.z));

	//write to the surface object
	uchar4 data = make_uchar4(255.99 * colour.x, 255.99 * colour.y, 255.99 * colour.z, 1);
	surf2Dwrite(data, image, i * sizeof(uchar4), j, cudaBoundaryModeClamp);
}

/*
* Initialises a scene of objects
* @param: List of scene objects
* @param: World to store the scene
* @param: Scene camera
* @param: Image x
* @param: Image y
*/
__global__ void createWorld(SceneObject** d_list, SceneObject** d_world, Camera** camera, int x, int y, glm::vec3 c1, glm::vec3 c2, int sceneNumber) {

	if (threadIdx.x == 0 && blockIdx.x == 0) {

		if (sceneNumber == 1) {
			Material* light = new Light(new ColouredTexture(glm::vec3(15.0f, 15.0f, 15.0f)));
			Texture* checkedTexture = new  CheckeredTexture(new ColouredTexture(glm::vec3(0, 0, 0)), new ColouredTexture(glm::vec3(1, 1, 1)));
			d_list[0] = new Sphere(glm::vec3(0, 0, -1), 0.5f, new Diffuse(new ColouredTexture(glm::vec3(0.4f, 0.7f, 0.1f))));
			d_list[1] = new Sphere(glm::vec3(0, -1000.5, -1), 1000.0f, new Diffuse(checkedTexture));
			d_list[2] = new Sphere(glm::vec3(1, 0, -1), 0.5f, new Specular(glm::vec3(0.8f, 0.6f, 0.2f)));
			d_list[3] = new Sphere(glm::vec3(-1, 0, -1), 0.5f, new Fresnel(1.5));
			d_list[4] = new Sphere(glm::vec3(-1, 0, -1), -0.45f, new Fresnel(1.5));
			d_list[5] = new XZ_Rectangle(446 * 2, 1046 * 4, 494 * 4, 1024 * 4, 1200 * 4, light);
			*d_world = new Scene(d_list, 6);
			*camera = new Camera(glm::vec3(c1.x, 4, 4), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0), 40.0f, float(x) / float(y));
		}
		if (sceneNumber == 2) {
			int i = 0;
			Material* red = new Diffuse(new ColouredTexture(glm::vec3(0.65f, 0.05f, 0.05f)));
			Material* white = new Diffuse(new ColouredTexture(glm::vec3(0.73f, 0.73f, 0.73f)));
			Material* green = new Diffuse(new ColouredTexture(glm::vec3(0.12f, 0.45f, 0.15f)));
			Material* purple = new Diffuse(new ColouredTexture(glm::vec3(0.52f, 0.25f, 0.85f)));
			Material* orange = new Diffuse(new ColouredTexture(glm::vec3(0.72f, 0.35f, 0.15f)));

			Material* r = new Specular(glm::vec3(0.82f, 0.35f, 0.24f));
			Material* reflectiveblue = new Specular(glm::vec3(0.24f, 0.35f, 0.9f));
			Material* reflectiveYellow = new Specular(glm::vec3(0.99f, 0.98f, 0.1f));

			Material* light = new Light(new ColouredTexture(glm::vec3(15.0f, 15.0f, 15.0f)));

			//cornells box
			d_list[i++] = new FlipNormals(new  YZ_Rectangle(0.0f, 1000.0f, 0.0f, 1000.0f, 1000.0f, green));
			d_list[i++] = new YZ_Rectangle(0.0f, 1000.0f, 0.0f, 1000.0f, 0.0f, red);
			d_list[i++] = new XZ_Rectangle(0.0f, 1000.0f, 0.0f, 1000.0f, 1000.0f, white);						//roof
			d_list[i++] = new XZ_Rectangle(0.0f, 1000.0f, 0.0f, 1000.0f, 0.0f, white);						//floor
			d_list[i++] = new FlipNormals(new XY_Rectangle(0, 1000.0f, 0, 1000.0f, 1000.0f, white));

			d_list[i++] = new Sphere(glm::vec3(180.0f, 120.0f, 378.0f), 120.0f, reflectiveblue);
			d_list[i++] = new Sphere(glm::vec3(400.0f, 120.0f, 198.0f), 120.0f, purple);
			d_list[i++] = new Sphere(glm::vec3(700.0f, 60.0f, 90.0f), 60.0f, orange);
			d_list[i++] = new Sphere(glm::vec3(840.0f, 60.0f, 90.0f), 60.0f, reflectiveYellow);
			d_list[i++] = new Sphere(glm::vec3(720.0f, 120.0f, 621.0f), 120.0f, new Fresnel(1.50f));
			d_list[i++] = new Sphere(glm::vec3(720.0f, 120.0f, 621.0f), 115.0f, new Fresnel(1.50f));
			d_list[i++] = new Sphere(glm::vec3(170.0f, 50.0f, 170.0f), 50.0f, new Fresnel(1.50f));
			d_list[i++] = new Sphere(glm::vec3(170.0f, 50.0f, 170.0f), 45.0f, new Fresnel(1.50f));
			d_list[i++] = new XZ_Rectangle(446 * 2, 1046 * 4, 494 * 4, 1024 * 4, 1200 * 4, light);

			*d_world = new Scene(d_list, i);
			*camera = new Camera(glm::vec3(0.0f + c1.x, 278.0f, -800.0f), glm::vec3(0.0f + c2.x, 278.0, 0.0f), glm::vec3(0, 1, 0), 40.0f, float(x) / float(y));
		}
	}
}

/*
* Deletes the scene of objects
* @param: List of scene objects
* @param: The scene
* @param: Camera
*/
__global__ void clearScene(SceneObject** d_list, SceneObject** d_scene, Camera** d_camera) {

	for (int i = 0; i < 14; i++) {
		delete d_list[i];
	}

	delete* d_scene;
	delete* d_camera;
}

/*
* Initialises the GLFW window
* @return - window object
*/
GLFWwindow* initGLFW() {

	//GLFW configuration
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	//GLFW window
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Real Time Ray Tracer", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return NULL;
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, key_callback);

	return window;
}

/*
* Initialises OpenGL and create a new texture
* to store the rendered image
*/
void initGL() {

	//GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
	}

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &viewGLTexture);
	glBindTexture(GL_TEXTURE_2D, viewGLTexture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

/*
* Initialises cuda and allocates memory on the GPU
* for the writable surface object
*/
void initCuda() {
	checkCudaErrors(cudaGLSetGLDevice(0));
	checkCudaErrors(cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	checkCudaErrors(cudaMalloc((void**)&deviceRes, SCR_WIDTH * SCR_HEIGHT * 4 * sizeof(float)));
}

/*
* Sets up the ray tracer and calls the CUDA kernel that
* begins the ray tracing
* @param: surface object to be writtent to
*/
void callCudaKernel(cudaSurfaceObject_t image) {

	//number of samples per pixel
	int samples = 50;

	//allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, SCR_WIDTH * SCR_HEIGHT * sizeof(curandState)));

	//define number of threads per block
	int xThreads = 8;
	int yThreads = 8;

	//List of objects
	SceneObject** d_list;
	checkCudaErrors(cudaMallocManaged((void**)&d_list, 14 * sizeof(SceneObject*)));

	//Complete scene
	SceneObject** d_scene;

	//Camera
	Camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
	checkCudaErrors(cudaMallocManaged((void**)&d_scene, sizeof(SceneObject*)));

	//different camera movements for different scenes
	if (sceneNumber == 1) {
		if (camLookFrom.x <= 20.0f) {
			camLookFrom += 0.2f;
		}
		else {
			camLookFrom.x = -20.0f;
		}
	}
	if (sceneNumber == 2) {

		if (camLookFrom.x >= 1000.0f) {
			camLookFrom.x = 0.0f;
			camLookAt.x = 0.0f;
		}
		else {
			camLookAt += 5.0f;
			camLookFrom += 5.0f;
		}
	}

	//initialise the world
	createWorld << <1, 1 >> > (d_list, d_scene, d_camera, SCR_WIDTH, SCR_HEIGHT, camLookFrom, camLookAt, sceneNumber);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//initialise blocks and threads
	dim3 blocks((SCR_WIDTH - 1) / xThreads + 1, (SCR_HEIGHT - 1) / yThreads + 1, 1);
	dim3 threads(xThreads, yThreads, 1);

	//init render
	render_init << <blocks, threads >> > (SCR_WIDTH, SCR_HEIGHT, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	//threads launch in blocks. block has 64 threads running the function
	render << <blocks, threads >> > (image, SCR_WIDTH, SCR_HEIGHT, samples, d_camera, d_scene, d_rand_state, deviceRes);				//start ray tracing
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	//delete scene objects
	clearScene << <1, 1 >> > (d_list, d_scene, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_scene));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
}

/*
* Deletes cuda resources
*/
void deInit() {
	checkCudaErrors(cudaFree(deviceRes));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaDeviceReset());
}

/*
* Initialises OpenGL, CUDA and GLFW as well as holds
* the main render loop for the application
*/
int main() {

	//setup 
	GLFWwindow* window = initGLFW();
	initGL();
	initCuda();

	//choose a scene
	std::cout << "Please choose a scene: " << std::endl;
	std::cout << "1) 3 spheres" << std::endl;
	std::cout << "2) Cornell's Box" << std::endl;
	std::cin >> sceneNumber;

	//Game loop
	while (!glfwWindowShouldClose(window)) {

		//poll events
		glfwPollEvents();

		//clear colour
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glColor3f(1.0f, 1.0f, 1.0f);

		//map cuda resource
		checkCudaErrors(cudaGraphicsMapResources(1, &viewCudaResource));

		cudaArray_t viewCudaArray;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));

		cudaResourceDesc viewCudaArrayResourceDesc;
		memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

		//create a writable cuda surface
		cudaSurfaceObject_t viewCudaSurfaceObject;
		checkCudaErrors(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

		//init the ray tracer and call the kernel
		callCudaKernel(viewCudaSurfaceObject);

		//clean up
		checkCudaErrors(cudaDestroySurfaceObject(viewCudaSurfaceObject));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &viewCudaResource));
		checkCudaErrors(cudaStreamSynchronize(0));

		//bind texture
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, viewGLTexture);

		//draw textured quad
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
		glEnd();

		//unbind texture
		glBindTexture(GL_TEXTURE_2D, 0);
		glFinish();

		//Swap buffers
		glfwSwapBuffers(window);
	}

	//deallocate resources
	deInit();
	glfwTerminate();
	return 0;
}

/*
* Processes keyboard input
*/
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {

	//ESC: closes the application
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

/*
* Resize viewport
*/
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}