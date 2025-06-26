#pragma once

// --- Graphics and Interop Headers ---

// GLAD must be included first to provide all the modern OpenGL definitions.
#include <glad/glad.h>

// Now, we define the include guard for the old gl.h to prevent it from being
// included. cuda_gl_interop.h will try to include it, which causes conflicts
// with windows.h. By defining this, we trick the compiler into thinking gl.h is
// already present. The definitions it would have provided are already covered
// by glad.h.
#define __gl_h_

// Tell GLFW to not include the old gl.h as well.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Now we can safely include the CUDA-OpenGL interop header.
#include <cuda_gl_interop.h>

// --- Math Library ---
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// --- Standard Library Headers ---
#include <string> // For shader loading
#include <vector>

// --- Project Headers ---
#include "nbody_simulation.h"

// Forward declare GLFW window struct
struct GLFWwindow;

class Renderer {
public:
  Renderer(int width, int height, const char *title);
  ~Renderer();

  bool shouldClose() const;
  void beginFrame();
  void endFrame(float time_scale);
  void renderParticles(ParticleSystem &particles, int particle_count);
  void processInput(NBodySimulation &simulation); // Handles keyboard input

  // Friend declarations for GLFW callbacks
  friend void mouse_callback(GLFWwindow *window, double xpos, double ypos);
  friend void scroll_callback(GLFWwindow *window, double xoffset,
                              double yoffset);

private:
  GLFWwindow *window;
  int width;
  int height;

  // OpenGL state
  GLuint shader_program;
  GLuint vao;

  // CUDA-OpenGL Interop
  GLuint pbo = 0; // OpenGL pixel buffer object
  cudaGraphicsResource_t cuda_pbo_resource = nullptr; // CUDA graphics resource

  void initGLFW();
  void initGLAD();
  void initShaders();
  void setupCallbacks();

  // Timing and FPS
  double delta_time = 0.0;
  double last_frame_time = 0.0;
  int frame_count = 0;
  double last_fps_update_time = 0.0;

  // Basic camera properties
  glm::vec3 camera_pos = glm::vec3(0.0f, 0.0f, 150.0f);
  glm::vec3 camera_front = glm::vec3(0.0f, 0.0f, -1.0f);
  glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  // Camera movement properties
  float camera_move_speed = 15.0f; // Adjusted for better movement
  float camera_zoom = 45.0f;
  float camera_pitch = 0.0f;
  float camera_yaw = -90.0f;
  double last_mouse_x = 0.0, last_mouse_y = 0.0;
  bool first_mouse = true;
};