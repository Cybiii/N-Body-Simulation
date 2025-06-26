#include "renderer.h"
#include <iostream>
#include <stdexcept>

// Callback for GLFW errors
void glfw_error_callback(int error, const char *description) {
  std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

Renderer::Renderer(int width, int height, const char *title)
    : width(width), height(height), window(nullptr) {
  initGLFW();

  // Create window
  window = glfwCreateWindow(width, height, title, NULL, NULL);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }
  glfwMakeContextCurrent(window);

  initGLAD();

  glViewport(0, 0, width, height);
  std::cout << "Renderer initialized." << std::endl;
}

Renderer::~Renderer() {
  if (window) {
    glfwDestroyWindow(window);
  }
  glfwTerminate();
  std::cout << "Renderer destroyed." << std::endl;
}

void Renderer::initGLFW() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void Renderer::initGLAD() {
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    throw std::runtime_error("Failed to initialize GLAD");
  }
}

bool Renderer::shouldClose() const { return glfwWindowShouldClose(window); }

void Renderer::beginFrame() {
  glClearColor(0.0f, 0.0f, 0.05f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::endFrame() {
  glfwSwapBuffers(window);
  glfwPollEvents();
}

// Stub for now
void Renderer::renderParticles(ParticleSystem &particles, int particle_count) {
  // Rendering logic will go here
}