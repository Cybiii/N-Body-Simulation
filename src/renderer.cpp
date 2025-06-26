#include "renderer.h"
#include "render_kernel.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// Callback for GLFW errors
void glfw_error_callback(int error, const char *description) {
  std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// Forward declaration of callback functions
// These are now C-style free functions, not class methods.
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);

// Helper function to read shader source from a file
std::string readShaderSource(const char *filePath) {
  std::ifstream shaderFile(filePath);
  if (!shaderFile) {
    throw std::runtime_error(std::string("Failed to open shader file: ") +
                             filePath);
  }
  std::stringstream buffer;
  buffer << shaderFile.rdbuf();
  return buffer.str();
}

// Helper function to compile a shader and check for errors
GLuint compileShader(GLenum type, const char *source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    GLint logLength;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<GLchar> log(logLength);
    glGetShaderInfoLog(shader, logLength, NULL, log.data());
    std::string errorMsg = "Shader compilation failed: ";
    errorMsg += log.data();
    glDeleteShader(shader);
    throw std::runtime_error(errorMsg);
  }
  return shader;
}

// Helper function to link shaders into a program
GLuint createShaderProgram(GLuint vertexShader, GLuint fragmentShader) {
  GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);

  GLint success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    GLint logLength;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<GLchar> log(logLength);
    glGetProgramInfoLog(program, logLength, NULL, log.data());
    std::string errorMsg = "Shader program linking failed: ";
    errorMsg += log.data();
    glDeleteProgram(program);
    throw std::runtime_error(errorMsg);
  }
  return program;
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
  glfwSetWindowUserPointer(window, this); // Store 'this' pointer

  initGLAD();
  initShaders();
  setupCallbacks(); // Call the new callback setup function

  glViewport(0, 0, width, height);
  std::cout << "Renderer initialized." << std::endl;
}

Renderer::~Renderer() {
  glDeleteProgram(shader_program);
  glDeleteVertexArrays(1, &vao);

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

void Renderer::initShaders() {
  // Read shader sources
  std::string vertexSource = readShaderSource("shaders/particle.vert");
  std::string fragmentSource = readShaderSource("shaders/particle.frag");

  // Compile shaders
  GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource.c_str());
  GLuint fragmentShader =
      compileShader(GL_FRAGMENT_SHADER, fragmentSource.c_str());

  // Link shaders into a program
  shader_program = createShaderProgram(vertexShader, fragmentShader);

  // Shaders are linked into the program, no need to keep them around
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // Create a VAO (Vertex Array Object) to hold our vertex attribute state
  glGenVertexArrays(1, &vao);
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
  if (particle_count == 0) {
    return;
  }

  // Create the PBO if it doesn't exist
  if (pbo == 0) {
    // Create a VBO (Vertex Buffer Object) and allocate memory
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glBufferData(GL_ARRAY_BUFFER, particle_count * sizeof(float2), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register this VBO with CUDA
    checkCudaError(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                                cudaGraphicsRegisterFlagsNone),
                   "cudaGraphicsGLRegisterBuffer failed");
  }

  // Map the PBO for writing from CUDA
  float2 *d_pbo_buffer = nullptr;
  checkCudaError(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0),
                 "cudaGraphicsMapResources failed");
  size_t num_bytes;
  checkCudaError(cudaGraphicsResourceGetMappedPointer(
                     (void **)&d_pbo_buffer, &num_bytes, cuda_pbo_resource),
                 "cudaGraphicsResourceGetMappedPointer failed");

  // Launch the kernel to copy particle positions
  launch_copy_positions_to_buffer(d_pbo_buffer, particles.getDeviceParticles(),
                                  particle_count);

  // Unmap the PBO
  checkCudaError(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0),
                 "cudaGraphicsUnmapResources failed");

  // --- OpenGL Drawing ---
  glUseProgram(shader_program);

  // Create transformations
  glm::mat4 projection = glm::perspective(
      glm::radians(camera_zoom), (float)width / (float)height, 0.1f, 1000.0f);
  glm::mat4 view =
      glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

  // Get matrix uniform locations
  GLint projLoc = glGetUniformLocation(shader_program, "projection");
  GLint viewLoc = glGetUniformLocation(shader_program, "view");

  // Pass them to the shaders
  glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
  glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, pbo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
  glEnableVertexAttribArray(0);

  glEnable(GL_PROGRAM_POINT_SIZE);
  glDrawArrays(GL_POINTS, 0, particle_count);
  glDisable(GL_PROGRAM_POINT_SIZE);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void Renderer::setupCallbacks() {
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
}

// --- GLFW Callbacks ---

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  Renderer *renderer =
      static_cast<Renderer *>(glfwGetWindowUserPointer(window));
  if (renderer->first_mouse) {
    renderer->last_mouse_x = xpos;
    renderer->last_mouse_y = ypos;
    renderer->first_mouse = false;
  }

  double xoffset = xpos - renderer->last_mouse_x;
  double yoffset = renderer->last_mouse_y -
                   ypos; // reversed since y-coordinates go from bottom to top
  renderer->last_mouse_x = xpos;
  renderer->last_mouse_y = ypos;

  float sensitivity = 0.1f;
  xoffset *= sensitivity;
  yoffset *= sensitivity;

  renderer->camera_yaw += xoffset;
  renderer->camera_pitch += yoffset;

  // Make sure that when pitch is out of bounds, screen doesn't get flipped
  if (renderer->camera_pitch > 89.0f)
    renderer->camera_pitch = 89.0f;
  if (renderer->camera_pitch < -89.0f)
    renderer->camera_pitch = -89.0f;

  glm::vec3 front;
  front.x = cos(glm::radians(renderer->camera_yaw)) *
            cos(glm::radians(renderer->camera_pitch));
  front.y = sin(glm::radians(renderer->camera_pitch));
  front.z = sin(glm::radians(renderer->camera_yaw)) *
            cos(glm::radians(renderer->camera_pitch));
  renderer->camera_front = glm::normalize(front);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  Renderer *renderer =
      static_cast<Renderer *>(glfwGetWindowUserPointer(window));
  renderer->camera_zoom -= (float)yoffset;
  if (renderer->camera_zoom < 1.0f)
    renderer->camera_zoom = 1.0f;
  if (renderer->camera_zoom > 90.0f)
    renderer->camera_zoom = 90.0f;
}