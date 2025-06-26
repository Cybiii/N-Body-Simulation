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

  std::cout << "Attempting to initialize renderer..." << std::endl;

  std::cout << "Initializing GLFW..." << std::endl;
  initGLFW();
  std::cout << "GLFW Initialized." << std::endl;

  std::cout << "Creating GLFW window..." << std::endl;
  window = glfwCreateWindow(width, height, title, NULL, NULL);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }
  glfwMakeContextCurrent(window);
  std::cout << "GLFW window created." << std::endl;

  glfwSetWindowUserPointer(
      window, this); // Associate this Renderer instance with the window

  std::cout << "Initializing GLAD..." << std::endl;
  initGLAD();
  std::cout << "GLAD Initialized." << std::endl;

  std::cout << "Initializing Shaders..." << std::endl;
  initShaders();
  std::cout << "Shaders Initialized." << std::endl;

  std::cout << "Setting up callbacks..." << std::endl;
  setupCallbacks(); // This includes mouse and scroll callbacks
  std::cout << "Callbacks set up." << std::endl;

  // Initialize timing
  last_frame_time = glfwGetTime();
  last_fps_update_time = last_frame_time;

  // Set the initial viewport size
  glViewport(0, 0, width, height);

  std::cout << "Renderer initialized successfully." << std::endl;
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
  // Calculate delta time
  double current_time = glfwGetTime();
  delta_time = current_time - last_frame_time;
  last_frame_time = current_time;

  // Handle user input - This is now called from main loop
  // processInput();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::endFrame(float time_scale) {
  // FPS calculation
  frame_count++;
  double current_time = glfwGetTime();
  if (current_time - last_fps_update_time >= 1.0) { // Update every second
    char title_buffer[256];
    sprintf(title_buffer,
            "N-Body Simulation | %d Particles | FPS: %d | Time Scale: %.2fx",
            8192, frame_count, time_scale);
    glfwSetWindowTitle(window, title_buffer);
    frame_count = 0;
    last_fps_update_time = current_time;
  }

  // Swap buffers and poll events
  glfwSwapBuffers(window);
  glfwPollEvents();

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void Renderer::renderParticles(ParticleSystem &particles, int particle_count) {
  if (particle_count == 0) {
    return;
  }

  // Create the VBO if it doesn't exist
  if (vbo == 0) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Allocate space for interleaved positions (float4) and colors (float4)
    glBufferData(GL_ARRAY_BUFFER, particle_count * 2 * sizeof(float4), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register this VBO with CUDA
    checkCudaError(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                cudaGraphicsRegisterFlagsNone),
                   "cudaGraphicsGLRegisterBuffer failed");
  }

  // Map the VBO for writing from CUDA
  float4 *d_vbo_buffer = nullptr;
  checkCudaError(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0),
                 "cudaGraphicsMapResources failed");
  size_t num_bytes;
  checkCudaError(cudaGraphicsResourceGetMappedPointer(
                     (void **)&d_vbo_buffer, &num_bytes, cuda_vbo_resource),
                 "cudaGraphicsResourceGetMappedPointer failed");

  // Launch the kernel to compute colors and fill the buffer
  // TODO: Make max_velocity_sq dynamic
  float max_velocity_sq = 0.5f;
  launch_compute_colors_and_interleave(d_vbo_buffer,
                                       particles.getDeviceParticles(),
                                       particle_count, max_velocity_sq);

  // Unmap the VBO
  checkCudaError(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0),
                 "cudaGraphicsUnmapResources failed");

  // --- OpenGL Drawing ---
  glUseProgram(shader_program);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  // Vertex attribute for position (location = 0)
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(float4),
                        (void *)0);

  // Vertex attribute for color (location = 1)
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4),
                        (void *)sizeof(float4));

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

  // Draw the particles
  glDrawArrays(GL_POINTS, 0, particle_count);

  // Clean up state
  glBindVertexArray(0);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
}

void Renderer::setupCallbacks() {
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  // Disable cursor to hide it and lock it to the window
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void Renderer::processInput(NBodySimulation &simulation) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  float current_move_speed = camera_move_speed * (float)delta_time;

  // Forward/backward
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera_pos += camera_front * current_move_speed;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera_pos -= camera_front * current_move_speed;

  // Left/right (strafe)
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera_pos -= glm::normalize(glm::cross(camera_front, camera_up)) *
                  current_move_speed;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera_pos += glm::normalize(glm::cross(camera_front, camera_up)) *
                  current_move_speed;

  // Up/down
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    camera_pos += camera_up * current_move_speed;
  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    camera_pos -= camera_up * current_move_speed;

  // Time scale controls
  if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) { // '+' key
    simulation.increaseTimeScale();
  }
  if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) { // '-' key
    simulation.decreaseTimeScale();
  }
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