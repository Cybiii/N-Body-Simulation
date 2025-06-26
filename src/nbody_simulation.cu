#include "initial_conditions.h"
#include "nbody_simulation.h"
#include <iomanip>
#include <iostream>
#include <vector>

const float G_scaled = 6.67430e-5f; // Scaled for simulation stability

/**
 * Brute-force O(N²) force calculation kernel
 * Each thread calculates forces on one particle from all other particles
 */
__global__ void calculate_forces_brute_force_kernel(Particle *particles, int N,
                                                    float softening_sq,
                                                    float G_constant) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  float3 force = make_float3(0.0f, 0.0f, 0.0f);
  float3 pos_i = particles[i].position;

  // Calculate force from all other particles
  for (int j = 0; j < N; j++) {
    if (i == j)
      continue; // Skip self-interaction

    float3 pos_j = particles[j].position;
    float mass_j = particles[j].mass;

    // Calculate distance vector
    float dx = pos_j.x - pos_i.x;
    float dy = pos_j.y - pos_i.y;
    float dz = pos_j.z - pos_i.z;

    // Calculate distance squared (with softening)
    float r_sq = dx * dx + dy * dy + dz * dz + softening_sq;
    float r = sqrtf(r_sq);
    float r_cubed = r_sq * r;

    // Calculate force magnitude: F = G * m_i * m_j / r²
    // Force direction: (r_j - r_i) / |r_j - r_i|
    float force_magnitude = G_constant * mass_j / r_cubed;

    force.x += force_magnitude * dx;
    force.y += force_magnitude * dy;
    force.z += force_magnitude * dz;
  }

  // Convert force to acceleration: a = F / m
  float mass_i = particles[i].mass;
  particles[i].acceleration.x = force.x / mass_i;
  particles[i].acceleration.y = force.y / mass_i;
  particles[i].acceleration.z = force.z / mass_i;
}

/**
 * Integration kernel using Leapfrog (Velocity Verlet) integration
 * More stable than Euler integration for orbital mechanics
 */
__global__ void update_particles_kernel(Particle *particles, int N, float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  Particle &p = particles[idx];

  // Leapfrog integration:
  // v(t + dt/2) = v(t) + a(t) * dt/2
  // r(t + dt) = r(t) + v(t + dt/2) * dt
  // v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2

  // For simplicity, we'll use Velocity Verlet:
  // v(t + dt) = v(t) + a(t) * dt
  // r(t + dt) = r(t) + v(t) * dt + 0.5 * a(t) * dt²

  float dt_half = dt * 0.5f;

  // Update velocity: v = v + a * dt
  p.velocity.x += p.acceleration.x * dt;
  p.velocity.y += p.acceleration.y * dt;
  p.velocity.z += p.acceleration.z * dt;

  // Update position: r = r + v * dt
  p.position.x += p.velocity.x * dt;
  p.position.y += p.velocity.y * dt;
  p.position.z += p.velocity.z * dt;
}

/**
 * Wrapper functions for kernel launches
 */
extern "C" void launch_calculate_forces_brute_force(Particle *d_particles,
                                                    int N, float softening_sq,
                                                    float G_constant) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  calculate_forces_brute_force_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_particles, N, softening_sq, G_constant);
  checkCudaError(cudaGetLastError(), "brute force calculation kernel launch");
}

extern "C" void launch_update_particles(Particle *d_particles, int N,
                                        float dt) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  update_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, N,
                                                              dt);
  checkCudaError(cudaGetLastError(), "particle update kernel launch");
}

/**
 * NBodySimulation class implementation
 */
NBodySimulation::NBodySimulation(int num_particles, Algorithm algo, float dt,
                                 float softening, float theta)
    : algorithm(algo), dt(dt), softening(softening), current_timestep(0),
      total_simulation_time(0.0), total_force_calc_time(0.0),
      total_integration_time(0.0) {

  particles = new ParticleSystem(num_particles);
  octree = new Octree(num_particles);
  barnes_hut_force = new BarnesHutForce(theta, softening);

  // Create CUDA events for timing
  checkCudaError(cudaEventCreate(&start_event));
  checkCudaError(cudaEventCreate(&stop_event));

  std::cout << "N-Body Simulation Initialized" << std::endl;
  std::cout << "  Number of particles: " << num_particles << std::endl;
  std::cout << "  Algorithm: "
            << (algorithm == BARNES_HUT ? "Barnes-Hut" : "Brute-Force")
            << std::endl;
}

NBodySimulation::~NBodySimulation() {
  checkCudaError(cudaEventDestroy(start_event));
  checkCudaError(cudaEventDestroy(stop_event));

  delete particles;
  delete octree;
  delete barnes_hut_force;
}

void NBodySimulation::step() {
  checkCudaError(cudaEventRecord(start_event));

  // --- Force Calculation ---
  if (algorithm == BARNES_HUT) {
    // 1. Build the Octree
    octree->build(*particles);

    // 2. Calculate forces using Barnes-Hut
    barnes_hut_force->calculateForces(*particles, *octree, G_scaled);

  } else { // BRUTE_FORCE
    // O(N^2) direct force calculation
    launch_calculate_forces_brute_force(particles->getDeviceParticles(),
                                        particles->getNumParticles(),
                                        softening * softening, G_scaled);
  }

  // --- Integration ---
  // Synchronize to ensure force calculation is complete
  checkCudaError(cudaDeviceSynchronize());

  // Launch kernel to update particle positions and velocities
  launch_update_particles(particles->getDeviceParticles(),
                          particles->getNumParticles(), dt);

  // Synchronize and update timers
  checkCudaError(cudaEventRecord(stop_event));
  checkCudaError(cudaEventSynchronize(stop_event));
  updateTimers();

  current_timestep++;
}

void NBodySimulation::simulate(int num_timesteps) {
  std::cout << "\nStarting simulation for " << num_timesteps << " timesteps..."
            << std::endl;
  last_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_timesteps; ++i) {
    step();
    if ((i + 1) % 10 == 0) {
      std::cout << "Timestep " << (current_timestep) << "/"
                << (current_timestep - i + num_timesteps - 1) << " complete."
                << std::endl;
    }
  }

  printPerformanceStats();
}

void NBodySimulation::setAlgorithm(Algorithm new_algo) {
  if (algorithm != new_algo) {
    algorithm = new_algo;
    std::cout << "Switched simulation algorithm to "
              << (algorithm == BARNES_HUT ? "Barnes-Hut" : "Brute-Force")
              << std::endl;
  }
}

void NBodySimulation::setSoftening(float new_softening) {
  softening = new_softening;
  if (barnes_hut_force) {
    barnes_hut_force->setSoftening(new_softening);
  }
}

void NBodySimulation::updateTimers() {
  float frame_time_ms;
  checkCudaError(cudaEventElapsedTime(&frame_time_ms, start_event, stop_event));
  total_simulation_time += frame_time_ms;

  // Note: More granular timing would require separate events for each stage
  // For now, attributing most of the time to force calculation
  if (algorithm == BARNES_HUT) {
    total_force_calc_time += barnes_hut_force->getLastForceCalcTime() * 1000.0;
  } else {
    total_force_calc_time += frame_time_ms; // Approximate
  }
}

void NBodySimulation::printPerformanceStats() const {
  std::cout << "\n=============== Simulation Performance ==============="
            << std::endl;
  std::cout << "Algorithm: "
            << (algorithm == BARNES_HUT ? "Barnes-Hut" : "Brute-Force")
            << std::endl;
  std::cout << "Total timesteps: " << current_timestep << std::endl;
  std::cout << "Total simulation time: " << total_simulation_time / 1000.0
            << " s" << std::endl;
  std::cout << "Average time per timestep: "
            << total_simulation_time / current_timestep << " ms" << std::endl;
  std.cout << "Average force calculation time: "
           << total_force_calc_time / current_timestep << " ms" << std::endl;

  if (algorithm == BARNES_HUT) {
    barnes_hut_force->printPerformanceStats();
  }

  std::cout << "======================================================"
            << std::endl;
}

void NBodySimulation::benchmark(const std::vector<int> &particle_counts,
                                int timesteps_per_test) {
  std::cout << "\n=== N-Body Simulation Benchmark ===" << std::endl;
  std::cout << std::setw(12) << "Particles" << std::setw(15)
            << "Force (ms/step)" << std::setw(15) << "Integration (ms/step)"
            << std::setw(15) << "Total (ms/step)" << std::setw(15) << "GFLOPS"
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  for (int N : particle_counts) {
    // Create temporary simulation for this particle count
    NBodySimulation temp_sim(N, algorithm, dt, softening, 0.5f);
    InitialConditions::generateBenchmarkParticles(
        *temp_sim.getParticleSystem());
    temp_sim.getParticleSystem()->copyToDevice();

    // Warm up
    temp_sim.step();

    // Reset timers
    temp_sim.total_force_calc_time = 0.0;
    temp_sim.total_integration_time = 0.0;

    // Run benchmark
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timesteps_per_test; i++) {
      temp_sim.step();
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_time =
        std::chrono::duration<double>(end_time - start_time).count();
    double force_time_per_step =
        (temp_sim.total_force_calc_time / timesteps_per_test) * 1000.0; // ms
    double integration_time_per_step =
        (temp_sim.total_integration_time / timesteps_per_test) * 1000.0; // ms
    double total_time_per_step =
        (total_time / timesteps_per_test) * 1000.0; // ms

    // Calculate GFLOPS for force calculation (roughly 20 operations per
    // particle pair)
    double operations_per_step = (double)N * N * 20.0;
    double gflops =
        (operations_per_step / (force_time_per_step / 1000.0)) / 1e9;

    std::cout << std::setw(12) << N << std::setw(15) << std::fixed
              << std::setprecision(3) << force_time_per_step << std::setw(15)
              << std::fixed << std::setprecision(3) << integration_time_per_step
              << std::setw(15) << std::fixed << std::setprecision(3)
              << total_time_per_step << std::setw(15) << std::fixed
              << std::setprecision(2) << gflops << std::endl;
  }
  std::cout << std::string(80, '-') << std::endl;
}