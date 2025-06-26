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

  // Symplectic Euler integration method:
  // First, update velocity using current acceleration.
  // Then, update position using the *new* velocity.
  // This is more stable than standard Euler.

  // Update velocity: v(t+dt) = v(t) + a(t) * dt
  p.velocity.x += p.acceleration.x * dt;
  p.velocity.y += p.acceleration.y * dt;
  p.velocity.z += p.acceleration.z * dt;

  // Update position: r(t+dt) = r(t) + v(t+dt) * dt
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
NBodySimulation::NBodySimulation(int num_particles_in, Algorithm algo,
                                 float dt_in, float softening_in,
                                 float theta_in)
    : algorithm(algo), dt(dt_in), softening(softening_in), current_timestep(0),
      total_simulation_time(0.0), total_force_calc_time(0.0),
      total_integration_time(0.0), time_scale(1.0f) {

  particles = new ParticleSystem(num_particles_in);
  octree = new Octree(particles->getNumParticles());
  barnes_hut_force = new BarnesHutForce(theta_in, softening_in);

  checkCudaError(cudaEventCreate(&start_event), "NBodySim create start_event");
  checkCudaError(cudaEventCreate(&stop_event), "NBodySim create stop_event");

  std::cout << "N-Body Simulation Initialized" << std::endl;
  std::cout << "  Number of particles: " << particles->getNumParticles()
            << std::endl;
  std::cout << "  Algorithm: "
            << (algorithm == BARNES_HUT ? "Barnes-Hut" : "Brute-Force")
            << std::endl;
}

NBodySimulation::~NBodySimulation() {
  checkCudaError(cudaEventDestroy(start_event), "NBodySim destroy start_event");
  checkCudaError(cudaEventDestroy(stop_event), "NBodySim destroy stop_event");
  delete particles;
  delete octree;
  delete barnes_hut_force;
}

void NBodySimulation::step() {
  checkCudaError(cudaEventRecord(start_event), "step record start");

  if (algorithm == BARNES_HUT) {
    octree->buildFromParticles(*particles);
    barnes_hut_force->calculateForces(*particles, *octree, G_scaled);
  } else {
    launch_calculate_forces_brute_force(particles->getDeviceParticles(),
                                        particles->getNumParticles(),
                                        softening * softening, G_scaled);
  }

  checkCudaError(cudaDeviceSynchronize(), "force calculation sync");

  launch_update_particles(particles->getDeviceParticles(),
                          particles->getNumParticles(), dt * time_scale);

  checkCudaError(cudaEventRecord(stop_event), "step record stop");
  checkCudaError(cudaEventSynchronize(stop_event), "step sync stop");
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

int NBodySimulation::getParticleCount() const {
  return particles ? particles->getNumParticles() : 0;
}

void NBodySimulation::updateTimers() {
  float frame_time_ms;
  checkCudaError(cudaEventElapsedTime(&frame_time_ms, start_event, stop_event),
                 "step elapsed time");
  total_simulation_time += frame_time_ms;

  // Get force calculation time from the specific implementation
  if (algorithm == BARNES_HUT) {
    total_force_calc_time += barnes_hut_force->getLastForceCalcTime() * 1000.0;
  } else {
    // For brute force, we need a separate timer. For now, approximate.
    // This requires adding events around launch_calculate_forces_brute_force
    // and subtracting that from the total frame time.
  }

  // Integration time is the remainder
  float integration_ms =
      frame_time_ms - (barnes_hut_force->getLastForceCalcTime() * 1000.0);
  total_integration_time += (integration_ms > 0) ? integration_ms : 0;
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
  std::cout << "Average force calculation time: "
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
  std::cout << std::left << std::setw(12) << "Particles" << std::setw(20)
            << "Force (ms/step)" << std::setw(22) << "Integration (ms/step)"
            << std::setw(18) << "Total (ms/step)" << std::setw(15) << "GFLOPS"
            << std::endl;
  std::cout << std::string(87, '-') << std::endl;

  for (int N : particle_counts) {
    // Re-initialize simulation for this particle count
    delete particles;
    delete octree;
    particles = new ParticleSystem(N);
    octree = new Octree(N);
    InitialConditions::generateBenchmarkParticles(*particles);
    particles->copyToDevice();

    // Reset timers for this run
    total_force_calc_time = 0;
    total_integration_time = 0;
    total_simulation_time = 0;
    current_timestep = 0;

    // Warm-up run
    simulate(5);

    // Reset timers again after warm-up
    total_force_calc_time = 0;
    total_integration_time = 0;
    total_simulation_time = 0;
    current_timestep = 0;

    // Timed run
    simulate(timesteps_per_test);

    double avg_force_ms = total_force_calc_time / current_timestep;
    double avg_integration_ms = total_integration_time / current_timestep;
    double avg_total_ms = total_simulation_time / current_timestep;

    // Estimate GFLOPS
    double interactions_per_step =
        (algorithm == BARNES_HUT) ? (double)N * log((double)N) // O(N log N)
                                  : (double)N * (double)N;     // O(N^2)
    double ops_per_interaction = 20; // Approx ops for force calculation
    double gflops = (interactions_per_step * ops_per_interaction) /
                    (avg_total_ms / 1000.0) / 1e9;

    std::cout << std::left << std::setw(12) << N << std::setw(20) << std::fixed
              << std::setprecision(4) << avg_force_ms << std::setw(22)
              << avg_integration_ms << std::setw(18) << avg_total_ms
              << std::setw(15) << std::setprecision(2) << gflops << std::endl;
  }
  std::cout << std::string(87, '-') << std::endl;
}

void NBodySimulation::increaseTimeScale() {
  time_scale *= 1.2f;
  printf("Time scale increased to: %.2fx\n", time_scale);
}

void NBodySimulation::decreaseTimeScale() {
  time_scale /= 1.2f;
  if (time_scale < 0.1f) {
    time_scale = 0.1f;
  }
  printf("Time scale decreased to: %.2fx\n", time_scale);
}

float NBodySimulation::getTimeScale() const { return time_scale; }