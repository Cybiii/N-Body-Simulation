#include "nbody_simulation.h"
#include "renderer.h"

void run_simulation_with_gui(NBodySimulation &sim) {
  // Create renderer
  Renderer renderer(1280, 720, "N-Body Simulation");

  // Main loop
  while (!renderer.shouldClose()) {
    renderer.beginFrame();

    // Run one simulation step
    sim.step();

    // Render particles (this is still a stub)
    renderer.renderParticles(*sim.getParticleSystem(),
                             sim.getParticleSystem()->getNumParticles());

    renderer.endFrame();
  }
}