#include "constants.h"
#include <cmath>

namespace Constants {
    const float M        = 2.0f;
    const float I        = 0.5f;
    const float D        = 1.0f;
    const float H        = 0.2f;
    const float L        = 0.4f;

    // Spring-damper stance anchor.
    // K=500, D=30 → natural freq ~15.8 rad/s, damping ratio ~0.47.
    // RK4 stable at DT=0.005 (stability margin ≈35×).
    const float SPRING_K = 500.0f;
    const float SPRING_D = 30.0f;

    // PD gain for body velocity tracking.
    // F_pd = PD_KP * (ref_vel - vel) * M → at typical errors this is 2–8 N.
    const float PD_KP     = 8.0f; // was 8.0f;

    // Two-stage angular PD:
    //   ref_omega  = KP_HEADING * heading_error   (rad → rad/s)
    //   torque_pd  = KP_ANG * (ref_omega - omega) * I
    // KP_HEADING=2 → 1 rad misalignment → 2 rad/s target spin.
    // KP_ANG=4     → matches translational PD gain; torque ≈ 2 N·m at 1 rad/s error.
    const float KP_HEADING = 4.0f; // was 2.0f;
    const float KP_ANG     = 8.0f; // waas 4.0f;

    const float DT       = 0.005f;

    // REACH_MAX = D/2 + sqrt(1 - (H/(2L))^2)
    const float REACH_MAX = (D / 2.0f)
                          + std::sqrt(1.0f - std::pow(H / (2.0f * L), 2.0f));

    // Per-leg circular placement zones (body frame).
    // Zone i is centred at angle i*2π/5, radius LEG_ZONE_CENTER_R.
    // Geometry guarantees:
    //   - Adjacent zones don't touch  (gap ≈ 0.40)
    //   - Zones clear the body disc   (inner edge 0.55 > body radius 0.5)
    //   - Zones fit within leg reach  (outer edge 1.15 < REACH_MAX ≈ 1.47)
    const float LEG_ZONE_CENTER_R = 0.85f;
    const float LEG_ZONE_RADIUS   = 0.30f;

    // 40 steps × 0.005 s = 0.2 s per swing phase.
    const int SWING_STEPS = 40;
}
