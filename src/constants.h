#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace Constants {
    extern const float M;           // Mass of disc
    extern const float I;           // Rotational inertia
    extern const float D;           // Diameter
    extern const float H;           // Height
    extern const float L;           // Leg link length
    extern const float SPRING_K;    // Stance anchor spring stiffness
    extern const float SPRING_D;    // Stance anchor spring damping
    extern const float PD_KP;       // Body velocity PD proportional gain
    extern const float KP_HEADING;  // Heading error (rad) → target omega (rad/s)
    extern const float KP_ANG;      // Angular velocity error → angular accel × I
    extern const float DT;          // Physics timestep
    extern const float REACH_MAX;        // Maximum leg reach
    extern const float LEG_ZONE_CENTER_R; // Body-frame radius of each leg's placement zone centre
    extern const float LEG_ZONE_RADIUS;   // Radius of each circular placement zone
    extern const int   SWING_STEPS;       // Physics steps per swing phase (= 0.2s / DT)
}

#endif
