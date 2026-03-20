#ifndef ROBOT_H
#define ROBOT_H

#include <vector>
#include <array>
#include <glm/glm.hpp>

struct State {
    glm::vec2 pos;
    float theta;
    glm::vec2 vel;
    float omega;
};

// One of the 5 legs.
struct Leg {
    glm::vec2 anchor_pos;     // Fixed hip position in body frame
    glm::vec2 foot_pos;       // Current foot position in body frame
    glm::vec2 foot_world_pos; // World-frame reference used for spring or idle tracking
    float mount_angle;
};

// ============================================================================
// Robot — always-3-on-the-ground tripod walker
//
// Roles at every moment:
//   stance_ids[3] — three legs whose world anchors are locked by spring-damper.
//   swing_id      — one leg interpolating in world space toward swing_world_target.
//   idle_id       — one airborne leg that just retired; drifts at its world pos.
//
// Gait cycle (initiated by calling initiateStep at each step event):
//   1. Swing leg reaches target → lands.              [update() returns true]
//   2. Caller validates retire slot (prospectiveMargins) and calls initiateStep().
//   3. Landed leg joins stance (locked anchor).
//   4. stance_ids[retire_slot] is lifted → becomes idle.
//   5. Old idle leg begins swinging toward the new RL-chosen world target.
//   Repeat — always exactly 3 on the ground.
//
// Observation (17D, must match RLwalkingtraining._get_obs exactly):
//   [vx_b/2, vy_b/2,
//    ref_vx_b/0.6, ref_vy_b/0.6,
//    s0x/R, s0y/R, s1x/R, s1y/R, s2x/R, s2y/R,   ← old stance legs
//    lx/R, ly/R,                                   ← just-landed leg
//    ix/R, iy/R,                                   ← idle leg
//    pm0, pm1, pm2]                                ← prospective margins
//
// Action (5D):
//   [r0, r1, r2,   argmax → which of stance_ids to retire
//    tx, ty]       target for new swing (body frame, normalised by REACH_MAX)
// ============================================================================
class Robot {
public:
    State state;
    std::vector<Leg> legs;       // exactly 5

    std::array<int,3> stance_ids;
    int swing_id;
    int idle_id;
    int just_landed_id;          // set when swing lands; -1 otherwise

    glm::vec2 swing_world_start;
    glm::vec2 swing_world_target;
    int swing_step;

    // Set by update() on landing, cleared by initiateStep().
    // Prevents retriggering the step event on consecutive frames.
    bool awaiting_step;

    Robot();

    // Advance one physics substep + swing leg motion.
    // Returns true exactly once per landing event (until initiateStep is called).
    bool update(glm::vec2 ref_vel);

    // Call when update() returns true.
    //   retire_slot  : 0/1/2 → which of stance_ids[] to lift
    //   world_target : where to send the idle leg (becomes new swing)
    void initiateStep(int retire_slot, glm::vec2 world_target);

    // 17-element observation vector — must match Python _get_obs() exactly.
    std::vector<float> buildObs(glm::vec2 ref_vel) const;

    // For each possible retire choice r∈{0,1,2}: minimum signed distance of the
    // CoM (body-frame origin) from the new triangle formed by
    //   { stance_ids[(r+1)%3], stance_ids[(r+2)%3], just_landed_id }.
    // Positive = CoM inside the new triangle = safe to retire.
    std::array<float,3> prospectiveMargins() const;

    bool isCMStable() const;

    State derivatives(const State& s, glm::vec2 ref_vel) const;

private:
    void stepRK4(glm::vec2 ref_vel);

    static glm::vec2 toWorld(float c, float s, glm::vec2 v);
    static glm::vec2 toBody (float c, float s, glm::vec2 v);

    // Signed distance of pt to the left of directed edge p1→p2.
    // Positive = pt is on the left (inside a CCW polygon).
    static float signedDist(glm::vec2 p1, glm::vec2 p2, glm::vec2 pt);

    // Minimum signed distance of the origin from triangle a,b,c.
    // Orientation is fixed internally (always tested as CCW).
    static float triMinMargin(glm::vec2 a, glm::vec2 b, glm::vec2 c);
};

#endif
