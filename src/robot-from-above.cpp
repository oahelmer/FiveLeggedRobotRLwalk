#include "robot-from-above.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

// ---- Frame helpers ---------------------------------------------------------
glm::vec2 Robot::toWorld(float c, float s, glm::vec2 v) {
    return { c*v.x - s*v.y, s*v.x + c*v.y };
}
glm::vec2 Robot::toBody(float c, float s, glm::vec2 v) {
    return { c*v.x + s*v.y, -s*v.x + c*v.y };
}

// ---- Geometry --------------------------------------------------------------

// Signed distance of pt to the left of directed edge p1→p2.
// Positive means pt is inside a CCW polygon on this edge's side.
float Robot::signedDist(glm::vec2 p1, glm::vec2 p2, glm::vec2 pt) {
    glm::vec2 e = p2 - p1;
    float len = glm::length(e);
    if (len < 1e-7f) return 0.0f;
    return (e.x*(pt.y - p1.y) - e.y*(pt.x - p1.x)) / len;
}

// Min signed distance of the origin {0,0} from triangle a,b,c.
// Internally ensures CCW winding before testing.
float Robot::triMinMargin(glm::vec2 a, glm::vec2 b, glm::vec2 c) {
    float area2 = (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
    if (area2 < 0.0f) std::swap(b, c);   // flip to CCW
    glm::vec2 origin{0.0f, 0.0f};
    return std::min({ signedDist(a,b,origin),
                      signedDist(b,c,origin),
                      signedDist(c,a,origin) });
}

// ---- Constructor -----------------------------------------------------------
Robot::Robot() {
    state = {{0,0}, 0, {0,0}, 0};

    for (int i = 0; i < 5; ++i) {
        float ang = i * (2.0f * M_PI / 5.0f);
        glm::vec2 pos = glm::vec2(cosf(ang), sinf(ang)) * 0.5f;
        legs.push_back({ pos, pos, pos, ang });
    }

    stance_ids  = {0, 1, 2};
    swing_id    = 3;
    idle_id     = 4;
    just_landed_id = 3;   // fire step event on the very first update()

    swing_world_start  = legs[3].foot_world_pos;
    swing_world_target = legs[3].foot_world_pos;
    swing_step  = Constants::SWING_STEPS; // starts as "already landed"
    awaiting_step = false; // let the first update() fire the initial step event
}

// ---- isCMStable ------------------------------------------------------------
bool Robot::isCMStable() const {
    std::vector<glm::vec2> pts;
    for (int si : stance_ids) pts.push_back(legs[si].foot_pos);
    if (pts.size() < 3) return false;

    // Sort by current body-frame angle (feet drift after replanting).
    std::sort(pts.begin(), pts.end(), [](const glm::vec2& a, const glm::vec2& b){
        return atan2f(a.y, a.x) < atan2f(b.y, b.x);
    });
    int n = static_cast<int>(pts.size());
    for (int i = 0; i < n; ++i) {
        if (signedDist(pts[i], pts[(i+1)%n], {0,0}) < 0.0f) return false;
    }
    return true;
}

// ---- prospectiveMargins ----------------------------------------------------
std::array<float,3> Robot::prospectiveMargins() const {
    if (just_landed_id < 0) return {0.0f, 0.0f, 0.0f};

    glm::vec2 landed = legs[just_landed_id].foot_pos;
    std::array<float,3> margins{};
    for (int r = 0; r < 3; ++r) {
        glm::vec2 a = legs[stance_ids[(r+1) % 3]].foot_pos;
        glm::vec2 b = legs[stance_ids[(r+2) % 3]].foot_pos;
        margins[r] = triMinMargin(a, b, landed);
    }
    return margins;
}

// ---- Physics derivatives ---------------------------------------------------
// PD force drives the body toward ref_vel; spring-dampers anchor stance feet.
State Robot::derivatives(const State& s, glm::vec2 ref_vel) const {
    float c  = cosf(s.theta);
    float st = sinf(s.theta);

    // PD proportional force (world frame)
    glm::vec2 F_pd = Constants::PD_KP * (ref_vel - s.vel) * Constants::M;

    // Two-stage angular PD: drive heading toward ref_vel direction.
    // Skip if ref_vel is nearly zero (no meaningful heading to track).
    float torque_pd = 0.0f;
    float ref_speed = glm::length(ref_vel);
    if (ref_speed > 1e-4f) {
        float ref_heading  = atan2f(ref_vel.y, ref_vel.x);
        float heading_err  = ref_heading - s.theta;
        // Wrap to [-π, π].
        heading_err -= 2.0f * M_PI * floorf((heading_err + M_PI) / (2.0f * M_PI));
        float ref_omega    = Constants::KP_HEADING * heading_err;
        torque_pd          = Constants::KP_ANG * (ref_omega - s.omega) * Constants::I;
    }

    glm::vec2 force_sum  = F_pd;
    float     torque_sum = torque_pd;

    for (int si : stance_ids) {
        // foot_pos (body frame) is kept in sync with the world anchor after each full step.
        // At s=state: foot_w == foot_world_pos, disp==0, only damping acts.
        // At RK4 sub-states: body has moved, foot_w diverges from the anchor,
        // and the spring correctly resists the displacement.
        glm::vec2 foot_w  = s.pos + toWorld(c, st, legs[si].foot_pos);
        glm::vec2 r_world = foot_w - s.pos;

        // Only rigid-body kinematics produce foot slide velocity.
        glm::vec2 foot_vel = s.vel
                           + glm::vec2(-s.omega * r_world.y, s.omega * r_world.x);

        glm::vec2 disp    = legs[si].foot_world_pos - foot_w;
        glm::vec2 f_world = Constants::SPRING_K * disp
                          - Constants::SPRING_D * foot_vel;

        force_sum  += f_world;
        torque_sum += r_world.x * f_world.y - r_world.y * f_world.x;
    }

    return { s.vel, s.omega,
             force_sum  / Constants::M,
             torque_sum / Constants::I };
}

// ---- RK4 -------------------------------------------------------------------
void Robot::stepRK4(glm::vec2 ref_vel) {
    auto k = [&](const State& s){ return derivatives(s, ref_vel); };

    State k1 = k(state);
    State s2 = { state.pos   + k1.pos  *(Constants::DT/2),
                 state.theta + k1.theta*(Constants::DT/2),
                 state.vel   + k1.vel  *(Constants::DT/2),
                 state.omega + k1.omega*(Constants::DT/2) };
    State k2 = k(s2);
    State s3 = { state.pos   + k2.pos  *(Constants::DT/2),
                 state.theta + k2.theta*(Constants::DT/2),
                 state.vel   + k2.vel  *(Constants::DT/2),
                 state.omega + k2.omega*(Constants::DT/2) };
    State k3 = k(s3);
    State s4 = { state.pos   + k3.pos  *Constants::DT,
                 state.theta + k3.theta*Constants::DT,
                 state.vel   + k3.vel  *Constants::DT,
                 state.omega + k3.omega*Constants::DT };
    State k4 = k(s4);

    const float h6 = Constants::DT / 6.0f;
    state.pos   += h6*(k1.pos   + 2.0f*k2.pos   + 2.0f*k3.pos   + k4.pos);
    state.theta += h6*(k1.theta + 2.0f*k2.theta + 2.0f*k3.theta + k4.theta);
    state.vel   += h6*(k1.vel   + 2.0f*k2.vel   + 2.0f*k3.vel   + k4.vel);
    state.omega += h6*(k1.omega + 2.0f*k2.omega + 2.0f*k3.omega + k4.omega);

    state.vel   *= 0.98f;
    state.omega *= 0.98f;
}

// ---- update ----------------------------------------------------------------
// Returns true exactly once per landing event (until initiateStep is called).
bool Robot::update(glm::vec2 ref_vel) {
    // If we're waiting for a step decision, do nothing until initiateStep().
    if (awaiting_step) return false;

    // 1. Advance physics.
    stepRK4(ref_vel);

    float c2  = cosf(state.theta);
    float st2 = sinf(state.theta);

    // 2. Reproject stance feet from their fixed world anchors.
    for (int si : stance_ids) {
        glm::vec2 r = legs[si].foot_world_pos - state.pos;
        legs[si].foot_pos = toBody(c2, st2, r);
    }

    // 3. Advance swing leg (linear world-space interpolation).
    swing_step++;
    float t = std::min((float)swing_step / (float)Constants::SWING_STEPS, 1.0f);
    glm::vec2 swing_world = (1.0f - t)*swing_world_start + t*swing_world_target;
    legs[swing_id].foot_pos = toBody(c2, st2, swing_world - state.pos);

    // 4. Keep idle leg reprojected from its fixed world position (drifts freely).
    legs[idle_id].foot_pos = toBody(c2, st2, legs[idle_id].foot_world_pos - state.pos);

    // 5. Check for landing.
    if (swing_step >= Constants::SWING_STEPS) {
        just_landed_id = swing_id;
        awaiting_step  = true;
        return true;
    }
    return false;
}

// ---- initiateStep ----------------------------------------------------------
void Robot::initiateStep(int retire_slot, glm::vec2 world_target) {
    float c  = cosf(state.theta);
    float st = sinf(state.theta);

    int landed   = just_landed_id;
    int retiring = stance_ids[retire_slot];
    int new_swing = idle_id;

    // Lock the just-landed leg as a proper stance anchor.
    legs[landed].foot_world_pos = state.pos + toWorld(c, st, legs[landed].foot_pos);

    // Build new stance_ids: remove retiring, add landed.
    std::array<int,3> ns{};
    int j = 0;
    for (int i = 0; i < 3; ++i)
        if (stance_ids[i] != retiring) ns[j++] = stance_ids[i];
    ns[2] = landed;
    stance_ids = ns;

    // Record retiring leg's world position (it now floats freely as idle).
    legs[retiring].foot_world_pos = state.pos + toWorld(c, st, legs[retiring].foot_pos);

    // Assign roles.
    idle_id  = retiring;
    swing_id = new_swing;

    // Set up new swing arc.
    swing_world_start  = state.pos + toWorld(c, st, legs[new_swing].foot_pos);
    swing_world_target = world_target;
    swing_step         = 0;

    just_landed_id = -1;
    awaiting_step  = false;
}

// ---- buildObs --------------------------------------------------------------
// Must match RLwalkingtraining._get_obs() exactly.
std::vector<float> Robot::buildObs(glm::vec2 ref_vel) const {
    float theta = state.theta;
    float c = cosf(theta), s = sinf(theta);

    float vx_b =  c*state.vel.x + s*state.vel.y;
    float vy_b = -s*state.vel.x + c*state.vel.y;
    float rvx  =  c*ref_vel.x   + s*ref_vel.y;
    float rvy  = -s*ref_vel.x   + c*ref_vel.y;

    std::vector<float> obs = { vx_b/2.0f, vy_b/2.0f, rvx/0.6f, rvy/0.6f };

    // Old stance legs (body frame, normalised by reach).
    for (int si : stance_ids) {
        obs.push_back(legs[si].foot_pos.x / Constants::REACH_MAX);
        obs.push_back(legs[si].foot_pos.y / Constants::REACH_MAX);
    }

    // Just-landed leg.
    if (just_landed_id >= 0) {
        obs.push_back(legs[just_landed_id].foot_pos.x / Constants::REACH_MAX);
        obs.push_back(legs[just_landed_id].foot_pos.y / Constants::REACH_MAX);
    } else {
        obs.push_back(0.0f); obs.push_back(0.0f);
    }

    // Idle leg.
    obs.push_back(legs[idle_id].foot_pos.x / Constants::REACH_MAX);
    obs.push_back(legs[idle_id].foot_pos.y / Constants::REACH_MAX);

    // Prospective margins (clipped to [-1, 1]).
    auto margins = prospectiveMargins();
    for (float m : margins)
        obs.push_back(std::max(-1.0f, std::min(1.0f, m)));

    return obs;   // 4 + 6 + 2 + 2 + 3 = 17 elements
}
