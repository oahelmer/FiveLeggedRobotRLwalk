#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "src/robot-from-above.h"
#include "src/constants.h"
#include "src/RLwalking.h"
#include <cmath>
#include <vector>
#include <algorithm>

// ---- Drawing helpers -------------------------------------------------------

static void drawCircle(float x, float y, float r, int seg, bool filled = false) {
    glBegin(filled ? GL_TRIANGLE_FAN : GL_LINE_LOOP);
    if (filled) glVertex2f(x, y);
    for (int i = 0; i <= seg; i++) {
        float a = i * 2.0f * M_PI / seg;
        glVertex2f(x + cosf(a)*r, y + sinf(a)*r);
    }
    glEnd();
}

static void drawX(float x, float y, float r) {
    glBegin(GL_LINES);
    glVertex2f(x-r, y-r); glVertex2f(x+r, y+r);
    glVertex2f(x+r, y-r); glVertex2f(x-r, y+r);
    glEnd();
}

static void drawArrowhead(float x, float y, float dx, float dy, float s) {
    float len = sqrtf(dx*dx + dy*dy);
    if (len < 1e-6f) return;
    float ux = dx/len, uy = dy/len;
    float lx = -uy,   ly =  ux;
    glBegin(GL_TRIANGLES);
    glVertex2f(x + ux*s,                        y + uy*s);
    glVertex2f(x - ux*s*0.5f + lx*s*0.5f,  y - uy*s*0.5f + ly*s*0.5f);
    glVertex2f(x - ux*s*0.5f - lx*s*0.5f,  y - uy*s*0.5f - ly*s*0.5f);
    glEnd();
}

// ---- World reference grid --------------------------------------------------
// Draws a grid of small circles so the robot's translation is clearly visible.
static void drawWorldGrid(float cam_x, float cam_y, float half_w) {
    const float GRID_SPACING = 1.0f;
    const float DOT_R        = 0.04f;
    const int   DOT_SEG      = 8;

    // Snap to nearest grid line so markers fill the viewport, not just origin.
    float x0 = floorf((cam_x - half_w) / GRID_SPACING) * GRID_SPACING;
    float y0 = floorf((cam_y - half_w) / GRID_SPACING) * GRID_SPACING;

    glColor3f(0.25f, 0.25f, 0.25f);
    for (float gx = x0; gx <= cam_x + half_w; gx += GRID_SPACING) {
        for (float gy = y0; gy <= cam_y + half_w; gy += GRID_SPACING) {
            // Larger marker at every 5-unit intersection.
            bool big = (fabsf(fmodf(gx, 5.0f)) < 0.01f &&
                        fabsf(fmodf(gy, 5.0f)) < 0.01f);
            if (big) {
                glColor3f(0.45f, 0.45f, 0.45f);
                drawCircle(gx, gy, DOT_R * 2.0f, DOT_SEG, true);
                glColor3f(0.25f, 0.25f, 0.25f);
            } else {
                drawCircle(gx, gy, DOT_R, DOT_SEG, true);
            }
        }
    }

    // Axis cross at world origin (if visible).
    if (fabsf(cam_x) < half_w + 2.0f && fabsf(cam_y) < half_w + 2.0f) {
        glLineWidth(1.0f);
        glColor3f(0.35f, 0.15f, 0.15f);
        glBegin(GL_LINES);
        glVertex2f(-half_w + cam_x - 1, 0); glVertex2f(half_w + cam_x + 1, 0);
        glEnd();
        glColor3f(0.15f, 0.35f, 0.15f);
        glBegin(GL_LINES);
        glVertex2f(0, -half_w + cam_y - 1); glVertex2f(0, half_w + cam_y + 1);
        glEnd();
    }
}

// ---- Per-leg placement zones (body frame, drawn before glPopMatrix) --------
static void drawLegZones(const Robot& robot) {
    for (int i = 0; i < 5; ++i) {
        float ang = i * (2.0f * M_PI / 5.0f);
        float cx  = cosf(ang) * Constants::LEG_ZONE_CENTER_R;
        float cy  = sinf(ang) * Constants::LEG_ZONE_CENTER_R;

        bool is_idle  = (i == robot.idle_id);
        bool is_swing = (i == robot.swing_id);

        // Fill: highlight the zone of the leg currently being placed (idle→swing).
        if (is_idle) {
            glColor4f(1.0f, 0.8f, 0.2f, 0.12f);   // gold tint = next target zone
            drawCircle(cx, cy, Constants::LEG_ZONE_RADIUS, 32, true);
        } else if (is_swing) {
            glColor4f(1.0f, 0.5f, 0.0f, 0.08f);   // orange tint = leg in flight
            drawCircle(cx, cy, Constants::LEG_ZONE_RADIUS, 32, true);
        }

        // Outline: all zones, dimly.
        glLineWidth(1.0f);
        glColor4f(0.5f, 0.5f, 0.5f, 0.4f);
        drawCircle(cx, cy, Constants::LEG_ZONE_RADIUS, 32);

        // Centre dot.
        glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
        glPointSize(4.0f);
        glBegin(GL_POINTS); glVertex2f(cx, cy); glEnd();
    }
}

// ===========================================================================
int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(900, 900,
        "5-Legged Robot  |  LEFT/RIGHT=turn  UP/DOWN=speed  SPACE=stop",
        NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    // Enable alpha blending so filled zone circles look translucent.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Robot robot;
    RLController rl("rl_policy.onnx");

    float ref_speed = 0.5f;
    float ref_angle = 0.0f;
    glm::vec2 ref_vel(ref_speed, 0.0f);

    const float TURN_RATE  = 0.04f; // was 0.04f
    const float ACCEL_RATE = 0.01f;
    const float SPEED_MAX  = 2.0f;   // was 1.0f cap at training max (trained on 0.2–0.5)

    float cam_x = 0.0f, cam_y = 0.0f;
    const int SUBSTEPS = 3;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ---- Keyboard ----
        if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) ref_angle += TURN_RATE;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) ref_angle -= TURN_RATE;
        if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS)
            ref_speed = std::min(ref_speed + ACCEL_RATE, SPEED_MAX);
        if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS)
            ref_speed = std::max(ref_speed - ACCEL_RATE, 0.0f);
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            ref_speed = 0.0f;

        ref_vel = glm::vec2(cosf(ref_angle) * ref_speed,
                            sinf(ref_angle) * ref_speed);

        // ---- Physics substeps ----
        for (int sub = 0; sub < SUBSTEPS; ++sub) {
            bool step_event = robot.update(ref_vel);

            if (step_event) {
                auto obs     = robot.buildObs(ref_vel);
                auto actions = rl.compute_action(obs);

                // Retire slot: argmax of first 3 logits.
                int retire_slot = static_cast<int>(
                    std::max_element(actions.begin(), actions.begin()+3)
                    - actions.begin());

                // Safety override.
                auto margins = robot.prospectiveMargins();
                if (margins[retire_slot] < 0.0f) {
                    retire_slot = static_cast<int>(
                        std::max_element(margins.begin(), margins.end())
                        - margins.begin());
                }

                // Zone-constrained foot placement.
                // actions[3:5] is an offset in [-1,1]^2 within the idle leg's zone.
                float idle_ang = robot.legs[robot.idle_id].mount_angle;
                float zone_cx  = cosf(idle_ang) * Constants::LEG_ZONE_CENTER_R;
                float zone_cy  = sinf(idle_ang) * Constants::LEG_ZONE_CENTER_R;

                float ox = std::clamp(actions[3], -1.0f, 1.0f);
                float oy = std::clamp(actions[4], -1.0f, 1.0f);
                float od = sqrtf(ox*ox + oy*oy);
                if (od > 1.0f) { ox /= od; oy /= od; }   // clamp to unit disc

                float tx = zone_cx + ox * Constants::LEG_ZONE_RADIUS;
                float ty = zone_cy + oy * Constants::LEG_ZONE_RADIUS;

                // Body → world frame.
                float theta = robot.state.theta;
                float c = cosf(theta), s = sinf(theta);
                glm::vec2 world_target = {
                    robot.state.pos.x + c*tx - s*ty,
                    robot.state.pos.y + s*tx + c*ty
                };

                robot.initiateStep(retire_slot, world_target);
            }
        }

        // Smooth camera follow.
        cam_x += 0.08f * (robot.state.pos.x - cam_x);
        cam_y += 0.08f * (robot.state.pos.y - cam_y);

        glClear(GL_COLOR_BUFFER_BIT);
        glLoadIdentity();
        const float HW = 4.5f;
        glOrtho(cam_x-HW, cam_x+HW, cam_y-HW, cam_y+HW, -1, 1);

        // ---- World reference grid (drawn before robot) ----
        drawWorldGrid(cam_x, cam_y, HW);

        bool  stable = robot.isCMStable();
        float theta  = robot.state.theta;

        // ---- Body-local drawing ----
        glPushMatrix();
        glTranslatef(robot.state.pos.x, robot.state.pos.y, 0);
        glRotatef(theta * 180.0f / M_PI, 0, 0, 1);

        // Per-leg placement zones.
        drawLegZones(robot);

        // Stance triangle fill.
        if (stable) glColor4f(0.0f, 0.6f, 0.2f, 0.30f);
        else        glColor4f(0.9f, 0.1f, 0.0f, 0.55f);
        glBegin(GL_POLYGON);
        for (int si : robot.stance_ids)
            glVertex2f(robot.legs[si].foot_pos.x, robot.legs[si].foot_pos.y);
        glEnd();

        // Stance triangle outline.
        glLineWidth(2.0f);
        glColor3f(0.0f, 1.0f, 0.4f);
        glBegin(GL_LINE_LOOP);
        for (int si : robot.stance_ids)
            glVertex2f(robot.legs[si].foot_pos.x, robot.legs[si].foot_pos.y);
        glEnd();

        // Body disc.
        glLineWidth(1.5f);
        glColor3f(0.75f, 0.75f, 0.75f);
        drawCircle(0, 0, Constants::D/2.0f, 32);

        // CoM dot.
        glColor3f(stable ? 0.2f : 1.0f, stable ? 1.0f : 0.1f, 0.1f);
        glPointSize(13.0f);
        glBegin(GL_POINTS); glVertex2f(0, 0); glEnd();

        // All 5 legs.
        for (int i = 0; i < 5; ++i) {
            const auto& leg = robot.legs[i];
            bool is_stance = std::find(robot.stance_ids.begin(),
                                       robot.stance_ids.end(), i)
                             != robot.stance_ids.end();
            bool is_swing  = (i == robot.swing_id);

            glColor3f(0.45f, 0.45f, 0.45f);
            glLineWidth(1.5f);
            glBegin(GL_LINES);
            glVertex2f(leg.anchor_pos.x, leg.anchor_pos.y);
            glVertex2f(leg.foot_pos.x,   leg.foot_pos.y);
            glEnd();

            if      (is_stance) glColor3f(0.1f, 1.0f, 0.3f);
            else if (is_swing)  glColor3f(1.0f, 0.6f, 0.0f);
            else                glColor3f(0.5f, 0.6f, 1.0f);

            glPointSize(10.0f);
            glBegin(GL_POINTS); glVertex2f(leg.foot_pos.x, leg.foot_pos.y); glEnd();
        }
        glPopMatrix();

        // ---- World-frame overlays ----
        // Swing target marker.
        if (!robot.awaiting_step) {
            glColor3f(1.0f, 0.6f, 0.0f);
            glLineWidth(2.0f);
            drawX(robot.swing_world_target.x, robot.swing_world_target.y, 0.12f);
        }

        // Reference velocity arrow.
        {
            float rx = robot.state.pos.x, ry = robot.state.pos.y;
            float ax = ref_vel.x * 2.5f,  ay = ref_vel.y * 2.5f;
            float sf = ref_speed / SPEED_MAX;
            glColor3f(sf, sf, 1.0f - sf * 0.5f);
            glLineWidth(2.5f);
            glBegin(GL_LINES);
            glVertex2f(rx, ry); glVertex2f(rx+ax, ry+ay);
            glEnd();
            drawArrowhead(rx+ax, ry+ay, ax, ay, 0.18f);
            if (ref_speed < 0.01f) {
                glColor3f(1.0f, 0.3f, 0.3f);
                glLineWidth(2.0f);
                drawCircle(rx, ry, 0.25f, 20);
            }
        }

        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
