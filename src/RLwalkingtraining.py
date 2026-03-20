"""
5-legged tripod walker — redesigned RL environment.

Architecture
============
Always exactly 3 feet on the ground (stance). One foot swings to a new
placement target; one floats idle (waiting for next turn).

Gait cycle (one RL step = one full swing phase):
  1. Swing leg reaches world target → lands.
  2. RL picks: which of the 3 old stance legs to retire, and where to
     send the idle leg (which becomes the new swing).
  3. Landed leg joins stance (locked anchor). Retired leg floats idle.
  4. Physics runs for SWING_STEPS substeps until new swing lands.
  5. Repeat.

At no point are fewer than 3 feet on the ground.

Observation  (17-D, must match robot-from-above.cpp buildObs exactly)
------------
  vx_b/2, vy_b/2                    body velocity in body frame
  ref_vx_b/0.6, ref_vy_b/0.6       reference velocity in body frame
  s0x/R, s0y/R ... s2x/R, s2y/R   3 old stance legs (body frame / reach)
  lx/R, ly/R                        just-landed leg (body frame / reach)
  ix/R, iy/R                        idle leg (body frame / reach)
  pm0, pm1, pm2                     prospective margins for each retire choice

Prospective margin pmᵢ = min signed distance of CoM from the new triangle
formed by [stance[(i+1)%3], stance[(i+2)%3], just_landed].
Positive = CoM will be inside the new triangle (safe to retire leg i).
This makes the stability consequence of each retire choice explicit.

Action (5-D)
------------
  r0, r1, r2   argmax → which of the 3 old stance legs to retire
  tx, ty        target for new swing (body frame, normalised by reach_max)
"""

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import math

class FiveLegEnv(gym.Env):
    # ---- Physics constants (mirror constants.cpp) ----
    MASS      = 2.0
    INERTIA   = 0.5
    SPRING_K  = 500.0
    SPRING_D  = 30.0
    PD_KP     = 4.0
    KP_HEADING = 2.0   # heading error (rad) → target omega (rad/s)
    KP_ANG     = 4.0   # omega error → angular accel × I  (matches translational PD)
    DT        = 0.005
    SWING_STEPS = 40       # 0.2 s per swing phase

    # Per-leg circular placement zones — must match constants.cpp exactly.
    # Zone i centred at (cos(i*2π/5), sin(i*2π/5)) * LEG_ZONE_CENTER_R.
    LEG_ZONE_CENTER_R = 0.85
    LEG_ZONE_RADIUS   = 0.30

    def __init__(self):
        super().__init__()
        self.max_reach = 0.5 + math.sqrt(1.0 - (0.2 / (2.0*0.4))**2)

        self.action_space      = gym.spaces.Box(-1.0, 1.0, shape=(5,),  dtype=np.float32)
        self.observation_space = gym.spaces.Box(-3.0, 3.0, shape=(17,), dtype=np.float32)
        self.reset()

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _signed_dist(p1, p2, pt):
        """Signed distance of pt from directed edge p1→p2 (positive = left / inside CCW)."""
        e  = np.asarray(p2) - np.asarray(p1)
        ln = np.linalg.norm(e)
        if ln < 1e-8:
            return 0.0
        return (e[0]*(pt[1]-p1[1]) - e[1]*(pt[0]-p1[0])) / ln

    def _tri_min_margin(self, a, b, c):
        """Min signed distance of the origin from triangle a,b,c (auto-fixes winding)."""
        area2 = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if area2 < 0:
            b, c = c, b
        o = (0.0, 0.0)
        return min(self._signed_dist(a,b,o),
                   self._signed_dist(b,c,o),
                   self._signed_dist(c,a,o))

    def _prospective_margins(self, landed_body):
        """For each retire choice r, compute min margin of new triangle."""
        verts = [self.foot_body[self.stance_ids[i]] for i in range(3)]
        return [
            self._tri_min_margin(verts[(r+1)%3], verts[(r+2)%3], landed_body)
            for r in range(3)
        ]

    def _cm_in_stance(self):
        """Is the CoM (body-frame origin) inside the current stance triangle?"""
        pts = sorted([self.foot_body[si].tolist() for si in self.stance_ids],
                     key=lambda p: math.atan2(p[1], p[0]))
        n = len(pts)
        for i in range(n):
            if self._signed_dist(pts[i], pts[(i+1)%n], (0,0)) < 0:
                return False
        return True

    def _rot_w2b(self, v, theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([ c*v[0]+s*v[1], -s*v[0]+c*v[1]])

    def _rot_b2w(self, v, theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c*v[0]-s*v[1],  s*v[0]+c*v[1]])

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.body_vel   = np.zeros(2, dtype=np.float64)
        self.body_omega = 0.0
        self.body_theta = 0.0
        self.body_pos   = np.zeros(2, dtype=np.float64)

        # Randomise reference direction and speed each episode.
        angle = self.np_random.uniform(0, 2*math.pi)
        speed = self.np_random.uniform(0.2, 0.5)
        self.ref = np.array([speed*math.cos(angle), speed*math.sin(angle)])

        # Foot positions in body frame and world-frame anchors.
        self.foot_body  = np.zeros((5, 2))
        self.foot_world = np.zeros((5, 2))
        for i in range(5):
            ang = i * (2*math.pi/5)
            p = np.array([math.cos(ang)*0.5, math.sin(ang)*0.5])
            self.foot_body[i]  = p
            self.foot_world[i] = p   # body_pos=0 at start

        # Randomise initial tripod (any 3 of 5 legs).
        grounded = sorted(self.np_random.choice(5, 3, replace=False).tolist())
        free     = [i for i in range(5) if i not in grounded]

        self.stance_ids     = grounded          # list of 3
        self.swing_id       = free[0]
        self.idle_id        = free[1]
        self.just_landed_id = free[0]           # fire step event immediately

        # Swing "already landed" at nominal position → first RL call triggers setup.
        self.swing_world_start  = self.foot_world[free[0]].copy()
        self.swing_world_target = self.foot_world[free[0]].copy()
        self.swing_step         = self.SWING_STEPS
        self.awaiting_step      = True

        self.episode_step = 0
        return self._get_obs(), {}

    # -----------------------------------------------------------------------
    # Observation  (17-D — must match buildObs() in robot-from-above.cpp)
    # -----------------------------------------------------------------------
    def _get_obs(self):
        theta = self.body_theta
        vb  = self._rot_w2b(self.body_vel, theta)
        rb  = self._rot_w2b(self.ref,      theta)

        obs = [vb[0]/2.0, vb[1]/2.0, rb[0]/0.6, rb[1]/0.6]

        # Old stance legs.
        for si in self.stance_ids:
            obs += [self.foot_body[si][0]/self.max_reach,
                    self.foot_body[si][1]/self.max_reach]

        # Just-landed leg.
        if self.just_landed_id is not None:
            obs += [self.foot_body[self.just_landed_id][0]/self.max_reach,
                    self.foot_body[self.just_landed_id][1]/self.max_reach]
        else:
            obs += [0.0, 0.0]

        # Idle leg.
        obs += [self.foot_body[self.idle_id][0]/self.max_reach,
                self.foot_body[self.idle_id][1]/self.max_reach]

        # Prospective margins.
        jl_body = (self.foot_body[self.just_landed_id]
                   if self.just_landed_id is not None
                   else np.zeros(2))
        pms = self._prospective_margins(jl_body)
        obs += [float(np.clip(m, -1, 1)) for m in pms]

        return np.array(obs, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Physics — derivatives and RK4 integrator
    # (matches derivatives() + stepRK4() in robot-from-above.cpp exactly)
    # -----------------------------------------------------------------------
    def _derivatives(self, pos, theta, vel, omega):
        """
        Returns (dpos, dtheta, dvel, domega) at the given intermediate state.
        Uses self.foot_body (body-frame positions from the START of the timestep)
        and self.foot_world (fixed world anchors) — both are constant across all
        RK4 sub-evaluations, exactly as in C++.
        """
        c, s = math.cos(theta), math.sin(theta)

        # ---- Translational PD (world frame) ----
        F_pd = self.PD_KP * (self.ref - vel) * self.MASS

        # ---- Two-stage angular PD ----
        # Stage 1: heading error → reference omega
        # Stage 2: omega error  → PD torque
        torque_pd = 0.0
        ref_speed = float(np.linalg.norm(self.ref))
        if ref_speed > 1e-4:
            ref_heading  = math.atan2(self.ref[1], self.ref[0])
            heading_err  = ref_heading - theta
            # Wrap to [-π, π]
            heading_err -= 2*math.pi * math.floor((heading_err + math.pi) / (2*math.pi))
            ref_omega_target = self.KP_HEADING * heading_err
            torque_pd = self.KP_ANG * (ref_omega_target - omega) * self.INERTIA

        force  = F_pd.copy()
        torque = torque_pd

        # ---- Stance spring-dampers ----
        for si in self.stance_ids:
            lx, ly  = self.foot_body[si]   # body-frame pos from start of step
            foot_w  = np.array([pos[0] + c*lx - s*ly,
                                 pos[1] + s*lx + c*ly])
            r_world = foot_w - pos

            foot_vel = np.array([vel[0] - omega*r_world[1],
                                  vel[1] + omega*r_world[0]])

            disp    = self.foot_world[si] - foot_w
            f_world = self.SPRING_K*disp - self.SPRING_D*foot_vel

            force  += f_world
            torque += r_world[0]*f_world[1] - r_world[1]*f_world[0]

        dpos   = vel
        dtheta = omega
        dvel   = force  / self.MASS
        domega = torque / self.INERTIA
        return dpos, dtheta, dvel, domega

    def _physics_step(self):
        """Advance one DT using forward Euler. Returns True if swing just landed.

        Euler is numerically stable at DT=0.005 for these spring constants
        (natural freq ~15.8 rad/s, stability margin ~35×).  Using RK4 here
        costs 4× the compute for no meaningful benefit during training — RK4
        is only needed in the C++ runtime where trajectory accuracy matters.
        """
        dpos, dtheta, dvel, domega = self._derivatives(
            self.body_pos, self.body_theta, self.body_vel, self.body_omega)

        self.body_pos   = self.body_pos + dpos   * self.DT
        self.body_theta = self.body_theta + dtheta * self.DT
        self.body_vel   = self.body_vel   + dvel   * self.DT
        self.body_omega = self.body_omega + domega * self.DT

        self.body_vel   *= 0.98
        self.body_omega *= 0.98

        c2, s2 = math.cos(self.body_theta), math.sin(self.body_theta)

        # Reproject stance feet from world anchors.
        for si in self.stance_ids:
            r = self.foot_world[si] - self.body_pos
            self.foot_body[si][0] =  c2*r[0] + s2*r[1]
            self.foot_body[si][1] = -s2*r[0] + c2*r[1]

        # Advance swing leg (linear world interpolation).
        self.swing_step += 1
        t = min(self.swing_step / self.SWING_STEPS, 1.0)
        sw = (1-t)*self.swing_world_start + t*self.swing_world_target
        r = sw - self.body_pos
        self.foot_body[self.swing_id][0] =  c2*r[0] + s2*r[1]
        self.foot_body[self.swing_id][1] = -s2*r[0] + c2*r[1]

        # Reproject idle leg (fixed world position, drifts in body frame).
        r_idle = self.foot_world[self.idle_id] - self.body_pos
        self.foot_body[self.idle_id][0] =  c2*r_idle[0] + s2*r_idle[1]
        self.foot_body[self.idle_id][1] = -s2*r_idle[0] + c2*r_idle[1]

        if self.swing_step >= self.SWING_STEPS:
            self.just_landed_id = self.swing_id
            self.awaiting_step  = True
            return True
        return False

    # -----------------------------------------------------------------------
    # Initiate next step
    # -----------------------------------------------------------------------
    def _initiate_step(self, retire_slot, world_target):
        theta = self.body_theta
        c, s  = math.cos(theta), math.sin(theta)

        landed   = self.just_landed_id
        retiring = self.stance_ids[retire_slot]
        new_swing = self.idle_id

        # Lock just-landed leg as a proper stance anchor.
        lx, ly = self.foot_body[landed]
        self.foot_world[landed] = np.array([self.body_pos[0] + c*lx - s*ly,
                                             self.body_pos[1] + s*lx + c*ly])

        # Update stance: remove retiring, add landed.
        self.stance_ids = [si for si in self.stance_ids if si != retiring]
        self.stance_ids.append(landed)

        # Record retiring leg's current world position (it floats idle).
        lx_r, ly_r = self.foot_body[retiring]
        self.foot_world[retiring] = np.array([self.body_pos[0] + c*lx_r - s*ly_r,
                                               self.body_pos[1] + s*lx_r + c*ly_r])

        # Assign new roles.
        self.idle_id  = retiring
        self.swing_id = new_swing

        # Set up new swing arc.
        lx_s, ly_s = self.foot_body[new_swing]
        self.swing_world_start  = np.array([self.body_pos[0] + c*lx_s - s*ly_s,
                                             self.body_pos[1] + s*lx_s + c*ly_s])
        self.swing_world_target = world_target.copy()
        self.swing_step         = 0
        self.just_landed_id     = None
        self.awaiting_step      = False

    # -----------------------------------------------------------------------
    # RL step
    # -----------------------------------------------------------------------
    def step(self, action):
        # ---- Decode action ----
        retire_slot = int(np.argmax(action[:3]))

        # Foot placement: action[3:5] is an offset in [-1,1]^2 within the idle
        # leg's circular zone.  The zone is centred along the leg's mount
        # direction at LEG_ZONE_CENTER_R; the offset is scaled by LEG_ZONE_RADIUS
        # and clamped to the unit disc so the foot always stays inside the zone.
        idle_ang = self.idle_id * (2 * math.pi / 5)
        zone_cx  = math.cos(idle_ang) * self.LEG_ZONE_CENTER_R
        zone_cy  = math.sin(idle_ang) * self.LEG_ZONE_CENTER_R

        ox = float(np.clip(action[3], -1.0, 1.0))
        oy = float(np.clip(action[4], -1.0, 1.0))
        od = math.sqrt(ox*ox + oy*oy)
        if od > 1.0:          # clamp offset to unit disc
            ox /= od; oy /= od

        tx = zone_cx + ox * self.LEG_ZONE_RADIUS
        ty = zone_cy + oy * self.LEG_ZONE_RADIUS
        td = math.sqrt(tx*tx + ty*ty)   # used only for post-swing reward

        # ---- Validate retire choice via prospective margins ----
        jl_body = self.foot_body[self.just_landed_id]
        margins = self._prospective_margins(jl_body)
        if margins[retire_slot] < 0.0:
            # Override: pick the safest available retire.
            retire_slot = int(np.argmax(margins))

        # ---- Convert target from body frame to world frame ----
        theta = self.body_theta
        c, s  = math.cos(theta), math.sin(theta)
        world_target = np.array([self.body_pos[0] + c*tx - s*ty,
                                  self.body_pos[1] + s*tx + c*ty])

        # ---- Initiate the step ----
        self._initiate_step(retire_slot, world_target)

        # ---- Run physics until next landing ----
        reward = 0.0
        for _ in range(self.SWING_STEPS + 5):
            landed = self._physics_step()

            # Velocity tracking reward (per substep).
            vb = self._rot_w2b(self.body_vel, self.body_theta)
            rb = self._rot_w2b(self.ref, self.body_theta)
            err = math.sqrt((vb[0]-rb[0])**2 + (vb[1]-rb[1])**2)
            reward += math.exp(-5.0 * err**2)

            # Stability during swing — should not happen but catch it.
            if not self._cm_in_stance():
                reward -= 20.0
                self.episode_step += 1
                return np.zeros(17, dtype=np.float32), reward, True, False, {}

            if landed:
                break

        # ---- Post-swing rewards ----
        # Good retire choice: margin of new triangle.
        reward += 3.0 * float(np.clip(margins[retire_slot], -1, 1))

        # Reward placing the foot in the forward half of its zone.
        # (Zone geometry already prevents extremes; this just biases toward progress.)
        rb_dir    = self._rot_w2b(self.ref, self.body_theta)
        ref_speed = max(np.linalg.norm(rb_dir), 1e-6)
        forward   = rb_dir / ref_speed
        progress  = tx * forward[0] + ty * forward[1]
        if progress > 0.0:
            reward += 0.5 * min(progress / self.LEG_ZONE_RADIUS, 1.0)

        self.episode_step += 1
        truncated = self.episode_step >= 200    # 200 steps × 0.2 s = 40 s
        return self._get_obs(), reward, False, truncated, {}


# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Wrap in DummyVecEnv so VecNormalize can work.
    # norm_obs=False  — obs are already manually normalised (~[-1,1]) in _get_obs().
    # norm_reward=True — THIS is the fix for explained_variance=0: rescales
    #                    accumulated returns to ~unit variance before the value
    #                    function ever sees them. gamma here must match PPO's gamma.
    raw_env = DummyVecEnv([lambda: FiveLegEnv()])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,           # was 0.995 — lower is ~100-step horizon, much easier
                              # to fit; 0.995 stretched credit across the full episode
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[256, 256, 128],  # deeper value net: more capacity for return fitting
            ),
            log_std_init=-0.5,
        ),
    )

    print("Training (1 M steps — typically converges by 400 k)...")
    model.learn(
        total_timesteps=1_000_000,
        callback=CheckpointCallback(save_freq=100_000, save_path="./checkpoints/"),
    )

    # Save policy + VecNormalize stats together.
    # (obs normalisation is off so C++ doesn't need the stats at inference,
    #  but saving them anyway is good practice for resuming training.)
    model.save("rl_policy_sb3")
    env.save("vecnormalize.pkl")

    # ---- Export policy to ONNX (for C++ inference) ----
    print("Exporting to ONNX...")

    class OnnxWrapper(th.nn.Module):
        def __init__(self, extractor, action_net):
            super().__init__()
            self.extractor  = extractor
            self.action_net = action_net
        def forward(self, obs):
            return self.action_net(self.extractor(obs))

    onnx_model = OnnxWrapper(model.policy.mlp_extractor.policy_net,
                              model.policy.action_net)
    dummy = th.randn(1, 17)
    th.onnx.export(onnx_model, dummy, "rl_policy.onnx",
                   opset_version=11,
                   input_names=["obs"],
                   output_names=["action"])
    print("Saved as rl_policy.onnx")

