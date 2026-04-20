
# HOW THIS SCRIPT WORKS (big picture):
#   1. Read the full LiDAR scan every loop iteration
#   2. Filter scan into front/left/right zones to measure clearance
#   3. Compute PID error from front distance vs safe distance
#   4. Map PID output → forward speed, and zone asymmetry → turn speed
#   5. Send resulting wheel targets to the closed-loop motor controller
#   6. Log data to CSV for analysis 


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import time
import numpy as np
import math

import L1_lidar                          # LiDAR hardware interface
import L2_speed_control as sc            # closed-loop motor control
import L2_kinematics as kin              # encoder feedback (current wheel speeds)
import L1_log as log                     # CSV data logging


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIDAR INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
#
#   Lidar()    — creates the sensor object
#   connect()  — opens the connection to the physical hardware
#   run()      — starts the background thread that continuously reads scan data
#   sleep(1)   — waits 1 second to let the thread fully start before we read it
#
# After this block, lidarsensor.get() returns the latest full scan at any time.
# ═════════════════════════════════════════════════════════════════════════════
lidarsensor = L1_lidar.Lidar()
lidarsensor.connect()
lidarsensor.run()
time.sleep(1)    # allow background thread to start before entering the loop


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PID CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
#
#   kp (proportional) — reacts to current error.
#                        Too small = sluggish. Too large = oscillates/overshoots.
#                        Start at 0.04 and increase by 0.01 until robot responds.
#
#   ki (integral)     — reacts to accumulated past error.
#                        Fixes the robot "stopping slightly too close/far" problem.
#                        Start at 0.005 after kp is tuned.
#
#   kd (derivative)   — reacts to how fast error is changing.
#                        Smooths out jerky responses to sudden LiDAR changes.
#                        Start at 0.02 after ki is tuned.
#
#   kp_turn           — separate gain just for the left/right steering correction.
#                        Tune this independently from kp.
# ═════════════════════════════════════════════════════════════════════════════
kp      = 0.04    # tune first
ki      = 0.00    # tune second
kd      = 0.00    # tune third
kp_turn = 0.30    # tune independently — controls how aggressively robot steers


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FIXED PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
#
#   SAFE_DISTANCE  — how far (meters) the robot tries to stay from any obstacle.
#                    The PID error = zero when front distance equals this value.
#
#   MAX_SPEED      — maximum wheel speed in rad/s. 
#
#   MAX_INTEGRAL   — caps the integral term so it cannot grow unbounded when the
#                    robot is stopped against a wall.
#
#   HALF_CONE      — half-width of each LiDAR zone in RADIANS (≈ 0.524 rad = 30°).
#            
# ═════════════════════════════════════════════════════════════════════════════
SAFE_DISTANCE = 0.5                  # meters
MAX_SPEED     = 9.7                  # rad/s
MAX_INTEGRAL  = 2.0                  # integral windup clamp
HALF_CONE     = math.pi / 6         # 30° in radians  (pi/6 ≈ 0.524 rad)

# Zone center angles in radians
# Assumes 0 rad = forward. Verify with calibration test below if unsure.
#
#   CALIBRATION TEST — run this once on hardware to confirm forward direction:
#       scan = lidarsensor.get()
#       print(min(scan, key=scan.get))   # prints angle of nearest object
#   Point the robot at a close wall. Whatever angle prints is your forward angle.
#   If it is not 0, set FORWARD_OFFSET to that value.
#
FORWARD_OFFSET = 0.0                 # radians — adjust if 0 is not forward
FRONT_CENTER   = 0.0   + FORWARD_OFFSET
LEFT_CENTER    = -math.pi / 2 + FORWARD_OFFSET    # -90° (π/2 rad to the left)
RIGHT_CENTER   =  math.pi / 2 + FORWARD_OFFSET    # +90° (π/2 rad to the right)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LIDAR ZONE HELPER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# get_zone_min() is a shared utility used by both get_front_distance() and
# get_turn_bias(). It takes the full LiDAR scan and filters it down to only
# the readings that fall within a cone of interest, then returns the closest one.
#
# Purpose:
#   For every reading in the scan, check whether the angular difference
#   between that reading's angle and the zone center falls within half_width.
#   The difference calculation handles wraparound (e.g. near ±π boundary)
#   This maps any angle difference into the range [-π, +π].
#
# Returns:
#   The minimum (closest) distance found in the zone.
#   If no valid readings exist in the zone, returns SAFE_DISTANCE * 3
#   (a large number meaning "no obstacle detected").
# ═════════════════════════════════════════════════════════════════════════════
def get_zone_min(scan, center, half_width=HALF_CONE):
    readings = []
    for angle, dist in scan.items():
        diff = (angle - center + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) <= half_width and dist > 0:
            readings.append(dist)
    return min(readings) if readings else SAFE_DISTANCE * 3


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FRONT DISTANCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# get_front_distance() returns how far (in meters) the nearest obstacle is
# directly in front of the robot within a ±30° forward cone.
#
# This value feeds directly into the PID as the "measured output" —
#
# When front_dist > SAFE_DISTANCE → error is positive → robot drives forward.
# When front_dist < SAFE_DISTANCE → error is negative → robot slows/stops.
# ═════════════════════════════════════════════════════════════════════════════
def get_front_distance():
    scan = lidarsensor.get()
    return get_zone_min(scan, center=FRONT_CENTER)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TURN BIAS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# get_turn_bias() tells the robot which direction to steer when it needs to
# avoid an obstacle. Rreads the left and right LiDAR zones and compares them.
#
# Logic:
#   left_dist  = closest object in the left zone  (-90° ± 30°)
#   right_dist = closest object in the right zone (+90° ± 30°)
#   bias = right_dist - left_dist
#
#   If bias is POSITIVE → more space on the right → steer right
#   If bias is NEGATIVE → more space on the left  → steer left
#   If bias is ZERO     → equal clearance both sides → go straight
#
# This value is multiplied by kp_turn in the main loop to produce turn_speed.
# ═════════════════════════════════════════════════════════════════════════════
def get_turn_bias():
    scan       = lidarsensor.get()
    left_dist  = get_zone_min(scan, center=LEFT_CENTER)
    right_dist = get_zone_min(scan, center=RIGHT_CENTER)
    return right_dist - left_dist


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN CONTROL LOOP
# ─────────────────────────────────────────────────────────────────────────────
#
# Each iteration of the while loop:
#   1. Reads LiDAR → gets front_dist and turn_bias
#   2. Computes dt, error, and derivative of error
#   3. Accumulates integral with windup protection
#   4. Computes total PID control effort u
#   5. Maps u → forward_speed and turn_bias → turn_speed
#   6. Combines them into left/right wheel targets (pdTargets)
#   7. Reads current wheel speeds from encoders (pdCurrents)
#   8. Calls driveClosedLoop to apply the targets with motor feedback
#   9. Logs data to CSV
#  10. Sleeps 50ms to run at 20Hz 
# ═════════════════════════════════════════════════════════════════════════════
def loop_obstacle_avoid():
    count = 0

    # t0 and t1 store consecutive timestamps so we can compute dt each loop.
    # dt is needed for both the derivative term (de/dt) and integral (e * dt).
    t0 = 0
    t1 = 1

    # ── Error tracking ──────────────────────────────────
    # Three generations of error are stored to compute the derivative.
    # e1 = current error, e0 = previous, e00 = two steps ago.
    e00 = 0
    e0  = 0
    e1  = 0

    dt       = 0
    de_dt    = 0
    integral = 0.0

    while True:
        count += 1

        # ── SENSE ─────────────────────────────────────────────────────────────
        # Read the LiDAR. front_dist is our "measured output" and turn_bias tells us which side has more clearance.
        front_dist = get_front_distance()
        turn_bias  = get_turn_bias()

        # ── TIME UPDATE ─────────────────────────────────
        # Shift timestamps forward and compute elapsed time since last iteration.
        # The guard (else 0.001) prevents division by zero on the first loop.
        t0 = t1
        t1 = time.time()
        dt = t1 - t0 if t1 != t0 else 0.001

        # ── ERROR UPDATE ────────────────────────────────
        # Shift error history forward and compute the new current error.
        # Error = measured distance - safe distance
        # Positive error = robot is safely far from obstacle = drive forward
        # Negative error = robot is too close to obstacle   = slow down/stop
        e00 = e0
        e0  = e1
        e1  = front_dist - SAFE_DISTANCE

        # ── DERIVATIVE ───────────────────────────────────────────────────────
        # Rate of change of error. 
        # A large de_dt means the obstacle is approaching quickly.
        # kd dampens the response to prevent overshoot.
        de_dt = (e1 - e0) / dt

        # ── INTEGRAL with windup guard ────────────────────────────────────────
        # Accumulates error over time. Fixes steady-state error (robot stopping
        # slightly too close or too far). The clamp prevents the integral from
        # growing too large when the robot is stuck against a wall.
        integral += e1 * dt
        integral  = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, integral))

        # ── PID CONTROL EFFORT ───────────────────────────────────────────────
        # Combines all three terms into a single output value u.
        # u ranges roughly -1 to +1 before being scaled to wheel speed.
        u = (kp * e1) + (ki * integral) + (kd * de_dt)

        # ── MAP PID OUTPUT → FORWARD WHEEL SPEED ─────────────────────────────
        # u is scaled by MAX_SPEED to get a real rad/s value.
        # max(0.0, ...) clamps to zero — the robot never drives INTO an obstacle.
        # min(MAX_SPEED, ...) caps at the physical motor limit.
        forward_speed = max(0.0, min(MAX_SPEED, u * MAX_SPEED))

        # ── MAP TURN BIAS → TURN SPEED ────────────────────────────────────────
        # turn_bias (from get_turn_bias) is scaled by kp_turn.
        # This is a simple proportional controller just for steering direction.
        # It runs in parallel with — and independently from — the forward PID.
        turn_speed = kp_turn * turn_bias

        # ── COMBINE INTO WHEEL TARGETS ────────────────────────────────────────
        # Differential drive: to turn right, left wheel goes faster than right.
        # pd_left  = forward + turn  (left wheel speeds up to turn right)
        # pd_right = forward - turn  (right wheel slows to turn right)
        # Each wheel is clamped individually to the valid motor range.
        pd_left  = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_speed))
        pd_right = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_speed))

        pdTargets = np.array([pd_left, pd_right])

        # ── ACTUATE ──────────────────────────────────────
        # Read current wheel speeds from encoders, then call driveClosedLoop.
        # driveClosedLoop handles the low-level motor PWM adjustment internally.
        kin.getPdCurrent()
        pdCurrents = kin.pdCurrents
        sc.driveClosedLoop(pdTargets, pdCurrents, np.array([de_dt, de_dt]))

        # ── LOGGING ──────────────────────────────────────
        # Logs up to 400 samples to CSV for post-run analysis in Excel.
        # Columns: [sample#, front_distance, safe_distance, error, PID_output]
        # On the first iteration the old file is cleared first.
        if count == 1:
            log.clear_file()
            log.csv_write([count, front_dist, SAFE_DISTANCE, e1, u])
        elif 1 < count <= 400:
            log.csv_write([count, front_dist, SAFE_DISTANCE, e1, u])

        # ── TIMING ───────────────────────────────────────
        # 50ms sleep = 20Hz loop rate. Consistent timing keeps dt stable,
        # which keeps the derivative and integral calculations accurate.
        time.sleep(0.05)

loop_obstacle_avoid()