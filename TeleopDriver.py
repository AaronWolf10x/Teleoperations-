# DON'T CHANGE ANY OF THE BELOW; NECESSARY FOR JOINING SIMULATION
import os, sys, time, datetime, traceback
import spaceteams as sc
import numpy as np

def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    sc.logger_fatal(error_message)
    sys.exit(1)

sys.excepthook = custom_exception_handler
sc.connect_to_sim(sys.argv)
# DON'T CHANGE ANY OF THE ABOVE; NECESSARY FOR JOINING SIMULATION
################################################################

HAS_KEYBOARD = False
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    sc.OnScreenLogMessage(
        r"[ControlRobot Warning] 'keyboard' library not found! WASD teleop disabled. Run 'pip install keyboard' to enable. For example, C:\\Users\\Neil\\AppData\\Local\\SimDynamX\\STPro\\Versions\\v0.54.2\\py\\Scripts>python.exe -m pip install keyboard",
        "Warning",
        sc.Severity.Warning
    )

this = sc.GetThisSystem()

# Pull entities configured in the simconf
Excavator: sc.Entity = this.GetParam(sc.VarType.entityRef, "Excavator")

if this.HasParam(sc.VarType.entityRef, "MoonFrame"):
    Planet: sc.Entity = this.GetParam(sc.VarType.entityRef, "MoonFrame")
elif this.HasParam(sc.VarType.entityRef, "MarsFrame"):
    Planet: sc.Entity = this.GetParam(sc.VarType.entityRef, "MarsFrame")
else:
    Planet: sc.Entity = this.GetParam(sc.VarType.entityRef, "Planet")

planetFrame = Planet.GetBodyFixedFrame()
robotFrame = Excavator.GetBodyFixedFrame()

MOVE_SPEED = 3.0       # Linear speed (m/s)
ROT_SPEED = 1.0        # Yaw rate (rad/s)
DT = 0.02              # Tick interval (s)

# Throttle timer to prevent log flooding at 50 Hz
last_log_time = 0.0
LOG_INTERVAL = 0.5     # Minimum seconds between repeated log messages

# True only while a drive command is active, so we send one final
# zero-velocity update on key release and then stop writing the location.
was_moving = False

# ==============================================================================
# TERRAIN FOLLOWING (DEM sampling, same pattern as LunarROS_Comm.py)
# ==============================================================================
ENABLE_GROUND_FOLLOW = True   # False = old behavior (flat driving, no DEM)

# Height (m) of the excavator's reference point above the terrain surface.
# Raise it if the wheels sink into the ground, lower it if it floats.
VEHICLE_HEIGHT_OFFSET = 0.10

# Tilt the excavator to match the slope. Turn on AFTER confirming driving works;
# the quaternion convention of AlignToGround is untested in this script.
ALIGN_TO_SLOPE = True

LOD_LEVEL = 19          # Matches SnapAssetToPlanet in the simconf

# Same datasets and priorities as SnapAssetToPlanet in the simconf
DEM_LAYERS = [
    ("Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", 0.0),
    ("Moon/Local/SouthPole/LDEM_Site01-Site-04_upscale_craters", 1.0),
]

# Planet radius: read from the planet entity if available, else standard Moon radius
if Planet.HasParam(sc.VarType.double, ["#Planet", "General", "Radius_m"]):
    planet_radius_m = Planet.GetParam(sc.VarType.double, ["#Planet", "General", "Radius_m"])
else:
    planet_radius_m = 1737400.0

demStore = None
if ENABLE_GROUND_FOLLOW:
    try:
        demStore = sc.ProcPlanet.DataStore()
        for dataset_path, priority in DEM_LAYERS:
            real_path = sc.path_utils.AssetPathToReal(sc.AssetType.PlanetData, dataset_path)
            demStore.AddGeoBinAltimetryLayer(priority, real_path, sc.ProcPlanet.GeoBin_Extra_Args())
            sc.logger_info(f"[TeleopDriver] Loaded DEM layer: {dataset_path} (priority {priority})")
        demStore.SetWritable(False)
        sc.OnScreenLogMessage("[TeleopDriver] DEM terrain loaded: ground following ON", "Teleop", sc.Severity.Info)
    except Exception as e:
        sc.logger_warn(f"[TeleopDriver] Could not load DEM: {e}. Ground following OFF.")
        sc.OnScreenLogMessage("[TeleopDriver] DEM failed to load: ground following OFF (see log)", "Teleop", sc.Severity.Warning)
        demStore = None

sample_fail_logged = False
align_fail_logged = False

def sample_ground(loc):
    """Return (location on the terrain surface plus ride height, surface normal).
    Falls back to (loc, None) if the DEM is unavailable or the query fails."""
    global sample_fail_logged
    if demStore is None:
        return loc, None
    try:
        ground_pos, surface_normal = sc.ProcPlanet.SampleGround(
            demStore, loc, planet_radius_m, 0.0, LOD_LEVEL
        )
        ground_pos = np.asarray(ground_pos, dtype=float).flatten()
        normal = np.asarray(surface_normal, dtype=float).flatten()
        n_mag = np.linalg.norm(normal)
        if n_mag < 1e-4:
            return loc, None
        normal = normal / n_mag
        return ground_pos + normal * VEHICLE_HEIGHT_OFFSET, normal
    except Exception as e:
        if not sample_fail_logged:
            sc.logger_warn(f"[TeleopDriver] SampleGround failed (logged once): {e}")
            sample_fail_logged = True
        return loc, None

def set_rotation_from_quat(qx, qy, qz, qw):
    """Convert quaternion [x, y, z, w] to a 3x3 DCM and apply it."""
    dcm = np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx**2 + qy**2)]
    ])
    Excavator.setRotation(sc.frames.FramedRot(dcm, planetFrame))

ground_ok = False

while True:
    if HAS_KEYBOARD:
        # 1. Read current location and orientation
        current_loc = Excavator.getLocation().WRT_ExprIn(planetFrame)
        # Quat_WRT returns [x, y, z, w]
        current_quat_xyzw = Excavator.getRotation().Quat_WRT(planetFrame)
        qx, qy, qz, qw = current_quat_xyzw

        # 2. Local surface normal vector ("up")
        loc_norm = np.linalg.norm(current_loc)
        up_vec = current_loc / loc_norm if loc_norm > 0 else np.array([0.0, 0.0, 1.0])

        # 3. Handle Steering (Yaw rotation around surface up-vector)
        yaw_delta = 0.0
        turn_action = None
        if keyboard.is_pressed('a'):
            yaw_delta += ROT_SPEED * DT
            turn_action = "Turn Left (A)"
        if keyboard.is_pressed('d'):
            yaw_delta -= ROT_SPEED * DT
            turn_action = "Turn Right (D)"

        if abs(yaw_delta) > 1e-5:
            half_angle = yaw_delta * 0.5
            sin_h = np.sin(half_angle)
            cos_h = np.cos(half_angle)

            # Rotation quaternion [x, y, z, w]
            dq_x = up_vec[0] * sin_h
            dq_y = up_vec[1] * sin_h
            dq_z = up_vec[2] * sin_h
            dq_w = cos_h

            # Quaternion multiplication: delta_q * current_rot
            new_qx = dq_w * qx + dq_x * qw + dq_y * qz - dq_z * qy
            new_qy = dq_w * qy - dq_x * qz + dq_y * qw + dq_z * qx
            new_qz = dq_w * qz + dq_x * qy - dq_y * qx + dq_z * qw
            new_qw = dq_w * qw - dq_x * qx - dq_y * qy - dq_z * qz

            # Normalize quaternion
            q_norm = np.sqrt(new_qx**2 + new_qy**2 + new_qz**2 + new_qw**2)
            qx, qy, qz, qw = new_qx / q_norm, new_qy / q_norm, new_qz / q_norm, new_qw / q_norm

            set_rotation_from_quat(qx, qy, qz, qw)

        # 4. Extract forward pointing vector from quaternion (+X body axis)
        forward_vec = np.array([
            1.0 - 2.0 * (qy**2 + qz**2),
            2.0 * (qx*qy + qw*qz),
            2.0 * (qx*qz - qw*qy)
        ])

        # Project tangent to the spherical terrain surface
        forward_vec = forward_vec - np.dot(forward_vec, up_vec) * up_vec
        fwd_norm = np.linalg.norm(forward_vec)
        if fwd_norm > 1e-4:
            forward_vec = forward_vec / fwd_norm

        # 5. Handle Forward / Backward Translation
        linear_cmd = 0.0
        move_action = None
        if keyboard.is_pressed('w'):
            linear_cmd += MOVE_SPEED
            move_action = "Drive Forward (W)"
        if keyboard.is_pressed('s'):
            linear_cmd -= MOVE_SPEED
            move_action = "Drive Reverse (S)"

        vel_vec = forward_vec * linear_cmd

        # 6. Apply updated kinematic state. Only write the location while driving
        #    (plus one final update on key release), so nothing else fights us idle.
        zero_vec = np.array([0.0, 0.0, 0.0])
        if linear_cmd != 0.0:
            # Driving: advance from the current location, then put it on the terrain
            new_loc = current_loc + vel_vec * DT
            new_loc, ground_normal = sample_ground(new_loc)
            ground_ok = ground_normal is not None

            # Optional: tilt to the slope
            if ALIGN_TO_SLOPE and ground_normal is not None:
                try:
                    q = np.asarray(sc.ProcPlanet.AlignToGround(ground_normal, forward_vec), dtype=float).flatten()
                    set_rotation_from_quat(q[0], q[1], q[2], q[3])
                except Exception as e:
                    if not align_fail_logged:
                        sc.logger_warn(f"[TeleopDriver] AlignToGround failed (logged once): {e}")
                        align_fail_logged = True

            rva = sc.frames.rva_struct(new_loc, vel_vec, zero_vec)
            Excavator.setLocVelAcc(sc.frames.FramedLocVelAcc(rva, planetFrame))
            was_moving = True
        elif was_moving:
            # Key released: one zero-velocity update on the terrain, then stop writing
            rest_loc, _ = sample_ground(current_loc)
            rva = sc.frames.rva_struct(rest_loc, zero_vec, zero_vec)
            Excavator.setLocVelAcc(sc.frames.FramedLocVelAcc(rva, planetFrame))
            was_moving = False

        # 7. Log User Input Actions (r = distance from planet center, ground = DEM sample ok)
        now = time.time()
        actions = [a for a in [move_action, turn_action] if a is not None]
        if actions and (now - last_log_time >= LOG_INTERVAL):
            msg = f"[ControlRobot Teleop] Input detected: {', '.join(actions)} r={loc_norm:.2f} ground={'ok' if ground_ok else 'none'}"
            # Write to Compute_Server console/logger
            sc.logger_info(msg)
            # Display directly inside the simulation screen
            sc.OnScreenLogMessage(msg, "Teleop", sc.Severity.Info)
            last_log_time = now

    time.sleep(DT)