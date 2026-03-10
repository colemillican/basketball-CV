# Under-$1,000 Basketball CV Build (1 Hoop)

Date recorded: February 16, 2026
Goal: Lowest-cost setup that still works for real-time make/miss + shot location + live display.

## Assumptions
- You already have a wall TV/display in the gym.
- One hoop deployment.
- Two-camera design:
  - Rim camera for make/miss
  - Wide camera for shot location
- TV-only interaction for MVP (no separate touchscreen kiosk).

## Parts List (Target: Under $1,000)
1. Jetson Orin Nano Super Dev Kit - $249
2. 2x Reolink RLC-811A PoE zoom cameras - ~$110 each (~$220 total)
3. TP-Link TL-SG1005P PoE switch - $44.99
4. DisplayPort-to-HDMI adapter/cable for TV connection - $10 to $20
5. Samsung 128GB microSD - $25.99
6. APC BE425M UPS - $61.97
7. Cat6 bulk cable (250 ft) - $81.89
8. Wireless keyboard + mouse (USB dongle) - $0 if already owned, ~$15.99 if needed
9. Mounts + RJ45 ends + junction boxes + misc install hardware - $120 (allowance)

Estimated subtotal (with existing keyboard/mouse): about $814 to $824
Estimated subtotal (if buying keyboard/mouse): about $830 to $840
Estimated total with tax/shipping buffer (~12-15%): about $912 to $966

## Why this works
- PoE wiring is practical for gym cable distances and cleaner than long USB runs.
- 25-30 FPS is acceptable for budget MVP make/miss and shot-chart operation.
- Jetson Orin Nano Super is enough for efficient YOLO/TensorRT inference at this scale.
- TV-only UI removes kiosk complexity and reduces failure points.

## Gym Installation Layout
1. Rim camera: mounted behind/above backboard, tight view of rim/net.
2. Wide camera: mounted high near sideline/half-court for court mapping.
3. Both cameras run Cat6 to PoE switch near wall compute cabinet.
4. Jetson + PoE switch + UPS in cabinet near TV.
5. TV connected to Jetson (DisplayPort-to-HDMI) for login and live results.
6. Wireless keyboard/mouse USB receiver plugged into Jetson for interaction.

## MVP Interaction Model (No Kiosk Tablet)
- Login, account selection, and mode selection happen on the TV UI.
- Player or coach interacts with UI using wireless keyboard/mouse.
- Same screen transitions into live workout dashboard and end-of-session results.

## Current Software Capability (As Built)
- Local TV UI flow is working end-to-end:
  - login/player select
  - mode select
  - live session stats
  - post-session results screen with shot chart image
- Mock mode is implemented for no-hardware testing (`runtime.backend: mock`).
- Real mode is implemented for camera-based detection (`runtime.backend: real`).
- Make/miss logic was upgraded to stricter sequence logic:
  - ball seen above rim
  - rim-zone confirmation
  - downward pass through net lane
- Debug overlay is available in OpenCV runtime (`debug.show_overlay: true`) to tune rim/net regions.

## Important Scope Note (Current Limitation)
- Hardware plan assumes 2 cameras (rim + wide), but current software runtime in `src/runtime.py` consumes one `video.source` at a time.
- This means current real-mode MVP should be treated as single-stream for initial gym validation.
- Two-stream fusion (rim stream + wide stream together) is a planned next phase.

## What This MVP Is Optimized For
- One shooter at a time.
- One active ball at a time.
- Fixed camera placement after calibration.
- Controlled workouts for reliability benchmarking.

## What To Test First In Real Gym
1. Session flow reliability:
   - Start session, run 20-30 minutes, end session, verify output files.
2. Make/miss accuracy:
   - Compare system counts to manual charting for at least 100 shots.
3. Shot chart reasonableness:
   - Confirm mapped dots are in expected court zones.
4. Calibration stability:
   - Verify no major drift when camera remains fixed.
5. Edge-case behavior:
   - Track rim-outs, short rim bounces, and blocked views.

## Key Runtime/Config Modes
- `runtime.backend: mock`
  - Generates simulated shots and full outputs for UI validation.
- `runtime.backend: real`
  - Uses real camera stream and detector/tracker/shot logic.
- `debug.show_overlay: true`
  - Draws scoring circle, entry line, net lane, and state counters for tuning.

## Basic Camera Settings (starting point)
- Rim cam: 1080p, 25 FPS, fixed exposure where possible.
- Wide cam: 1080p, 15-20 FPS.
- Keep both camera positions fixed after calibration.

## Recommended Rim Camera Geometry
- Mount on support structure near backboard top (not on glass).
- Rough starting target:
  - 2-4 feet above rim center
  - 15-25 degrees downward tilt
- Keep full rim + full net in frame with margin above and below.
- Use optical zoom to achieve adequate rim pixel size without remounting.

## Rim Camera Mounting Plan (RLC-811A)
Goal: prevent angle drift, prevent falls, and avoid interference with other gym activities.

### Mounting Strategy
1. Preferred mount point:
   - Fixed building steel/truss above or behind hoop (best if goals fold up).
2. Fallback mount point:
   - Goal support structure only if fixed building steel is not practical.
3. Do not mount to backboard glass.

### Mechanical Approach
1. Attach a short Unistrut section to steel using two beam clamps.
2. Mount a Reolink-compatible junction box/adapter plate for the 811A housing.
3. Mount RLC-811A base to the adapter plate and lock camera angle.
4. Add:
   - threadlocker on hardware
   - nylock nuts
   - secondary stainless safety tether to independent anchor point
5. Add light vibration isolation (thin neoprene pad/washer) between structure and mount plate.

### Cable Routing Rules
1. Use Cat6 with strain relief near camera and at junction box.
2. Route along structure with P-clamps and ties; protect with loom near abrasion points.
3. Keep cable out of moving/pinch points if goal can retract.
4. If any moving section must be crossed, include protected service loop.

### Aiming/Locking Procedure
1. Aim for full rim + full net + small margin above rim and below net.
2. Set slight downward angle (about 15-25 degrees), avoid straight-down.
3. Fully tighten camera lock screws and bracket hardware.
4. Mark fasteners with paint pen so future slip is visible during inspection.

## Rim Mounting BOM (Per Camera)
1. Reolink-compatible junction box/adapter for RLC-811A, qty 1
2. Unistrut channel (1-5/8 in, 12-18 in length), qty 1
3. Beam clamps (Unistrut style), qty 2
4. Unistrut spring nuts + bolts + washers + nylock nuts, qty 1 set
5. Flat adapter plate (if needed), qty 1
6. Stainless safety tether cable + hardware, qty 1
7. Neoprene/rubber isolation washers or pad, qty 1 set
8. Cable clamps/P-clips + zip ties + split loom, qty 1 set
9. Threadlocker (blue), qty 1

Typical mounting-only cost per rim camera: about $100-$230 depending on hardware grade.

## Install-Day Mount Checklist
1. Confirm mount location has no collision with folded goal path.
2. Verify no glass drilling and no load on moving backboard parts.
3. Perform shake test after torqueing hardware.
4. Verify safety tether is independent of primary mount.
5. Run stream and confirm framing after 24 hours; re-check paint marks for any slip.

## Sources used
- Jetson Orin Nano Super: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
- Reolink RLC-811A product page/specs: https://reolink.com/us/product/rlc-811a/
- RLC-811A support/spec page: https://support.reolink.com/c/rlc-811a/
- TP-Link TL-SG1005P: https://www.bhphotovideo.com/c/product/1373960-REG/tp_link_tl_sg1005p_5_port_gigabit_desktop_switch.html
- Samsung 128GB microSD: https://www.bhphotovideo.com/c/product/1762872-REG/samsung_mb_md128sa_am_128gb_pro_plus.html
- APC BE425M UPS: https://www.bhphotovideo.com/c/product/1286400-REG/apc_be425m_back_ups_6_outlet_425va_ups.html
- Monoprice Ethernet cables: https://www.monoprice.com/pages/Ethernet_Cables

## Notes
- Prices can change quickly; verify current prices before checkout.
- Marketplace listings can be less predictable than authorized sellers.
- This document defines MVP phase goals; multi-ball and multi-shooter attribution are out of current phase scope.
