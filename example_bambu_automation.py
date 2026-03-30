"""
Example: Bambu Lab Automated Part Removal
This script demonstrates how to automate part removal from a Bambu Lab 3D printer

SETUP:
1. Configure your printer connection details below
2. Adjust robot positions for your setup
3. Run this script in Arctos Studio Python panel
"""

# ========== CONFIGURATION ==========
PRINTER_IP = "192.168.1.100"      # Your printer IP
PRINTER_SERIAL = "AC12309BH109"   # Your printer serial number
PRINTER_ACCESS_CODE = "12347890"  # Your printer access code

# Robot positions (adjust for your setup)
HOME_POSITION = [0, -45, 90, 0, 45, 0]
APPROACH_HEIGHT = 150  # mm above bed
PICK_HEIGHT = 50       # mm - height to grab part
DROP_POSITION = [400, 200, 100]  # X, Y, Z in mm

# ========== MAIN SCRIPT ==========

print("=" * 60)
print("BAMBU LAB AUTOMATED PART REMOVAL")
print("=" * 60)

# Step 1: Connect to printer
print("\n[1/6] Connecting to Bambu Lab printer...")
if connect_bambu(PRINTER_IP, PRINTER_SERIAL, PRINTER_ACCESS_CODE):
    print("[OK] Connected successfully")
else:
    print("✗ Connection failed - check your settings")
    exit()

# Step 2: Check printer status
print("\n[2/6] Checking printer status...")
status = get_bambu_status()
print(f"  Status: {status['status']}")
print(f"  File: {status['file']}")
print(f"  Progress: {status['progress']}%")

# Step 3: Wait for print to complete
print("\n[3/6] Waiting for print to complete...")
print("  (This will block until print finishes or timeout)")

if wait_for_bambu_print_complete(timeout=7200):  # 2 hour timeout
    print("[OK] Print completed successfully!")
else:
    print("✗ Print failed or timed out")
    disconnect_bambu()
    exit()

# Step 4: Move to safe position
print("\n[4/6] Moving to safe position...")
move_joints(HOME_POSITION)
open_gripper()
print("[OK] Ready to pick")

# Step 5: Pick part from bed
print("\n[5/6] Picking part from bed...")

# Calculate bed center position (adjust for your printer)
bed_center_x = 250  # mm
bed_center_y = 0    # mm

# Approach from above
print("  Moving above part...")
move_ee_to_world_pose(bed_center_x, bed_center_y, APPROACH_HEIGHT, 90, 0, 0)

# Lower to pick height
print("  Lowering to pick height...")
move_ee_to_world_pose(bed_center_x, bed_center_y, PICK_HEIGHT, 90, 0, 0)

# Close gripper to grab part
print("  Closing gripper...")
close_gripper()
time.sleep(1)  # Wait for gripper to close

# Lift part
print("  Lifting part...")
move_ee_to_world_pose(bed_center_x, bed_center_y, APPROACH_HEIGHT, 90, 0, 0)

print("[OK] Part picked successfully")

# Step 6: Move to drop location
print("\n[6/6] Moving to drop location...")
move_ee_to_world_pose(DROP_POSITION[0], DROP_POSITION[1], DROP_POSITION[2], 90, 0, 0)

# Release part
print("  Releasing part...")
open_gripper()
time.sleep(1)

# Return home
print("  Returning home...")
go_to_zero()

print("\n[OK] Part removal complete!")

# Disconnect from printer
disconnect_bambu()
print("[OK] Disconnected from printer")

print("\n" + "=" * 60)
print("AUTOMATION COMPLETE")
print("=" * 60)
