"""
Fix missing/invalid inertia in my_droid.xml (Franka Panda + Robotiq 2F-85).

Usage:
    python fix_mjcf_inertia.py path/to/my_droid.xml path/to/my_droid_fixed.xml
"""
import sys
import xml.etree.ElementTree as ET

# Franka Panda inertia data from the official MuJoCo Menagerie
# (mass kg, CoM pos m, diaginertia kg*m^2)
PANDA_INERTIA = {
    "panda":          dict(mass="1.0",        pos="0 0 0",                          diaginertia="0.01 0.01 0.01"),
    "panda_rootJoint":dict(mass="1.0",        pos="0 0 0",                          diaginertia="0.01 0.01 0.01"),
    "panda_joint1":   dict(mass="4.970684",   pos="-0.041018 -0.000140 0.049974",   diaginertia="0.703375 0.707198 0.009383"),
    "panda_joint2":   dict(mass="0.646926",   pos="0.003875 0.002081 0.000269",     diaginertia="0.007902 0.008229 0.002507"),
    "panda_joint3":   dict(mass="3.228604",   pos="-0.025586 0.000090 0.032776",    diaginertia="0.037265 0.036219 0.010830"),
    "panda_joint4":   dict(mass="3.587895",   pos="-0.005607 0.035791 -0.038990",   diaginertia="0.025850 0.019258 0.028409"),
    "panda_joint5":   dict(mass="1.225946",   pos="-0.011503 -0.041172 -0.038168",  diaginertia="0.035317 0.029537 0.008736"),
    "panda_joint6":   dict(mass="1.666555",   pos="0.060149 -0.014117 -0.010517",   diaginertia="0.005327 0.005028 0.007576"),
    "panda_joint7":   dict(mass="0.735522",   pos="0.010517 -0.010517 0.006600",    diaginertia="0.012152 0.012517 0.002122"),
    "panda_joint8":   dict(mass="0.09",       pos="0 0 0",                          diaginertia="0.001 0.001 0.001"),
    # Franka hand
    "panda_hand_joint":               dict(mass="0.73",  pos="-0.01 0 0.0584",  diaginertia="0.001 0.0025 0.0017"),
    # Robotiq 2F-85 fingers — small but above mjMINVAL
    "finger_joint":                   dict(mass="0.01",  pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
    "left_outer_finger_FixedJoint":   dict(mass="0.01",  pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
    "left_inner_finger_joint":        dict(mass="0.01",  pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
    "left_inner_finger_knuckle_joint":dict(mass="0.01",  pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
    "right_outer_knuckle_joint":      dict(mass="0.01",  pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
    "right_outer_finger_FixedJoint":  dict(mass="0.01",  pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
    "right_inner_finger_joint":       dict(mass="0.01",  pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
    "right_inner_finger_knuckle_joint":dict(mass="0.01", pos="0 0 0",           diaginertia="1e-4 1e-4 1e-4"),
}


def fix_inertia(input_path: str, output_path: str) -> None:
    ET.register_namespace("", "")
    tree = ET.parse(input_path)
    root = tree.getroot()

    fixed, skipped = [], []

    for body in root.iter("body"):
        name = body.get("name", "")
        if name not in PANDA_INERTIA:
            skipped.append(name)
            continue

        # Remove any existing (placeholder) inertial element
        for old in body.findall("inertial"):
            body.remove(old)

        d = PANDA_INERTIA[name]
        inertial = ET.Element("inertial")
        inertial.set("pos", d["pos"])
        inertial.set("mass", d["mass"])
        inertial.set("diaginertia", d["diaginertia"])
        body.insert(0, inertial)  # first child so it's easy to find
        fixed.append(name)

    ET.indent(tree, space="\t")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    print(f"Fixed  ({len(fixed)}): {', '.join(fixed)}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_mjcf_inertia.py <input.xml> <output.xml>")
        sys.exit(1)
    fix_inertia(sys.argv[1], sys.argv[2])