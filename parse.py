import re

def parse_data(file_path):
    """
    Parses the data from the given file and organizes it into a structured dictionary.

    Args:
        file_path (str): Path to the input data file.

    Returns:
        dict: A dictionary containing parsed data including weights, days, specializations, wards, 
        and patients.
    """
    data = {
        "weights": {},  # Weights for overtime, undertime, and delay
        "days": 0,  # Number of days in the planning period
        "specializations": [],  # List of specializations
        "wards": [],  # List of wards
        "patients": []  # List of patients
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("Weight_overtime"):
            data["weights"]["overtime"] = int(line.split(":")[-1].strip())
        elif line.startswith("Weight_undertime"):
            data["weights"]["undertime"] = int(line.split(":")[-1].strip())
        elif line.startswith("Weight_delay"):
            data["weights"]["delay"] = int(line.split(":")[-1].strip())
        elif line.startswith("Days:"):
            data["days"] = int(line.split(":")[-1].strip())
        elif re.match(r'^[A-Z]{3}\t\d+\.\d+', line):
            parts = line.strip().split()
            data["specializations"].append({
                "id": parts[0],
                "scaling_factor": float(parts[1]),
                "OT_availability": list(map(int, parts[2].split(';')))
            })
        elif line.startswith("Wards:"):
            ward_start = lines.index(line) + 1
        elif line.startswith("Patients:"):
            patient_start = lines.index(line) + 1

    for line in lines[ward_start:patient_start - 1]:
        parts = line.strip().split()
        data["wards"].append({
            "id": parts[0],
            "bed_capacity": int(parts[1]),
            "workload_capacity": float(parts[2]),
            "major_specialization": parts[3],
            "minor_specializations": parts[4] if parts[4] != "NONE" else [],
            "carryover_patients": list(map(int, parts[5].split(';'))),
            "carryover_workload": list(map(float, parts[6].split(';')))
        })
    
    
    specialization_index_map = {spec["id"]: idx for idx, spec in enumerate(data["specializations"])}

    for line in lines[patient_start:]:
        parts = line.strip().split()
        data["patients"].append({
            "id": parts[0],
            "specialization": specialization_index_map[parts[1]],
            "earliest_admission": int(parts[2]),
            "latest_admission": int(parts[3]),
            "length_of_stay": int(parts[4]),
            "surgery_duration": int(parts[5]),
            "workload_per_day": list(map(float, parts[6].split(';')))
        })

    return data

'''
data = parse_data("dataset/s0m0.dat")

print(f"Days in planning period: {data['days']}")
print(f"First ward details: {data['wards'][0]}")
print(f"First patient details: {data['patients'][0]['specialization']}")'
'''