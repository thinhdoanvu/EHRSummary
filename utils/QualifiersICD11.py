import os
import re

class ICD11Qualifiers:
    def __init__(self):
        # 1. Severity
        self.severity_map = {
            "no": "No", "none": "No", "mild": "Mild",
            "moderate": "Moderate", "severe": "Severe",
            "profound": "Profound", "extreme": "Extreme",
            "life-threatening": "Life-threatening"
        }
        self.severity_codes = {
            "No": "XK0G", "Mild": "XS5W", "Moderate": "XS2R",
            "Severe": "XS0T", "Profound": "XS1M",
            "Extreme": "XS6V", "Life-threatening": "XS7A"
        }

        # 2. Laterality
        self.laterality_map = {
            "left": "Left", "right": "Right", "bilateral": "Bilateral",
            "unilateral": "Unilateral", "unspecified": "Unspecified"
        }
        self.laterality_codes = {
            "Left": "XK8G", "Right": "XK9J",
            "Bilateral": "XK70", "Unilateral": "XK71",
            "Unspecified": "XK9K"
        }

        # 3. Temporal course
        self.temporal_map = {
            "acute": "Acute", "subacute": "Subacute", "chronic": "Chronic",
            "recurrent": "Recurrent", "persistent": "Persistent",
            "intermittent": "Intermittent", "progressive": "Progressive",
            "relapsing-remitting": "Relapsing-remitting"
        }
        self.temporal_codes = {
            "Acute": "XT5R", "Subacute": "XT1L", "Chronic": "XT8W",
            "Recurrent": "XT2V", "Persistent": "XT3B",
            "Intermittent": "XT4C", "Progressive": "XT6D",
            "Relapsing-remitting": "XT7E"
        }

        # 4. Location / Topology
        self.location_map = {
            "upper": "Upper", "lower": "Lower", "anterior": "Anterior",
            "posterior": "Posterior", "medial": "Medial", "lateral": "Lateral",
            "proximal": "Proximal", "distal": "Distal", "central": "Central",
            "peripheral": "Peripheral", "midline": "Midline", "diffuse": "Diffuse",
            "multiple": "Multiple sites", "overlapping": "Overlapping sites"
        }
        self.location_codes = {
            "Upper": "XA4650", "Lower": "XA4660", "Anterior": "XA4640",
            "Posterior": "XA4670", "Medial": "XA4680", "Lateral": "XA4690",
            "Proximal": "XA46A0", "Distal": "XA46B0", "Central": "XA46C0",
            "Peripheral": "XA46D0", "Midline": "XA46E0", "Diffuse": "XA46F0",
            "Multiple sites": "XA46G0", "Overlapping sites": "XA46H0"
        }

        # 5. Aetiology
        self.aetiology_map = {
            "infectious": "Infectious agent", "trauma": "Trauma",
            "congenital": "Congenital", "iatrogenic": "Iatrogenic",
            "environmental": "Environmental exposure"
        }
        self.aetiology_codes = {
            "Infectious agent": "XN0A", "Trauma": "XN0B",
            "Congenital": "XN0C", "Iatrogenic": "XN0D",
            "Environmental exposure": "XN0E"
        }

        # 6. Morphology
        self.morphology_map = {
            "hypertrophy": "Hypertrophy", "atrophy": "Atrophy",
            "necrosis": "Necrosis", "fibrosis": "Fibrosis",
            "inflammation": "Inflammation", "neoplasm": "Neoplasm"
        }
        self.morphology_codes = {
            "Hypertrophy": "XM0A", "Atrophy": "XM0B",
            "Necrosis": "XM0C", "Fibrosis": "XM0D",
            "Inflammation": "XM0E", "Neoplasm": "XM0F"
        }

        # 7. Histopathology
        self.histopathology_map = {
            "carcinoma": "Carcinoma", "sarcoma": "Sarcoma",
            "adenoma": "Adenoma", "dysplasia": "Dysplasia",
            "hyperplasia": "Hyperplasia"
        }
        self.histopathology_codes = {
            "Carcinoma": "XH0A", "Sarcoma": "XH0B",
            "Adenoma": "XH0C", "Dysplasia": "XH0D",
            "Hyperplasia": "XH0E"
        }

        # 8. Dimensions of injury
        self.injury_map = {
            "laceration": "Laceration", "contusion": "Contusion",
            "fracture": "Fracture", "burn": "Burn", "crush": "Crush injury"
        }
        self.injury_codes = {
            "Laceration": "XI0A", "Contusion": "XI0B",
            "Fracture": "XI0C", "Burn": "XI0D", "Crush injury": "XI0E"
        }

        # 9. External causes
        self.external_map = {
            "traffic": "Road traffic accident", "fall": "Fall",
            "drowning": "Drowning", "poisoning": "Poisoning",
            "fire": "Fire", "assault": "Assault"
        }
        self.external_codes = {
            "Road traffic accident": "XE0A", "Fall": "XE0B",
            "Drowning": "XE0C", "Poisoning": "XE0D",
            "Fire": "XE0E", "Assault": "XE0F"
        }

        # 10. Consciousness
        self.consciousness_map = {
            "alert": "Alert", "drowsy": "Drowsy",
            "stupor": "Stupor", "coma": "Coma",
            "vegetative": "Vegetative state"
        }
        self.consciousness_codes = {
            "Alert": "XC0A", "Drowsy": "XC0B",
            "Stupor": "XC0C", "Coma": "XC0D",
            "Vegetative state": "XC0E"
        }

        # 11. Substances
        self.substance_map = {
            "alcohol": "Alcohol", "tobacco": "Tobacco",
            "cannabis": "Cannabis", "opioids": "Opioids",
            "stimulants": "Stimulants", "hallucinogens": "Hallucinogens"
        }
        self.substance_codes = {
            "Alcohol": "XSUB1", "Tobacco": "XSUB2",
            "Cannabis": "XSUB3", "Opioids": "XSUB4",
            "Stimulants": "XSUB5", "Hallucinogens": "XSUB6"
        }

        # 12. Devices / Equipment
        self.device_map = {
            "pacemaker": "Pacemaker", "prosthesis": "Prosthesis",
            "catheter": "Catheter", "ventilator": "Ventilator",
            "implant": "Implant"
        }
        self.device_codes = {
            "Pacemaker": "XD0A", "Prosthesis": "XD0B",
            "Catheter": "XD0C", "Ventilator": "XD0D",
            "Implant": "XD0E"
        }

        # 13. Functioning / Disability qualifiers
        self.functioning_map = {
            "no limitation": "No limitation", "mild": "Mild limitation",
            "moderate": "Moderate limitation", "severe": "Severe limitation",
            "complete": "Complete limitation"
        }
        self.functioning_codes = {
            "No limitation": "XF0A", "Mild limitation": "XF0B",
            "Moderate limitation": "XF0C", "Severe limitation": "XF0D",
            "Complete limitation": "XF0E"
        }

    def extract_from_text(self, text):
        """Extract ICD-11 extension categories from raw string"""
        found = {
            "Severity": set(),
            "Laterality": set(),
            "Temporal": set(),
            "Location": set(),
            "Aetiology": set(),
            "Morphology": set(),
            "Histopathology": set(),
            "Injury": set(),
            "External": set(),
            "Consciousness": set(),
            "Substance": set(),
            "Device": set(),
            "Functioning": set()
        }
        if not text:
            return found

        words = re.findall(r'\w+', text.lower())

        for word in words:
            if word in self.severity_map:
                found["Severity"].add(self.severity_map[word])
            if word in self.laterality_map:
                found["Laterality"].add(self.laterality_map[word])
            if word in self.temporal_map:
                found["Temporal"].add(self.temporal_map[word])
            if word in self.location_map:
                found["Location"].add(self.location_map[word])
            if word in self.aetiology_map:
                found["Aetiology"].add(self.aetiology_map[word])
            if word in self.morphology_map:
                found["Morphology"].add(self.morphology_map[word])
            if word in self.histopathology_map:
                found["Histopathology"].add(self.histopathology_map[word])
            if word in self.injury_map:
                found["Injury"].add(self.injury_map[word])
            if word in self.external_map:
                found["External"].add(self.external_map[word])
            if word in self.consciousness_map:
                found["Consciousness"].add(self.consciousness_map[word])
            if word in self.substance_map:
                found["Substance"].add(self.substance_map[word])
            if word in self.device_map:
                found["Device"].add(self.device_map[word])
            if word in self.functioning_map:
                found["Functioning"].add(self.functioning_map[word])

        # Convert sets to lists for output
        return {k: list(v) for k, v in found.items()}

    def map_to_codes(self, qualifiers_found):
        """Convert categories to ICD-11 X-codes for all 13 groups"""
        codes = {
            "Severity": [],
            "Laterality": [],
            "Temporal": [],
            "Location": [],
            "Aetiology": [],
            "Morphology": [],
            "Histopathology": [],
            "Injury": [],
            "External": [],
            "Consciousness": [],
            "Substance": [],
            "Device": [],
            "Functioning": []
        }

        for sev in qualifiers_found.get("Severity", []):
            if sev in self.severity_codes:
                codes["Severity"].append(self.severity_codes[sev])

        for lat in qualifiers_found.get("Laterality", []):
            if lat in self.laterality_codes:
                codes["Laterality"].append(self.laterality_codes[lat])

        for temp in qualifiers_found.get("Temporal", []):
            if temp in self.temporal_codes:
                codes["Temporal"].append(self.temporal_codes[temp])

        for loc in qualifiers_found.get("Location", []):
            if loc in self.location_codes:
                codes["Location"].append(self.location_codes[loc])

        for aet in qualifiers_found.get("Aetiology", []):
            if aet in self.aetiology_codes:
                codes["Aetiology"].append(self.aetiology_codes[aet])

        for morph in qualifiers_found.get("Morphology", []):
            if morph in self.morphology_codes:
                codes["Morphology"].append(self.morphology_codes[morph])

        for hist in qualifiers_found.get("Histopathology", []):
            if hist in self.histopathology_codes:
                codes["Histopathology"].append(self.histopathology_codes[hist])

        for inj in qualifiers_found.get("Injury", []):
            if inj in self.injury_codes:
                codes["Injury"].append(self.injury_codes[inj])

        for ext in qualifiers_found.get("External", []):
            if ext in self.external_codes:
                codes["External"].append(self.external_codes[ext])

        for con in qualifiers_found.get("Consciousness", []):
            if con in self.consciousness_codes:
                codes["Consciousness"].append(self.consciousness_codes[con])

        for sub in qualifiers_found.get("Substance", []):
            if sub in self.substance_codes:
                codes["Substance"].append(self.substance_codes[sub])

        for dev in qualifiers_found.get("Device", []):
            if dev in self.device_codes:
                codes["Device"].append(self.device_codes[dev])

        for func in qualifiers_found.get("Functioning", []):
            if func in self.functioning_codes:
                codes["Functioning"].append(self.functioning_codes[func])

        return codes

    def extract_from_file(self, filepath):
        """Extract ICD-11 extension categories from a text file"""
        found = {
            "Severity": set(),
            "Laterality": set(),
            "Temporal": set(),
            "Location": set(),
            "Aetiology": set(),
            "Morphology": set(),
            "Histopathology": set(),
            "Injury": set(),
            "External": set(),
            "Consciousness": set(),
            "Substance": set(),
            "Device": set(),
            "Functioning": set()
        }

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        words = re.findall(r'\w+', text.lower())

        for word in words:
            if word in self.severity_map:
                found["Severity"].add(self.severity_map[word])
            if word in self.laterality_map:
                found["Laterality"].add(self.laterality_map[word])
            if word in self.temporal_map:
                found["Temporal"].add(self.temporal_map[word])
            if word in self.location_map:
                found["Location"].add(self.location_map[word])
            if word in self.aetiology_map:
                found["Aetiology"].add(self.aetiology_map[word])
            if word in self.morphology_map:
                found["Morphology"].add(self.morphology_map[word])
            if word in self.histopathology_map:
                found["Histopathology"].add(self.histopathology_map[word])
            if word in self.injury_map:
                found["Injury"].add(self.injury_map[word])
            if word in self.external_map:
                found["External"].add(self.external_map[word])
            if word in self.consciousness_map:
                found["Consciousness"].add(self.consciousness_map[word])
            if word in self.substance_map:
                found["Substance"].add(self.substance_map[word])
            if word in self.device_map:
                found["Device"].add(self.device_map[word])
            if word in self.functioning_map:
                found["Functioning"].add(self.functioning_map[word])

        # Convert sets to lists for output
        return {k: list(v) for k, v in found.items()}
