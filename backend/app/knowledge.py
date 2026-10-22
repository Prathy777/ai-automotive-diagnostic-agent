"""Small local automotive reference set used to ground assistant answers."""
import re

KNOWLEDGE = [
    {"id":"dtc-p0301", "type":"dtc", "title":"P0301 — Cylinder 1 misfire detected", "text":"A misfire means cylinder 1 is not contributing consistently. Common causes include ignition components, injector/fuel delivery, compression, or wiring. Diagnosis requires vehicle-specific checks."},
    {"id":"dtc-p0420", "type":"dtc", "title":"P0420 — Catalyst system efficiency below threshold", "text":"The engine controller sees catalyst efficiency below its calibrated threshold. Check for exhaust leaks, sensor behavior, and engine-running faults before replacing a catalytic converter."},
    {"id":"dtc-p0171", "type":"dtc", "title":"P0171 — System too lean, Bank 1", "text":"A lean condition means measured air is high relative to fuel. Possible causes include intake leaks, fuel delivery, airflow measurement, or exhaust leaks."},
    {"id":"dtc-p0128", "type":"dtc", "title":"P0128 — Coolant thermostat temperature below regulating temperature", "text":"The engine did not reach its expected operating temperature in the expected time. Thermostat and coolant temperature sensor behavior are common checks."},
    {"id":"light-abs", "type":"warning_light", "title":"ABS warning light", "text":"The anti-lock braking system reports a fault. Conventional braking may remain, but anti-lock assistance may be unavailable. Arrange prompt inspection and drive cautiously."},
    {"id":"component-abs", "type":"component", "title":"Anti-lock braking system (ABS)", "text":"ABS monitors wheel speed and can modulate brake pressure to help prevent wheel lock during hard braking."},
    {"id":"symptom-vibration", "type":"symptom", "title":"Engine vibration", "text":"Engine vibration can come from misfire, mounts, uneven combustion, or accessory faults. Vibration that tracks engine RPM suggests an engine-related cause; vibration that tracks road speed may point toward tires, wheels, or driveline."},
    {"id":"symptom-overheat", "type":"symptom", "title":"Overheating", "text":"Stop safely if the temperature warning is active or the gauge is in the red. Do not open a hot cooling system. Cooling fan, coolant level/leaks, thermostat, radiator flow, and pump operation require inspection."},
    {"id":"maintenance-trip", "type":"maintenance", "title":"Before a long trip", "text":"Check tire condition and pressure, fluid levels when safe, lights, wipers, spare tire/tools, and scheduled maintenance. Follow the owner's manual and do not ignore warning lights."},
    {"id":"symptom-fuel", "type":"symptom", "title":"Increasing fuel consumption", "text":"Fuel use can rise with tire underinflation, load, driving conditions, maintenance issues, dragging brakes, sensor faults, or engine misfire. Compare under similar conditions and scan for warning codes."},
]

def search_knowledge(query: str, limit: int = 4) -> list[dict[str, str]]:
    dtc = re.search(r"\bP\d{4}\b", query, re.IGNORECASE)
    if dtc:
        code = dtc.group().upper()
        exact = [item for item in KNOWLEDGE if item["title"].upper().startswith(code + " ")]
        return [dict(item) for item in exact[:limit]]
    terms = {word.lower().strip("?.!,:") for word in query.split() if len(word) > 2}
    ranked = []
    for item in KNOWLEDGE:
        text = f"{item['title']} {item['text']}".lower()
        score = sum(term in text for term in terms)
        if score:
            ranked.append((score, item))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [dict(item) for _, item in ranked[:limit]]
