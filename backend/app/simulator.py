"""Deterministic digital twin scenarios for the diagnostic demo."""
from dataclasses import dataclass, field
from typing import Literal

Fault = Literal["normal", "misfire", "overheating", "alternator_failure", "wheel_imbalance", "fuel_air_mixture"]

@dataclass
class Vehicle:
    fault: Fault = "normal"
    tick: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def telemetry(self) -> dict[str, float]:
        self.tick += 1
        t = self.tick
        base = {"rpm": 820 + (t % 3 - 1) * 7, "engine_temperature": 91.0,
                "coolant_temperature": 88.0, "oil_pressure": 39.0,
                "fuel_consumption": 2.4, "battery_voltage": 12.6,
                "alternator_output": 14.1, "engine_vibration": 0.18,
                "vehicle_speed": 0.0, "throttle_position": 8.0,
                "engine_load": 18.0, "air_fuel_ratio": 14.7, "intake_pressure": 32.0,
                "electrical_load": 28.0, "acceleration": 0.0,
                "brake_temperature": 35.0, "wheel_vibration": 0.12,
                "cooling_fan_rpm": 1450.0}
        if self.fault == "misfire":
            base.update(rpm=785 + (t % 4) * 32, engine_vibration=.83,
                        fuel_consumption=3.15, air_fuel_ratio=16.2)
        elif self.fault == "overheating":
            base.update(engine_temperature=112 + min(t, 10) * .7,
                        coolant_temperature=108 + min(t, 10) * .8,
                        cooling_fan_rpm=400.0, engine_load=24.0)
        elif self.fault == "alternator_failure":
            base.update(battery_voltage=11.25, alternator_output=11.4)
        elif self.fault == "wheel_imbalance":
            base.update(vehicle_speed=72.0, acceleration=.22, wheel_vibration=.91)
        elif self.fault == "fuel_air_mixture":
            base.update(fuel_consumption=3.8, air_fuel_ratio=18.1, intake_pressure=45.0,
                        engine_vibration=.42, throttle_position=13.0)
        self.history.append(dict(base))
        return base

    def run_test(self, name: str) -> dict:
        data = self.telemetry()
        if name == "run_2000_rpm_test":
            return {"test": name, "target_rpm": 2000, "observed_rpm": 1940 if self.fault == "misfire" else 2000,
                    "engine_vibration": .96 if self.fault == "misfire" else data["engine_vibration"],
                    "result": "abnormal engine-correlated vibration" if self.fault == "misfire" else "within expected range"}
        if name == "run_electrical_load_test":
            return {"test": name, "loaded_voltage": 10.8 if self.fault == "alternator_failure" else 13.7,
                    "result": "charging system voltage drops under load" if self.fault == "alternator_failure" else "charging system stable"}
        if name == "run_cooling_system_test":
            return {"test": name, "coolant_temperature": data["coolant_temperature"], "fan_rpm": data["cooling_fan_rpm"],
                    "result": "cooling system cannot control temperature" if self.fault == "overheating" else "cooling response normal"}
        if name == "run_acceleration_test":
            return {"test": name, "speed_kph": 72.0, "wheel_vibration": .91 if self.fault == "wheel_imbalance" else .14,
                    "result": "speed-correlated wheel vibration" if self.fault == "wheel_imbalance" else "no abnormal wheel vibration"}
        if name == "run_engine_load_test":
            return {"test": name, "engine_load": data["engine_load"], "rpm": data["rpm"],
                    "air_fuel_ratio": data["air_fuel_ratio"],
                    "result": "mixture deviates under load" if self.fault == "fuel_air_mixture" else "engine response within expected range"}
        if name == "run_mixture_test":
            return {"test": name, "air_fuel_ratio": data["air_fuel_ratio"],
                    "intake_pressure": data["intake_pressure"], "fuel_consumption": data["fuel_consumption"],
                    "result": "lean mixture indicated" if self.fault == "fuel_air_mixture" else "mixture within expected range"}
        return {"test": name, "result": "test unavailable"}
