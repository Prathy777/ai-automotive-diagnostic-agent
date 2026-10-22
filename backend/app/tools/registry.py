"""Allowlisted simulator diagnostic tools. Model text is never executed as code."""
from dataclasses import dataclass
from typing import Any, Callable

from ..simulator import Vehicle


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[[Vehicle, dict[str, Any]], dict[str, Any]]
    input_schema: dict[str, Any]
    kind: str


def _read(field: str):
    def run(vehicle: Vehicle, _args: dict[str, Any]) -> dict[str, Any]:
        return {field: vehicle.telemetry()[field]}
    return run


def _test(name: str):
    def run(vehicle: Vehicle, _args: dict[str, Any]) -> dict[str, Any]:
        return vehicle.run_test(name)
    return run


def _recent(vehicle: Vehicle, _args: dict[str, Any]) -> dict[str, Any]:
    return {"samples": [dict(sample) for sample in vehicle.history[-3:]]}


_read_tools = {
    "read_engine_rpm": ("Read engine revolutions per minute.", "rpm"),
    "read_engine_temperature": ("Read engine temperature in Celsius.", "engine_temperature"),
    "read_coolant_temperature": ("Read coolant temperature in Celsius.", "coolant_temperature"),
    "read_battery_voltage": ("Read battery voltage.", "battery_voltage"),
    "read_alternator_output": ("Read alternator output voltage.", "alternator_output"),
    "read_engine_vibration": ("Measure engine vibration.", "engine_vibration"),
    "read_wheel_vibration": ("Measure wheel vibration.", "wheel_vibration"),
    "read_fuel_consumption": ("Read fuel consumption rate.", "fuel_consumption"),
    "read_vehicle_speed": ("Read vehicle speed.", "vehicle_speed"),
    "read_air_fuel_ratio": ("Read oxygen/fuel-mixture ratio.", "air_fuel_ratio"),
    "read_intake_pressure": ("Read intake manifold pressure.", "intake_pressure"),
    "read_throttle_position": ("Read throttle position.", "throttle_position"),
    "read_electrical_load": ("Read current electrical system load.", "electrical_load"),
}
_tests = {
    "run_2000_rpm_test": "Check RPM tracking and engine vibration while the engine is raised to 2000 RPM.",
    "run_acceleration_test": "Check acceleration response and speed-related wheel vibration.",
    "run_electrical_load_test": "Check charging voltage while electrical load is applied.",
    "run_cooling_system_test": "Check coolant temperature and cooling fan response.",
    "run_engine_load_test": "Check engine response, RPM, and mixture under load.",
    "run_mixture_test": "Check air-fuel ratio and intake pressure for mixture faults.",
}
_EMPTY_OBJECT = {"type": "object", "properties": {}, "additionalProperties": False}

TOOLS: dict[str, Tool] = {
    name: Tool(name, description, _read(field), _EMPTY_OBJECT, "telemetry")
    for name, (description, field) in _read_tools.items()
}
TOOLS["inspect_recent_telemetry"] = Tool(
    "inspect_recent_telemetry", "Inspect the vehicle's last three telemetry samples.", _recent,
    _EMPTY_OBJECT, "telemetry",
)
TOOLS.update({
    name: Tool(name, description, _test(name), _EMPTY_OBJECT, "diagnostic_test")
    for name, description in _tests.items()
})


def tool_descriptions() -> list[dict[str, Any]]:
    return [{"name": tool.name, "description": tool.description,
             "arguments": tool.input_schema, "kind": tool.kind}
            for tool in TOOLS.values()]
