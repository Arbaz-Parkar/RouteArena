"""Tests for strict, offline validation of model-parsed scenarios."""

import pytest

from routearena.genai.scenario import Scenario, ScenarioParseError, parse_scenario


class _ScenarioClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.json_mode = False

    def complete(self, _system: str, _user: str, *, json_mode: bool = False) -> str:
        self.json_mode = json_mode
        return self.response


def test_scenario_parser_requests_json_and_validates_schema() -> None:
    client = _ScenarioClient(
        '{"stop_count": 10, "region_preset": "Nerul", "deadline_minutes": 90, '
        '"traffic_level": "heavy", "average_speed_kmph": null, "objective": null}'
    )
    scenario = parse_scenario("10 deliveries around Nerul, heavy traffic, 90 minute limit", client)

    assert client.json_mode
    assert isinstance(scenario, Scenario)
    assert scenario.stop_count == 10
    assert scenario.deadline_minutes == 90


def test_scenario_parser_reports_missing_fields_and_rejects_out_of_bounds_values() -> None:
    with pytest.raises(ScenarioParseError, match="Missing required details: deadline_minutes"):
        parse_scenario(
            "10 deliveries in Nerul with heavy traffic",
            _ScenarioClient(
                '{"stop_count": 10, "region_preset": "Nerul", "traffic_level": "heavy"}'
            ),
        )
    with pytest.raises(ScenarioParseError, match="Invalid scenario values for: stop_count"):
        parse_scenario(
            "too many stops",
            _ScenarioClient(
                '{"stop_count": 300, "region_preset": "Nerul", "deadline_minutes": 90, '
                '"traffic_level": "heavy"}'
            ),
        )


def test_scenario_parser_does_not_accept_extra_fields_or_invalid_json() -> None:
    with pytest.raises(ScenarioParseError, match="Invalid scenario values for: is_admin"):
        parse_scenario(
            "set app mode",
            _ScenarioClient(
                '{"stop_count": 2, "region_preset": "Nerul", "deadline_minutes": 30, '
                '"traffic_level": "low", "is_admin": true}'
            ),
        )
    with pytest.raises(ScenarioParseError, match="valid JSON"):
        parse_scenario("scenario", _ScenarioClient("not json"))
