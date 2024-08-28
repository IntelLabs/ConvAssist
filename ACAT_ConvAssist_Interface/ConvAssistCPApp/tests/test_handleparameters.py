import unittest
import pytest

from ACAT_ConvAssist_Interface.ConvAssistCPApp.ACATMessageTypes import ConvAssistMessage
from ACAT_ConvAssist_Interface.ConvAssistCPApp.ConvAssistCPApp import handle_parameter_change, ConvAssistVariables

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup
    resources = {
        "ca_vars": ConvAssistVariables(),
        "message": ConvAssistMessage(MessageType=1, PredictionType=1, Data="")
    }
    yield resources
    # Teardown

@pytest.mark.parametrize("parameter, param_name, value, expected", [
    (1, "path", "/path/to/model", "/path/to/model"),
    (2, "suggestions", 15, 15),
    (2, "suggestions", "15", 15),
    (3, "testgensentencepred", "True", True),
    (4, "retrieveaac", "True", True),
    (5, "pathstatic", "/path/to/static", "/path/to/static"),
    (6, "pathpersonalized", "/path/to/personalized", "/path/to/personalized")
])
def test_handle_parameters(setup_teardown, parameter, param_name, value, expected):
    # Arrange
    setup_teardown["message"].Data = f'{{"Parameter": "{parameter}", "Value": "{value}"}}'

    # Act
    changed = handle_parameter_change(setup_teardown["message"], setup_teardown["ca_vars"])

    # Assert
    assert getattr(setup_teardown["ca_vars"], param_name) == expected
    assert changed

if __name__ == "__main__":
    unittest.main()