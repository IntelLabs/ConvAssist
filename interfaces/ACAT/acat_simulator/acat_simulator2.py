import websockets.sync.client

from interfaces.ACAT.utilities.ACATMessageTypes import (
    ConvAssistMessage,
    ConvAssistMessageTypes,
    ConvAssistPredictionTypes,
    ConvAssistSetParam,
    ParameterType,
    WordAndCharacterPredictionResponse,
)

def gen_prediction_msgs(message: str) -> list[str]:
    msgs = []
    for prediction_type in list(ConvAssistPredictionTypes)[1:5]:
        msgs.append(
            ConvAssistMessage(
                ConvAssistMessageTypes.NEXTSENTENCEPREDICTION, prediction_type, message
            ).jsonSerialize()
        )
        msgs.append(
            ConvAssistMessage(
                ConvAssistMessageTypes.NEXTWORDPREDICTION, prediction_type, message
            ).jsonSerialize()
        )
    return msgs

# set_path_param = ConvAssistSetParam(
#     ParameterType.PATH,
#     r"C:\\Users\\mbeale\\OneDrive - Intel Corporation\\Documents\\ACAT\\Users\\DefaultUser\\en\\WordPredictors\\ConvAssist\\Settings"
# )

set_static_path_param = ConvAssistSetParam(
    ParameterType.PATHSTATIC,
    r"C:\\Program Files (x86)\\ACAT\\en\\WordPredictors\\ConvAssist",
)
set_personalized_path_param = ConvAssistSetParam(
    ParameterType.PATHPERSONALIZED,
    r"C:\\Users\\mbeale\\OneDrive - Intel Corporation\\Documents\\ACAT\\Users\\DefaultUser\\en\WordPredictors\\ConvAssist\\Database",
)
set_log_param = ConvAssistSetParam(
    ParameterType.PATHLOG,
    r"C:\\Users\\mbeale\\OneDrive - Intel Corporation\\Documents\\ACAT\\Logs",
)
set_enable_logs_param = ConvAssistSetParam(ParameterType.ENABLELOGS, True)
set_suggestions_param = ConvAssistSetParam(ParameterType.SUGGESTIONS, 10)
set_test_general_sentence_prediction_param = ConvAssistSetParam(
    ParameterType.TESTGENSENTENCEPRED, False
)
set_retrieve_aac_param = ConvAssistSetParam(ParameterType.RETRIEVEAAC, False)

params = [
    set_log_param,
    set_enable_logs_param,
    set_suggestions_param,
    set_test_general_sentence_prediction_param,
    set_retrieve_aac_param,
    # set_path_param,
    set_static_path_param,
    set_personalized_path_param,
]

def start_client():
    """Connects to the WebSocket server and sends messages."""
    try:
        uri = "ws://localhost:8765"
        with websockets.sync.client.connect(uri) as ws:

            # Send parameters
            for param in params:
                msg = ConvAssistMessage(
                    ConvAssistMessageTypes.SETPARAM,
                    ConvAssistPredictionTypes.NONE,
                    param.jsonSerialize(),
                ).jsonSerialize()
                ws.send(msg)
                response = ws.recv()
                print(f"Server: {WordAndCharacterPredictionResponse.jsonDeserialize(str(response))}")

            while True:
                message = input("You: ")
                if message == ":exit":
                    msg = ConvAssistMessage(
                        ConvAssistMessageTypes.FORCEQUITAPP, ConvAssistPredictionTypes.NONE, ""
                    ).jsonSerialize()
                    ws.send(msg)
                    ws.close()
                    break
                msgs = gen_prediction_msgs(message)
                for msg in msgs:
                    ws.send(msg)
                    response = ws.recv()
                    print(f"Server: {WordAndCharacterPredictionResponse.jsonDeserialize(str(response))}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print("Disconnected from server.")

# Start client
if __name__ == "__main__":
    start_client()
