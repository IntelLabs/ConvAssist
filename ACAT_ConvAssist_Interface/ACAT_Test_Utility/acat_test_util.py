import sys
import os

if sys.platform == 'win32':
    try:
        import win32pipe
        import win32file
        import win32security
        import pywintypes
    except ImportError:
        pass

current_path = os.path.dirname(os.path.realpath(__file__))
if current_path not in sys.path:
    sys.path.append(current_path)

from ACAT_ConvAssist_Interface.ConvAssistCPApp.ACATMessageTypes import ConvAssistSetParam, ParameterType, ConvAssistMessage, ConvAssistMessageTypes, ConvAssistPredictionTypes
from ACAT_ConvAssist_Interface.ConvAssistCPApp.Win32PipeHandler import get_incoming_message

set_path_param = ConvAssistSetParam(ParameterType.PATH, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources")
set_static_path_param = ConvAssistSetParam(ParameterType.PATHSTATIC, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/static_resources") 
set_personalized_path_param = ConvAssistSetParam(ParameterType.PATHPERSONALIZED, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/personalized")
set_log_param = ConvAssistSetParam(ParameterType.PATHLOG, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/logs")
set_enable_logs_param = ConvAssistSetParam(ParameterType.ENABLELOGS, True)
set_suggestions_param = ConvAssistSetParam(ParameterType.SUGGESTIONS, 4)
set_test_general_sentence_prediction_param = ConvAssistSetParam(ParameterType.TESTGENSENTENCEPRED, True)
set_retrieve_aac_param = ConvAssistSetParam(ParameterType.RETRIEVEAAC, True)

params = [set_log_param, set_enable_logs_param, set_suggestions_param, \
        set_test_general_sentence_prediction_param, set_retrieve_aac_param, \
        set_path_param, set_static_path_param, set_personalized_path_param, \
        ]


# Define the named pipe
pipe_name = r'\\.\pipe\ACATConvAssistPipe'

def create_named_pipe(pipe_name):
    """Create a named pipe with the given name."""
    try:
        pipe_handle = win32pipe.CreateNamedPipe(
            pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_MESSAGE | 
                win32pipe.PIPE_WAIT |
                win32pipe.PIPE_READMODE_MESSAGE,
            1, 65536, 65536,
            300,
            win32security.SECURITY_ATTRIBUTES()
        )
        return pipe_handle
    except pywintypes.error as e:
        print(f"Failed to create named pipe: {e}")
        return None

def send_message_to_pipe(pipe_handle, message):
    """Send a message to the named pipe."""
    try:
        win32file.WriteFile(pipe_handle, message.encode('utf-8'))
        print(f"Message sent to pipe: {message}")
    except pywintypes.error as e:
        print(f"Failed to send message: {e}")

def wait_for_message_from_pipe(pipe_handle):
    """Wait for a message from the named pipe."""
    try:
        message = get_incoming_message(pipe_handle)
        print(f"Received message from pipe: {message}")
        return True
    except pywintypes.error as e:
        print(f"Failed to read message: {e}")
        return False

def cli_prompt(pipe_handle):
    """Provide a CLI prompt to send messages to the pipe."""
    print("Enter your messages below. Type 'exit' to quit.")
    breakloop = False
    while not breakloop:
        msgs = []
        # Get input from the user
        message = input("Message> ")
        

        if message.lower() == 'exit':
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.FORCEQUITAPP, ConvAssistPredictionTypes.NONE, "").jsonSerialize())
            breakloop = True
        else:
            message += ' '
            # msgs.append(ConvAssistMessage(ConvAssistMessageTypes.LEARNSHORTHAND, ConvAssistPredictionTypes.NONE, message).jsonSerialize())
          
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.NEXTWORDPREDICTION, ConvAssistPredictionTypes.NORMAL, message).jsonSerialize())
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.NEXTWORDPREDICTION, ConvAssistPredictionTypes.CANNEDPHRASESMODE, message).jsonSerialize())
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.NEXTSENTENCEPREDICTION, ConvAssistPredictionTypes.SENTENCES, message).jsonSerialize())
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.NEXTWORDPREDICTION, ConvAssistPredictionTypes.SHORTHANDMODE, message).jsonSerialize())
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.LEARNWORDS, ConvAssistPredictionTypes.NONE, message).jsonSerialize())
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.LEARNCANNED, ConvAssistPredictionTypes.NONE, message).jsonSerialize())
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.LEARNSENTENCES, ConvAssistPredictionTypes.NONE, message).jsonSerialize())

        for msg in msgs:
            send_message_to_pipe(pipe_handle, msg)
            if not wait_for_message_from_pipe(pipe_handle):
                break

if __name__ == "__main__":
    # Create the named pipe
    pipe_handle = create_named_pipe(pipe_name)
    if pipe_handle is not None:
        # Wait for a client to connect
        print("Waiting for client to connect to the pipe...")
        win32pipe.ConnectNamedPipe(pipe_handle, None)
        
        # # Start the CLI prompt
        # cli_prompt(pipe_handle)
        for param in params:
            msg = ConvAssistMessage(ConvAssistMessageTypes.SETPARAM, ConvAssistPredictionTypes.NONE, param.jsonSerialize()).jsonSerialize()
            send_message_to_pipe(pipe_handle, msg)
            wait_for_message_from_pipe(pipe_handle)

        cli_prompt(pipe_handle)

        # Disconnect and close the named pipe
        win32pipe.DisconnectNamedPipe(pipe_handle)
        win32file.CloseHandle(pipe_handle)
        print(f"Named pipe '{pipe_name}' has been closed.")
