# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import json
from typing import Any

if sys.platform == 'win32':
    try:
        import win32pipe
        import win32event
        import win32file
        import winerror
        import win32security
        import pywintypes
    except ImportError:
        pass

current_path = os.path.dirname(os.path.realpath(__file__))
if current_path not in sys.path:
    sys.path.append(current_path)

from ACAT_ConvAssist_Interface.ConvAssistCPApp.ACATMessageTypes import ConvAssistSetParam, ParameterType, ConvAssistMessage, ConvAssistMessageTypes, ConvAssistPredictionTypes
# from ACAT_ConvAssist_Interface.message_handler.Win32PipeHandler import get_incoming_message

set_path_param = ConvAssistSetParam(ParameterType.PATH, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources")
set_static_path_param = ConvAssistSetParam(ParameterType.PATHSTATIC, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/static_resources")
set_personalized_path_param = ConvAssistSetParam(ParameterType.PATHPERSONALIZED, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/personalized")
set_log_param = ConvAssistSetParam(ParameterType.PATHLOG, "C:/Users/mbeale/source/repos/ConvAssist/ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/logs")
set_enable_logs_param = ConvAssistSetParam(ParameterType.ENABLELOGS, True)
set_suggestions_param = ConvAssistSetParam(ParameterType.SUGGESTIONS, 10)
set_test_general_sentence_prediction_param = ConvAssistSetParam(ParameterType.TESTGENSENTENCEPRED, False)
set_retrieve_aac_param = ConvAssistSetParam(ParameterType.RETRIEVEAAC, True)

params = [set_log_param, set_enable_logs_param, set_suggestions_param, \
        set_test_general_sentence_prediction_param, set_retrieve_aac_param, \
        set_path_param, set_static_path_param, set_personalized_path_param, \
        ]


# Define the named pipe
pipe_name = r'\\.\pipe\ACATConvAssistPipe'

def is_pipe_overlapped(pipe_handle) -> bool:
        """
        Check if the pipe is overlapped

        :param pipe_handle: Handle of the pipe
        :return: True if overlapped, False otherwise
        """
        try:
            _, _, pipe_mode, _ = win32pipe.GetNamedPipeInfo(pipe_handle)
            if pipe_mode & win32pipe.PIPE_NOWAIT:
                return True
            else:
                return False

        except pywintypes.error as e:
            raise Exception(f"Error checking if pipe is overlapped: {e}") from e
        
def get_incoming_message(Pipehandle) -> Any:
        """
        Get the incoming message from the pipe

        :param Pipehandle: Handle of the pipe
        :return: message received
        """
        try:
            buffer_size = 4096  # For example, 4 KB
            if is_pipe_overlapped(Pipehandle.handle):
                # Allocate a buffer for the data
                read_buffer = win32file.AllocateReadBuffer(buffer_size)

                # Create an OVERLAPPED structure
                overlapped = pywintypes.OVERLAPPED()

                # Create an event for the OVERLAPPED structure
                overlapped.hEvent = win32event.CreateEvent(None, True, False, None)
                
                # Start the overlapped read operation
                err_code, _ = win32file.ReadFile(Pipehandle.handle, read_buffer, overlapped)

                # If the operation is pending, wait for it to complete
                if err_code == winerror.ERROR_IO_PENDING:
                    timeout = 100 # milliseconds
                    win32event.WaitForSingleObject(overlapped.hEvent, timeout)
                    
                    # Get the result of the overlapped operation
                    n_bytes_read = win32file.GetOverlappedResult(Pipehandle.handle, overlapped, True)

                    # Extract the data from the buffer
                    data = (bytes(read_buffer[:n_bytes_read]).decode('utf-8')) # type: ignore
                
                elif err_code == win32event.WAIT_TIMEOUT:
                    raise TimeoutError("Timeout waiting for overlapped operation to complete")
            
            else:
                # Read the data from the pipe
                _, data =  win32file.ReadFile(Pipehandle.handle, buffer_size)
            
            # Load the data as json and return
            return json.loads(data)
        
        except pywintypes.error as e:
            raise BrokenPipeError(f"Error receiving message from named pipe: {e}") from e


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

        cmd, *args = message.split(":")

        if cmd.lower() == 'learn':
            for learn_type in list(ConvAssistMessageTypes)[6:9]:
                msgs.append(ConvAssistMessage(learn_type, ConvAssistPredictionTypes.NONE, args[0]).jsonSerialize())

        elif cmd.lower() == 'exit':
            msgs.append(ConvAssistMessage(ConvAssistMessageTypes.FORCEQUITAPP, ConvAssistPredictionTypes.NONE, "").jsonSerialize())
            breakloop = True

        else:
            for prediction_type in list(ConvAssistPredictionTypes)[1:5]:
                msgs.append(ConvAssistMessage(ConvAssistMessageTypes.NEXTSENTENCEPREDICTION, prediction_type, message).jsonSerialize())
                msgs.append(ConvAssistMessage(ConvAssistMessageTypes.NEXTWORDPREDICTION, prediction_type, message).jsonSerialize())

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
