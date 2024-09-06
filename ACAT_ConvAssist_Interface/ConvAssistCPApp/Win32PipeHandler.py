import time
from typing import Any
import win32file

import win32pipe
import win32event
import pywintypes
import winerror
from pywintypes import HANDLE
import json


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

def send_message(Pipehandle, message: Any) -> None:
    """
    Send the message to the pipe

    :param Pipehandle: Handle of the pipe
    :param message: message to send
    :return: void
    """
    try:
        win32file.WriteFile(Pipehandle.handle, message.encode('utf-8')) # type: ignore
    except pywintypes.error as e:
        raise BrokenPipeError(f"Error sending message to named pipe: {e}") from e

def ConnectToNamedPipe(PipeServerName, retries, logger) -> tuple[bool, Any]:
    """
    Set the Pipe as Client

    :param PipeServerName: Name of the pipe
    :return: tuple of clientConnected and handle
    """

    pipeName = rf"\\.\pipe\{PipeServerName}"

    clientConnected = False
    handle = None
    
    while retries > 0:
        try:
            handle = win32file.CreateFile(
                pipeName,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                win32file.FILE_FLAG_OVERLAPPED,
                None
            )
            # handle = win32file.CreateFile(
            #     pipeName,
            #     win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            #     0,
            #     None,
            #     win32file.OPEN_EXISTING,
            #     0,
            #     None,
            # )
            if handle:
                win32pipe.SetNamedPipeHandleState(
                    handle.handle, win32pipe.PIPE_READMODE_MESSAGE, None, None
                )

                clientConnected = True
                break
        except pywintypes.error as e:
            if e.args[0] == winerror.ERROR_PIPE_BUSY:
                logger.debug(f"Pipe {pipeName} is busy, retrying...")
                retries -= 1
            elif e.args[0] == winerror.ERROR_FILE_NOT_FOUND:
                logger.debug(f"Pipe {pipeName} not found, retrying...")
                retries -= 1
            else:
                raise Exception(f"Error connecting to named pipe: {e}") from e

            time.sleep(5)

    return clientConnected, handle

def DisconnectNamedPipe(handle) -> None:
    """
    Disconnect the named pipe

    :param handle: Handle of the pipe
    :return: void
    """
    try:
        win32file.FlushFileBuffers(handle)
        win32file.CloseHandle(handle)
    except Exception as e:
        # raise Exception(f"Error disconnecting named pipe: {e}") from e
        pass