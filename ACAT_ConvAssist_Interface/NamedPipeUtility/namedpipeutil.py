import sys

if sys.platform == 'win32':
    try:
        import win32pipe
        import win32file
        import win32security
        import pywintypes
    except ImportError:
        pass

# Define the named pipe
pipe_name = r'\\.\pipe\ACATConvAssistPipe'

def create_named_pipe(pipe_name):
    """Create a named pipe with the given name."""
    try:
        pipe_handle = win32pipe.CreateNamedPipe(
            pipe_name,
            win32pipe.PIPE_ACCESS_OUTBOUND,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
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
        win32file.WriteFile(pipe_handle, message.encode())
        print(f"Message sent to pipe: {message}")
    except pywintypes.error as e:
        print(f"Failed to send message: {e}")

def cli_prompt(pipe_handle):
    """Provide a CLI prompt to send messages to the pipe."""
    print("Enter your messages below. Type 'exit' to quit.")
    while True:
        # Get input from the user
        message = input("Message> ")
        if message.lower() == 'exit':
            break
        send_message_to_pipe(pipe_handle, message)

if __name__ == "__main__":
    # Create the named pipe
    pipe_handle = create_named_pipe(pipe_name)
    if pipe_handle is not None:
        # Wait for a client to connect
        print("Waiting for client to connect to the pipe...")
        win32pipe.ConnectNamedPipe(pipe_handle, None)
        
        # Start the CLI prompt
        cli_prompt(pipe_handle)
        
        # Disconnect and close the named pipe
        win32pipe.DisconnectNamedPipe(pipe_handle)
        win32file.CloseHandle(pipe_handle)
        print(f"Named pipe '{pipe_name}' has been closed.")
