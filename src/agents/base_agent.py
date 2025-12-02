import logging
import sys

# Configure logging to print to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class Agent:
    # Colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    def __init__(self, name="Base Agent", color=WHITE):
        self.name = name
        self.color = color

    def log(self, message):
        print(f"{self.color}[{self.name}] {message}{self.RESET}")