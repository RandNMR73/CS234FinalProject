# debug enable parameters
DEBUG_ENABLED = False
DEBUG_CODES = []

# debug print functions
def debug(text):
    if (DEBUG_ENABLED):
        print(text)

def debugc(text, code):
    if code in DEBUG_CODES:
        print(text)