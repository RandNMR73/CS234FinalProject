# random util functions

def clip(num, low, high):
    # bounds for clipping are inclusive
    if (num < low):
        num = low
    elif (num > high):
        num = high
    
    return num