def accuracy(logits, target):
    pred = logits.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).sum().item()
    return correct / logits.size(0)

def adjust_time(seconds: int) -> str:
    """
    Converts the given seconds into a more appropriate time unit

    Args:
        seconds (int): The number of seconds to be converted.

    Return:
        (str): A string representing the time duration in a more convenient unit.
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

def fix_data_type(input_string: str):
    """Detect data type in string and convert to appropiate type"""
    # First try to convert to int
    try:
        if input_string.isdigit() or (input_string.startswith('-') and input_string[1:].isdigit()):
            return int(input_string)
    except ValueError:
        pass

    # Try to convert to float
    try:
        return float(input_string)
    except ValueError:
        pass

    # Check for boolean
    if input_string.lower() in ('true', 'false'):
        return input_string.lower() == 'true'

    # If nothing else matches, return as string
    return input_string
