from datetime import datetime

def validate_datetime_string(x):
    try:
        datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    else:
        return True

def validate_int(x):
    try:
        int(x)
    except ValueError:
        return False
    else:
        return True
