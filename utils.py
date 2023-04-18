import os, numpy as np

def check_paths_exists(paths):
    if not is_array(paths): paths = [paths]
  
    for _,path in enumerate(paths):
        if not os.path.exists(path):
            return False, path
    
    return True, None

def string_equals(str1, str2):
    return (
        True
        if str1.isalnum() and str2.isalnum() and str1.casefold() == str2.casefold()
        else False
    )

def is_array(object):
    return True if isinstance(object, list) else False
