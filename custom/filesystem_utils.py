import os




def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def split_filename_extension(file_path):
    name, ext = os.path.splitext(file_path)
    return os.path.basename(name), ext


def is_folder(path):
    return os.path.isdir(path)


def get_batch_iterator(array, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]

