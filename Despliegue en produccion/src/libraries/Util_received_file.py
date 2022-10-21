ALLOWED = ['wav']


def evaluar_extension_archivo(filename):
    tiene_punto = "." in filename
    if tiene_punto:
        extension_archivo = filename.split(".", 1)[1].lower()
        if extension_archivo in ALLOWED:
            return True
    return False