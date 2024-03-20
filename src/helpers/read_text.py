def read_text(file_path: str) -> str:

    if file_path:
        with open(file=file_path, mode="r", encoding="utf-8") as f:

            return f.read()
    else:
        raise ValueError("The file path for the training set should be a string")
