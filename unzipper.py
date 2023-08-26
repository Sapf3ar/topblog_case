import zipfile
file_path = "./data.zip"
out_dir = "out_data/"
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(out_dir)
