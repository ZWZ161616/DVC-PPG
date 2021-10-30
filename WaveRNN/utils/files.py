from pathlib import Path
from typing import Union
from natsort import natsorted

def get_files(path: Union[str, Path], extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    pathlib_path = list(path.rglob(f'*{extension}'))
    pathlib_path_name = []
    
    for i, path_ in enumerate(pathlib_path):
        pathlib_path_name.append((path_.stem,i))
        
    pathlib_path_name_natsort = natsorted(pathlib_path_name)
    path2 = []
    
    for _,i in pathlib_path_name_natsort:
        path2.append(pathlib_path[i])
        
    return path2
