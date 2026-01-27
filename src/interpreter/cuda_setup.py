import os
import sys
import site
from pathlib import Path

from . import log

logger = log.get_logger()

def setup_cuda_paths() -> None:
    """Add NVIDIA CUDA runtime paths to PATH/DLL search path for onnxruntime-gpu.
    
    This is required on Windows because onnxruntime-gpu doesn't automatically 
    find the DLLs provided by the nvidia-* pypi packages.
    """
    if sys.platform != "win32":
        return

    try:
        # Attempt to find site-packages
        # site.getsitepackages() might return multiple paths
        candidate_paths = site.getsitepackages()
        user_site = site.getusersitepackages()
        if user_site:
            candidate_paths.append(user_site)
            
        nvidia_paths = []
        
        for sp in candidate_paths:
            sp_path = Path(sp)
            # Look for nvidia packages
            nv_path = sp_path / "nvidia"
            if not nv_path.exists():
                continue
                
            # These are the components needed for onnxruntime-gpu (CUDA 12)
            # cublas contains cublas64_12.dll
            # cudnn contains cudnn_cnn_infer64_8.dll etc
            # cuda_runtime usually implicitly handled or in bin
            for component in ["cudnn", "cublas", "cuda_runtime"]:
                bin_path = nv_path / component / "bin"
                if bin_path.exists():
                    str_path = str(bin_path)
                    if str_path not in nvidia_paths:
                        nvidia_paths.append(str_path)
                        logger.debug("found cuda dll path", path=str_path)

        if nvidia_paths:
            # 1. Add to PATH environment variable (traditional way)
            current_path = os.environ.get("PATH", "")
            # Prepend to ensure they are found first
            os.environ["PATH"] = os.pathsep.join(nvidia_paths + [current_path])
            
            # 2. Use add_dll_directory (Python 3.8+ secure DLL loading on Windows)
            # This is often strictly required for imports to work
            if hasattr(os, "add_dll_directory"):
                for p in nvidia_paths:
                    try:
                        os.add_dll_directory(p)
                    except Exception as e:
                        logger.warning("failed to add dll directory", path=p, error=str(e))
                        
            logger.info("configured cuda paths", count=len(nvidia_paths))
            
    except Exception as e:
        logger.warning("failed to setup cuda paths", error=str(e))
