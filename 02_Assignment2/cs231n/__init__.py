
# 各作业保留自己的 cs231n 包，只共享项目级路径与图像工具。
import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
