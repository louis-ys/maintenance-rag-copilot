"""테스트에서 'src' 패키지를 임포트할 수 있도록 프로젝트 루트를 sys.path에 추가한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
