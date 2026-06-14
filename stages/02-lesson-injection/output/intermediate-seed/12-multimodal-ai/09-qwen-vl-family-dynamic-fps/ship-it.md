## Ship It

In production, video-VLM inference needs a token-budget manager that caps total visual tokens to fit context windows, a frame-selection policy that adapts to content, and an observability layer that reports what the model actually saw. The code below implements a production-grade wrapper that does all three, structured for deployment behind an API.

```python
import numpy as np
from PIL import Image
from dataclasses import dataclass, field, asdict
from typing import List, Optional