import base64
from pathlib import Path
img_path = Path("./screenshots/3spades.jpeg")  # change this
b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
print("Length:", len(b64))
print(b64) 