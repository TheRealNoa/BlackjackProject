import base64
from pathlib import Path
img_path = Path("./screenshots/Ace-of-Spades-Poster_1.jpg")
b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
print("Length:", len(b64))
print(b64) 