"""离线验证码生成 — PIL 实现, 4 位字母数字"""
from __future__ import annotations

import io
import random
import string

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import FONT_PATH


_CHARS: str = string.ascii_uppercase + string.digits
_W, _H = 130, 44


def generate_captcha_text(length: int = 4) -> str:
    return "".join(random.choices(_CHARS, k=length))


def render_captcha_image(text: str) -> bytes:
    """渲染验证码图片为 PNG bytes, 直接喂给 QImage.fromData"""
    img = Image.new("RGB", (_W, _H), (32, 36, 44))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(str(FONT_PATH), 28)
    except Exception:
        font = ImageFont.load_default()

    # 干扰线
    for _ in range(5):
        x1, y1 = random.randint(0, _W), random.randint(0, _H)
        x2, y2 = random.randint(0, _W), random.randint(0, _H)
        draw.line(
            [(x1, y1), (x2, y2)],
            fill=(random.randint(60, 130),) * 3,
            width=1,
        )
    # 噪声点
    for _ in range(80):
        draw.point(
            (random.randint(0, _W - 1), random.randint(0, _H - 1)),
            fill=(random.randint(80, 180),) * 3,
        )

    # 字符 — 每个字符独立色与偏移
    char_w = _W // len(text)
    for i, ch in enumerate(text):
        offset_y = random.randint(-3, 5)
        color = (
            random.randint(180, 255),
            random.randint(180, 255),
            random.randint(180, 255),
        )
        draw.text(
            (i * char_w + 6, 4 + offset_y),
            ch,
            font=font,
            fill=color,
        )

    img = img.filter(ImageFilter.SMOOTH)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
