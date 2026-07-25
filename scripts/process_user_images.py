"""Process user-provided images from D:\\Downloads\\vajra-img and populate all branding directories."""

import os
from pathlib import Path
from PIL import Image

SRC_DIR = Path(r"D:\Downloads\vajra-img")
BRANDING_DIR = Path("branding")
WEB_DIR = BRANDING_DIR / "web"
SOCIAL_DIR = BRANDING_DIR / "social"

for directory in [BRANDING_DIR, WEB_DIR, SOCIAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# List of files in sorted order
files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(".png")])
print("Found user images:", files)

img_0 = Image.open(SRC_DIR / files[0])  # Dark Horizontal Lockup (1947, 808)
img_1 = Image.open(SRC_DIR / files[1])  # Light Horizontal Lockup (1891, 831)
img_2 = Image.open(SRC_DIR / files[2])  # Dark Square Symbol (1254, 1254)
img_3 = Image.open(SRC_DIR / files[3])  # Light Lockup / Vertical (1842, 854)
img_4 = Image.open(SRC_DIR / files[4])  # Light Square Symbol (1254, 1254)
img_5 = Image.open(SRC_DIR / files[5])  # HuggingFace Avatar Icon (1254, 1254)
img_6 = Image.open(SRC_DIR / files[6])  # Wordmark (2050, 767)

# Helper function to save PNG with optional resize
def save_png(img: Image.Image, dst_path: Path, size: tuple[int, int] | None = None) -> None:
    out = img.copy()
    if size:
        out = out.resize(size, Image.Resampling.LANCZOS)
    out.save(dst_path, "PNG", quality=95)
    print(f"Saved PNG: {dst_path} ({dst_path.stat().st_size} bytes)")

# Helper function to generate a wrapper SVG referencing embedded PNG image data or SVG image tag
def save_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    import base64
    with open(png_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    with Image.open(png_path) as img:
        w, h = img.size
    
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <image width="{w}" height="{h}" href="data:image/png;base64,{encoded}" />
</svg>'''
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Saved SVG wrapper: {svg_path}")

# -----------------------------------------------------------------------------
# 1. Primary Branding Assets
# -----------------------------------------------------------------------------

save_png(img_0, BRANDING_DIR / "logo.png")
save_png(img_0, BRANDING_DIR / "logo-dark.png")
save_png(img_1, BRANDING_DIR / "logo-light.png")
save_png(img_0, BRANDING_DIR / "horizontal-lockup.png")
save_png(img_3, BRANDING_DIR / "vertical-lockup.png")
save_png(img_2, BRANDING_DIR / "symbol.png")
save_png(img_6, BRANDING_DIR / "wordmark.png")

# Monochrome version
mono_img = img_6.convert("L").convert("RGB")
save_png(mono_img, BRANDING_DIR / "logo-monochrome.png")

# Avatars & Favicons
save_png(img_2, BRANDING_DIR / "github-avatar.png", (512, 512))
save_png(img_5, BRANDING_DIR / "huggingface-avatar.png", (512, 512))
save_png(img_2, BRANDING_DIR / "app-icon-512.png", (512, 512))
save_png(img_2, BRANDING_DIR / "apple-touch-icon.png", (180, 180))
save_png(img_2, BRANDING_DIR / "favicon-16.png", (16, 16))
save_png(img_2, BRANDING_DIR / "favicon-32.png", (32, 32))
save_png(img_2, BRANDING_DIR / "favicon-64.png", (64, 64))

# Social Banner & Preview Card
save_png(img_0, BRANDING_DIR / "social-banner.png", (1200, 630))
save_png(img_0, BRANDING_DIR / "social-preview-1280x640.png", (1280, 640))

# Save SVG wrappers for all vector asset paths using user images
save_svg_wrapper(BRANDING_DIR / "logo.png", BRANDING_DIR / "logo.svg")
save_svg_wrapper(BRANDING_DIR / "logo-dark.png", BRANDING_DIR / "logo-dark.svg")
save_svg_wrapper(BRANDING_DIR / "logo-light.png", BRANDING_DIR / "logo-light.svg")
save_svg_wrapper(BRANDING_DIR / "symbol.png", BRANDING_DIR / "symbol.svg")
save_svg_wrapper(BRANDING_DIR / "wordmark.png", BRANDING_DIR / "wordmark.svg")
save_svg_wrapper(BRANDING_DIR / "horizontal-lockup.png", BRANDING_DIR / "horizontal-lockup.svg")
save_svg_wrapper(BRANDING_DIR / "vertical-lockup.png", BRANDING_DIR / "vertical-lockup.svg")
save_svg_wrapper(BRANDING_DIR / "logo-monochrome.png", BRANDING_DIR / "logo-monochrome.svg")

# -----------------------------------------------------------------------------
# 2. Web Directory Assets
# -----------------------------------------------------------------------------

save_png(img_0, WEB_DIR / "header-dark.png", (1200, 500))
save_png(img_1, WEB_DIR / "header-light.png", (1200, 500))
save_png(img_0, WEB_DIR / "navbar-logo.png", (400, 160))
save_png(img_2, WEB_DIR / "hero-symbol.png", (800, 800))
save_png(img_2, WEB_DIR / "favicon.png", (64, 64))
save_png(img_3, WEB_DIR / "mobile-logo.png", (400, 400))

# -----------------------------------------------------------------------------
# 3. Social Directory Assets
# -----------------------------------------------------------------------------

save_png(img_0, SOCIAL_DIR / "linkedin-banner.png", (1584, 396))
save_png(img_0, SOCIAL_DIR / "twitter-header.png", (1500, 500))
save_png(img_3, SOCIAL_DIR / "square-announcement.png", (1080, 1080))
save_png(img_2, SOCIAL_DIR / "discord-icon.png", (512, 512))
save_png(img_2, SOCIAL_DIR / "slack-icon.png", (512, 512))
save_png(img_0, SOCIAL_DIR / "email-signature.png", (400, 160))

print("\nSuccessfully replaced all brand assets with exact user-provided images from D:\\Downloads\\vajra-img!")
