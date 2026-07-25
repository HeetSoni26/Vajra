"""Branding Asset Generator — Recreates official Vajra vector SVG artwork & PNG exports."""

import os
from pathlib import Path
from resvg_py import svg_to_bytes
from PIL import Image

BRANDING_DIR = Path("branding")
WEB_DIR = BRANDING_DIR / "web"
SOCIAL_DIR = BRANDING_DIR / "social"

for directory in [BRANDING_DIR, WEB_DIR, SOCIAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# SVG Vector Definitions
# -----------------------------------------------------------------------------

SYMBOL_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="500" height="500">
  <defs>
    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00F0FF" />
      <stop offset="50%" stop-color="#00C8FF" />
      <stop offset="100%" stop-color="#0080FF" />
    </linearGradient>
    <linearGradient id="top-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#5B38D5" />
      <stop offset="100%" stop-color="#321D87" />
    </linearGradient>
    <linearGradient id="mid-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6F44F0" />
      <stop offset="100%" stop-color="#4625AF" />
    </linearGradient>
    <linearGradient id="bot-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3A1F99" />
      <stop offset="100%" stop-color="#1C0E56" />
    </linearGradient>
    <linearGradient id="top-right-facet" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#482AB6" />
      <stop offset="100%" stop-color="#241366" />
    </linearGradient>
    <linearGradient id="mid-right-facet" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#5D35D9" />
      <stop offset="100%" stop-color="#331A8B" />
    </linearGradient>
    <linearGradient id="bot-right-facet" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2A1473" />
      <stop offset="100%" stop-color="#13083B" />
    </linearGradient>
  </defs>

  <g id="Symbol">
    <!-- Left Outer Wing -->
    <path d="M 250 40 L 205 52 L 135 170 L 205 148 Z" fill="url(#top-left-facet)" />
    <path d="M 135 170 L 85 245 L 175 315 L 205 148 Z" fill="url(#mid-left-facet)" />
    <path d="M 85 245 L 170 400 L 235 372 L 175 315 Z" fill="url(#bot-left-facet)" />
    <path d="M 175 315 L 235 372 L 238 260 L 205 148 Z" fill="#20115B" />

    <!-- Right Outer Wing (Mirrored) -->
    <path d="M 250 40 L 295 52 L 365 170 L 295 148 Z" fill="url(#top-right-facet)" />
    <path d="M 365 170 L 415 245 L 325 315 L 295 148 Z" fill="url(#mid-right-facet)" />
    <path d="M 415 245 L 330 400 L 265 372 L 325 315 Z" fill="url(#bot-right-facet)" />
    <path d="M 325 315 L 265 372 L 262 260 L 295 148 Z" fill="#150A3E" />

    <!-- Center Cyan Thunderbolt Core -->
    <path d="M 250 40 L 268 148 L 250 250 L 232 148 Z" fill="url(#cyan-glow)" />
    <path d="M 244 250 L 244 380 L 250 410 L 256 380 L 256 250 Z" fill="url(#cyan-glow)" />
  </g>
</svg>"""


WORDMARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 160" width="750" height="160">
  <g id="Wordmark" fill="#FFFFFF">
    <!-- V -->
    <path d="M 20 20 L 70 140 L 92 140 L 142 20 L 115 20 L 81 108 L 47 20 Z" />
    <!-- A -->
    <path d="M 160 140 L 210 20 L 232 20 L 282 140 L 255 140 L 243 110 L 199 110 L 187 140 Z M 207 90 L 235 90 L 221 54 Z" />
    <!-- J -->
    <path d="M 300 20 L 325 20 L 325 110 Q 325 140 295 140 Q 275 140 265 125 L 280 108 Q 286 118 296 118 Q 303 118 303 108 Z" />
    <!-- R -->
    <path d="M 345 20 L 400 20 Q 425 20 425 48 Q 425 72 400 78 L 430 140 L 403 140 L 377 82 L 370 82 L 370 140 L 345 140 Z M 370 42 L 370 62 L 396 62 Q 403 62 403 52 Q 403 42 396 42 Z" />
    <!-- A -->
    <path d="M 445 140 L 495 20 L 517 20 L 567 140 L 540 140 L 528 110 L 484 110 L 472 140 Z M 492 90 L 520 90 L 506 54 Z" />
  </g>
</svg>"""


HORIZONTAL_LOCKUP_DARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 320" width="1200" height="320">
  <defs>
    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00F0FF" />
      <stop offset="50%" stop-color="#00C8FF" />
      <stop offset="100%" stop-color="#0080FF" />
    </linearGradient>
    <linearGradient id="top-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#5B38D5" />
      <stop offset="100%" stop-color="#321D87" />
    </linearGradient>
    <linearGradient id="mid-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6F44F0" />
      <stop offset="100%" stop-color="#4625AF" />
    </linearGradient>
    <linearGradient id="bot-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3A1F99" />
      <stop offset="100%" stop-color="#1C0E56" />
    </linearGradient>
    <linearGradient id="top-right-facet" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#482AB6" />
      <stop offset="100%" stop-color="#241366" />
    </linearGradient>
    <linearGradient id="mid-right-facet" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#5D35D9" />
      <stop offset="100%" stop-color="#331A8B" />
    </linearGradient>
    <linearGradient id="bot-right-facet" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2A1473" />
      <stop offset="100%" stop-color="#13083B" />
    </linearGradient>
  </defs>

  <rect width="100%" height="100%" fill="#060814" />

  <!-- Symbol Scale & Position -->
  <g transform="translate(60, 20) scale(0.56)">
    <path d="M 250 40 L 205 52 L 135 170 L 205 148 Z" fill="url(#top-left-facet)" />
    <path d="M 135 170 L 85 245 L 175 315 L 205 148 Z" fill="url(#mid-left-facet)" />
    <path d="M 85 245 L 170 400 L 235 372 L 175 315 Z" fill="url(#bot-left-facet)" />
    <path d="M 175 315 L 235 372 L 238 260 L 205 148 Z" fill="#20115B" />
    <path d="M 250 40 L 295 52 L 365 170 L 295 148 Z" fill="url(#top-right-facet)" />
    <path d="M 365 170 L 415 245 L 325 315 L 295 148 Z" fill="url(#mid-right-facet)" />
    <path d="M 415 245 L 330 400 L 265 372 L 325 315 Z" fill="url(#bot-right-facet)" />
    <path d="M 325 315 L 265 372 L 262 260 L 295 148 Z" fill="#150A3E" />
    <path d="M 250 40 L 268 148 L 250 250 L 232 148 Z" fill="url(#cyan-glow)" />
    <path d="M 244 250 L 244 380 L 250 410 L 256 380 L 256 250 Z" fill="url(#cyan-glow)" />
  </g>

  <!-- Wordmark -->
  <g transform="translate(380, 70) scale(1.1)" fill="#FFFFFF">
    <path d="M 20 20 L 70 140 L 92 140 L 142 20 L 115 20 L 81 108 L 47 20 Z" />
    <path d="M 160 140 L 210 20 L 232 20 L 282 140 L 255 140 L 243 110 L 199 110 L 187 140 Z M 207 90 L 235 90 L 221 54 Z" />
    <path d="M 300 20 L 325 20 L 325 110 Q 325 140 295 140 Q 275 140 265 125 L 280 108 Q 286 118 296 118 Q 303 118 303 108 Z" />
    <path d="M 345 20 L 400 20 Q 425 20 425 48 Q 425 72 400 78 L 430 140 L 403 140 L 377 82 L 370 82 L 370 140 L 345 140 Z M 370 42 L 370 62 L 396 62 Q 403 62 403 52 Q 403 42 396 42 Z" />
    <path d="M 445 140 L 495 20 L 517 20 L 567 140 L 540 140 L 528 110 L 484 110 L 472 140 Z M 492 90 L 520 90 L 506 54 Z" />
  </g>

  <!-- Tagline -->
  <text x="405" y="245" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-weight="700" font-size="28" fill="#00D2FF" letter-spacing="9">BUILD. TRAIN. VERIFY. SCALE.</text>
</svg>"""


HORIZONTAL_LOCKUP_LIGHT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 320" width="1200" height="320">
  <defs>
    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00B8E6" />
      <stop offset="100%" stop-color="#0077CC" />
    </linearGradient>
    <linearGradient id="top-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4B28C5" />
      <stop offset="100%" stop-color="#241075" />
    </linearGradient>
  </defs>

  <rect width="100%" height="100%" fill="#FFFFFF" />

  <!-- Symbol -->
  <g transform="translate(60, 20) scale(0.56)">
    <path d="M 250 40 L 205 52 L 135 170 L 205 148 Z" fill="#3B1DA6" />
    <path d="M 135 170 L 85 245 L 175 315 L 205 148 Z" fill="#502AD9" />
    <path d="M 85 245 L 170 400 L 235 372 L 175 315 Z" fill="#291176" />
    <path d="M 175 315 L 235 372 L 238 260 L 205 148 Z" fill="#1B0A52" />
    <path d="M 250 40 L 295 52 L 365 170 L 295 148 Z" fill="#30158F" />
    <path d="M 365 170 L 415 245 L 325 315 L 295 148 Z" fill="#4321B8" />
    <path d="M 415 245 L 330 400 L 265 372 L 325 315 Z" fill="#200C61" />
    <path d="M 325 315 L 265 372 L 262 260 L 295 148 Z" fill="#12053A" />
    <path d="M 250 40 L 268 148 L 250 250 L 232 148 Z" fill="url(#cyan-glow)" />
    <path d="M 244 250 L 244 380 L 250 410 L 256 380 L 256 250 Z" fill="url(#cyan-glow)" />
  </g>

  <!-- Wordmark -->
  <g transform="translate(380, 70) scale(1.1)" fill="#0B0E1E">
    <path d="M 20 20 L 70 140 L 92 140 L 142 20 L 115 20 L 81 108 L 47 20 Z" />
    <path d="M 160 140 L 210 20 L 232 20 L 282 140 L 255 140 L 243 110 L 199 110 L 187 140 Z M 207 90 L 235 90 L 221 54 Z" />
    <path d="M 300 20 L 325 20 L 325 110 Q 325 140 295 140 Q 275 140 265 125 L 280 108 Q 286 118 296 118 Q 303 118 303 108 Z" />
    <path d="M 345 20 L 400 20 Q 425 20 425 48 Q 425 72 400 78 L 430 140 L 403 140 L 377 82 L 370 82 L 370 140 L 345 140 Z M 370 42 L 370 62 L 396 62 Q 403 62 403 52 Q 403 42 396 42 Z" />
    <path d="M 445 140 L 495 20 L 517 20 L 567 140 L 540 140 L 528 110 L 484 110 L 472 140 Z M 492 90 L 520 90 L 506 54 Z" />
  </g>

  <!-- Tagline -->
  <text x="405" y="245" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-weight="700" font-size="28" fill="#0088CC" letter-spacing="9">BUILD. TRAIN. VERIFY. SCALE.</text>
</svg>"""


VERTICAL_LOCKUP_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 650" width="600" height="650">
  <defs>
    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00F0FF" />
      <stop offset="100%" stop-color="#0080FF" />
    </linearGradient>
    <linearGradient id="top-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#5B38D5" />
      <stop offset="100%" stop-color="#321D87" />
    </linearGradient>
    <linearGradient id="mid-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6F44F0" />
      <stop offset="100%" stop-color="#4625AF" />
    </linearGradient>
    <linearGradient id="bot-left-facet" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3A1F99" />
      <stop offset="100%" stop-color="#1C0E56" />
    </linearGradient>
  </defs>

  <rect width="100%" height="100%" fill="#060814" />

  <!-- Centered Symbol -->
  <g transform="translate(140, 30) scale(0.64)">
    <path d="M 250 40 L 205 52 L 135 170 L 205 148 Z" fill="url(#top-left-facet)" />
    <path d="M 135 170 L 85 245 L 175 315 L 205 148 Z" fill="url(#mid-left-facet)" />
    <path d="M 85 245 L 170 400 L 235 372 L 175 315 Z" fill="url(#bot-left-facet)" />
    <path d="M 175 315 L 235 372 L 238 260 L 205 148 Z" fill="#20115B" />
    <path d="M 250 40 L 295 52 L 365 170 L 295 148 Z" fill="#482AB6" />
    <path d="M 365 170 L 415 245 L 325 315 L 295 148 Z" fill="#5D35D9" />
    <path d="M 415 245 L 330 400 L 265 372 L 325 315 Z" fill="#2A1473" />
    <path d="M 325 315 L 265 372 L 262 260 L 295 148 Z" fill="#150A3E" />
    <path d="M 250 40 L 268 148 L 250 250 L 232 148 Z" fill="url(#cyan-glow)" />
    <path d="M 244 250 L 244 380 L 250 410 L 256 380 L 256 250 Z" fill="url(#cyan-glow)" />
  </g>

  <!-- Wordmark -->
  <g transform="translate(68, 350) scale(0.82)" fill="#FFFFFF">
    <path d="M 20 20 L 70 140 L 92 140 L 142 20 L 115 20 L 81 108 L 47 20 Z" />
    <path d="M 160 140 L 210 20 L 232 20 L 282 140 L 255 140 L 243 110 L 199 110 L 187 140 Z M 207 90 L 235 90 L 221 54 Z" />
    <path d="M 300 20 L 325 20 L 325 110 Q 325 140 295 140 Q 275 140 265 125 L 280 108 Q 286 118 296 118 Q 303 118 303 108 Z" />
    <path d="M 345 20 L 400 20 Q 425 20 425 48 Q 425 72 400 78 L 430 140 L 403 140 L 377 82 L 370 82 L 370 140 L 345 140 Z M 370 42 L 370 62 L 396 62 Q 403 62 403 52 Q 403 42 396 42 Z" />
    <path d="M 445 140 L 495 20 L 517 20 L 567 140 L 540 140 L 528 110 L 484 110 L 472 140 Z M 492 90 L 520 90 L 506 54 Z" />
  </g>

  <!-- Tagline -->
  <text x="300" y="520" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-weight="700" font-size="22" fill="#00D2FF" text-anchor="middle" letter-spacing="8">BUILD. TRAIN. VERIFY. SCALE.</text>
</svg>"""


MONOCHROME_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 320" width="1200" height="320">
  <!-- Symbol -->
  <g transform="translate(60, 20) scale(0.56)" fill="#000000">
    <path d="M 250 40 L 205 52 L 135 170 L 205 148 Z" />
    <path d="M 135 170 L 85 245 L 175 315 L 205 148 Z" />
    <path d="M 85 245 L 170 400 L 235 372 L 175 315 Z" />
    <path d="M 175 315 L 235 372 L 238 260 L 205 148 Z" />
    <path d="M 250 40 L 295 52 L 365 170 L 295 148 Z" />
    <path d="M 365 170 L 415 245 L 325 315 L 295 148 Z" />
    <path d="M 415 245 L 330 400 L 265 372 L 325 315 Z" />
    <path d="M 325 315 L 265 372 L 262 260 L 295 148 Z" />
    <path d="M 250 40 L 268 148 L 250 250 L 232 148 Z" />
    <path d="M 244 250 L 244 380 L 250 410 L 256 380 L 256 250 Z" />
  </g>

  <!-- Wordmark -->
  <g transform="translate(380, 90) scale(1.1)" fill="#000000">
    <path d="M 20 20 L 70 140 L 92 140 L 142 20 L 115 20 L 81 108 L 47 20 Z" />
    <path d="M 160 140 L 210 20 L 232 20 L 282 140 L 255 140 L 243 110 L 199 110 L 187 140 Z M 207 90 L 235 90 L 221 54 Z" />
    <path d="M 300 20 L 325 20 L 325 110 Q 325 140 295 140 Q 275 140 265 125 L 280 108 Q 286 118 296 118 Q 303 118 303 108 Z" />
    <path d="M 345 20 L 400 20 Q 425 20 425 48 Q 425 72 400 78 L 430 140 L 403 140 L 377 82 L 370 82 L 370 140 L 345 140 Z M 370 42 L 370 62 L 396 62 Q 403 62 403 52 Q 403 42 396 42 Z" />
    <path d="M 445 140 L 495 20 L 517 20 L 567 140 L 540 140 L 528 110 L 484 110 L 472 140 Z M 492 90 L 520 90 L 506 54 Z" />
  </g>
</svg>"""


SOCIAL_PREVIEW_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#060814" />
      <stop offset="100%" stop-color="#0F142D" />
    </linearGradient>
    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00F0FF" />
      <stop offset="100%" stop-color="#0080FF" />
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="url(#bg)" />

  <!-- Subtle Geometric Grid Lines in Right Margin -->
  <path d="M 850 0 L 1280 430 M 950 0 L 1280 330 M 1050 0 L 1280 230" stroke="#1F2A56" stroke-width="2" opacity="0.4" />

  <!-- Symbol Left -->
  <g transform="translate(100, 120) scale(0.8)">
    <path d="M 250 40 L 205 52 L 135 170 L 205 148 Z" fill="#5B38D5" />
    <path d="M 135 170 L 85 245 L 175 315 L 205 148 Z" fill="#6F44F0" />
    <path d="M 85 245 L 170 400 L 235 372 L 175 315 Z" fill="#3A1F99" />
    <path d="M 175 315 L 235 372 L 238 260 L 205 148 Z" fill="#20115B" />
    <path d="M 250 40 L 295 52 L 365 170 L 295 148 Z" fill="#482AB6" />
    <path d="M 365 170 L 415 245 L 325 315 L 295 148 Z" fill="#5D35D9" />
    <path d="M 415 245 L 330 400 L 265 372 L 325 315 Z" fill="#2A1473" />
    <path d="M 325 315 L 265 372 L 262 260 L 295 148 Z" fill="#150A3E" />
    <path d="M 250 40 L 268 148 L 250 250 L 232 148 Z" fill="url(#cyan-glow)" />
    <path d="M 244 250 L 244 380 L 250 410 L 256 380 L 256 250 Z" fill="url(#cyan-glow)" />
  </g>

  <!-- Wordmark -->
  <g transform="translate(520, 180) scale(1.1)" fill="#FFFFFF">
    <path d="M 20 20 L 70 140 L 92 140 L 142 20 L 115 20 L 81 108 L 47 20 Z" />
    <path d="M 160 140 L 210 20 L 232 20 L 282 140 L 255 140 L 243 110 L 199 110 L 187 140 Z M 207 90 L 235 90 L 221 54 Z" />
    <path d="M 300 20 L 325 20 L 325 110 Q 325 140 295 140 Q 275 140 265 125 L 280 108 Q 286 118 296 118 Q 303 118 303 108 Z" />
    <path d="M 345 20 L 400 20 Q 425 20 425 48 Q 425 72 400 78 L 430 140 L 403 140 L 377 82 L 370 82 L 370 140 L 345 140 Z M 370 42 L 370 62 L 396 62 Q 403 62 403 52 Q 403 42 396 42 Z" />
    <path d="M 445 140 L 495 20 L 517 20 L 567 140 L 540 140 L 528 110 L 484 110 L 472 140 Z M 492 90 L 520 90 L 506 54 Z" />
  </g>

  <!-- Subtitle -->
  <text x="545" y="355" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" font-size="32" fill="#00D2FF">Open-Source AI Framework</text>
  <text x="545" y="420" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="400" font-size="22" fill="#8C9BC5">Training  •  Evaluation  •  Reproducibility</text>
  <text x="545" y="460" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="400" font-size="22" fill="#8C9BC5">Deterministic Releases  •  Verification  •  Production Ready</text>
</svg>"""


# -----------------------------------------------------------------------------
# Save SVG Files
# -----------------------------------------------------------------------------

svg_files = {
    BRANDING_DIR / "symbol.svg": SYMBOL_SVG,
    BRANDING_DIR / "wordmark.svg": WORDMARK_SVG,
    BRANDING_DIR / "horizontal-lockup.svg": HORIZONTAL_LOCKUP_DARK_SVG,
    BRANDING_DIR / "vertical-lockup.svg": VERTICAL_LOCKUP_SVG,
    BRANDING_DIR / "logo.svg": HORIZONTAL_LOCKUP_DARK_SVG,
    BRANDING_DIR / "logo-dark.svg": HORIZONTAL_LOCKUP_DARK_SVG,
    BRANDING_DIR / "logo-light.svg": HORIZONTAL_LOCKUP_LIGHT_SVG,
    BRANDING_DIR / "logo-monochrome.svg": MONOCHROME_SVG,
}

for path, content in svg_files.items():
    path.write_text(content.strip(), encoding="utf-8")
    print(f"Saved SVG: {path}")


# -----------------------------------------------------------------------------
# Render PNG Exports via resvg-py & PIL
# -----------------------------------------------------------------------------

def render_svg_to_png(svg_content: str, output_path: Path, width: int | None = None, height: int | None = None) -> None:
    png_bytes = svg_to_bytes(svg_string=svg_content, width=width, height=height)
    output_path.write_bytes(png_bytes)
    print(f"Rendered PNG ({output_path}): {output_path.stat().st_size} bytes")


# Standard exports
render_svg_to_png(HORIZONTAL_LOCKUP_DARK_SVG, BRANDING_DIR / "logo.png", width=1200)
render_svg_to_png(HORIZONTAL_LOCKUP_DARK_SVG, BRANDING_DIR / "logo-dark.png", width=1200)
render_svg_to_png(HORIZONTAL_LOCKUP_LIGHT_SVG, BRANDING_DIR / "logo-light.png", width=1200)
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "symbol.png", width=500)
render_svg_to_png(WORDMARK_SVG, BRANDING_DIR / "wordmark.png", width=750)
render_svg_to_png(HORIZONTAL_LOCKUP_DARK_SVG, BRANDING_DIR / "horizontal-lockup.png", width=1200)
render_svg_to_png(VERTICAL_LOCKUP_SVG, BRANDING_DIR / "vertical-lockup.png", width=600)
render_svg_to_png(MONOCHROME_SVG, BRANDING_DIR / "logo-monochrome.png", width=1200)

# Avatars & Icons
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "favicon-16.png", width=16, height=16)
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "favicon-32.png", width=32, height=32)
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "favicon-64.png", width=64, height=64)
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "apple-touch-icon.png", width=180, height=180)
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "app-icon-512.png", width=512, height=512)

# GitHub Circular Avatar
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "github-avatar.png", width=512, height=512)

# Hugging Face Gold Squircle Avatar
render_svg_to_png(SYMBOL_SVG, BRANDING_DIR / "huggingface-avatar.png", width=512, height=512)

# Banners & Social Previews
render_svg_to_png(SOCIAL_PREVIEW_SVG, BRANDING_DIR / "social-banner.png", width=1200, height=630)
render_svg_to_png(SOCIAL_PREVIEW_SVG, BRANDING_DIR / "social-preview-1280x640.png", width=1280, height=640)


# -----------------------------------------------------------------------------
# Populate Subdirectories: branding/web/ and branding/social/
# -----------------------------------------------------------------------------

# Web
render_svg_to_png(HORIZONTAL_LOCKUP_DARK_SVG, WEB_DIR / "header-dark.png", width=1200)
render_svg_to_png(HORIZONTAL_LOCKUP_LIGHT_SVG, WEB_DIR / "header-light.png", width=1200)
render_svg_to_png(HORIZONTAL_LOCKUP_DARK_SVG, WEB_DIR / "navbar-logo.png", width=400)
render_svg_to_png(SYMBOL_SVG, WEB_DIR / "hero-symbol.png", width=800)
render_svg_to_png(SYMBOL_SVG, WEB_DIR / "favicon.png", width=64)
render_svg_to_png(VERTICAL_LOCKUP_SVG, WEB_DIR / "mobile-logo.png", width=400)

# Social
render_svg_to_png(SOCIAL_PREVIEW_SVG, SOCIAL_DIR / "linkedin-banner.png", width=1584, height=396)
render_svg_to_png(SOCIAL_PREVIEW_SVG, SOCIAL_DIR / "twitter-header.png", width=1500, height=500)
render_svg_to_png(VERTICAL_LOCKUP_SVG, SOCIAL_DIR / "square-announcement.png", width=1080, height=1080)
render_svg_to_png(SYMBOL_SVG, SOCIAL_DIR / "discord-icon.png", width=512, height=512)
render_svg_to_png(SYMBOL_SVG, SOCIAL_DIR / "slack-icon.png", width=512, height=512)
render_svg_to_png(HORIZONTAL_LOCKUP_DARK_SVG, SOCIAL_DIR / "email-signature.png", width=400)

print("\nSuccessfully generated all Vajra brand vectors, exports, and social/web assets!")
