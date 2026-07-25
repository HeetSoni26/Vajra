# GitHub Social Preview Image Specification

This document provides the design specification for generating the official GitHub social preview image (Open Graph image) for the **Vajra** repository.

---

## 1. Canvas & Export Specifications

- **Dimensions**: `1280 x 640 px` (Standard 2:1 aspect ratio for GitHub / Twitter card preview)
- **Export Format**: PNG (24-bit RGB) or WebP
- **Resolution**: `72 DPI` / `@2x` retina scaling (`2560 x 1280 px`)
- **Safe Margin**: `64 px` inset padding from all canvas borders

---

## 2. Color Palette & Aesthetics

### Dark Theme (Primary Brand Identity)
- **Background**: Deep Navy / Charcoal Gradient (`#0D1117` to `#161B22`)
- **Accent Glow**: Electric Violet / Sapphire Cyan Gradient (`#7C3AED` to `#2563EB`)
- **Primary Text**: Pure White (`#FFFFFF`)
- **Secondary Text**: Cool Gray (`#9CA3AF`)
- **Badge Accent**: Emerald Green (`#10B981`) for `v1.0.0` and `8/8 Verified` badges

### Light Theme Variant (Optional)
- **Background**: Off-White Gradient (`#F8FAFC` to `#F1F5F9`)
- **Accent**: Slate Violet (`#6D28D9`)
- **Primary Text**: Dark Slate (`#0F172A`)
- **Secondary Text**: Slate Gray (`#475569`)

---

## 3. Typography & Hierarchy

- **Title / Logotype**: `Inter`, `Outfit`, or `Fira Code` (Bold / Semi-Bold, `64 pt`, Line Height `1.1`)
  - Text: **Vajra**
- **Tagline**: `Inter` or `Roboto` (Medium, `28 pt`, Line Height `1.3`)
  - Text: *Open-Source Foundation Language Model Framework*
- **Sub-tagline**: `Fira Code` or `JetBrains Mono` (Regular, `18 pt`, `#9CA3AF`)
  - Text: `Decoder-Only Transformer • Deterministic Builds • 8/8 Verified`
- **Badge Text**: `Fira Code` (Semi-Bold, `14 pt`)
  - Pill 1: `v1.0.0 Release`
  - Pill 2: `248/248 Tests Passed`

---

## 4. Graphic Layout & Grid Structure

```
+-------------------------------------------------------------------+
|  [Safe Margin: 64px]                                             |
|                                                                   |
|   (Vajra Emblem / Geometric Tensor Node Icon)                     |
|                                                                   |
|   VAJRA                                                           |
|   Scalable Foundation Language Model Framework                    |
|                                                                   |
|   [Badge: v1.0.0]  [Badge: 8/8 Verified]  [Badge: 248 Tests Pass]  |
|                                                                   |
|   https://github.com/HeetSoni26/Vajra                             |
+-------------------------------------------------------------------+
```

- **Alignment**: Left-aligned content with vertical centering.
- **Icon Placement**: `80 x 80 px` stylized geometric emblem on top left.
- **Spacing**: `24 px` gap between emblem and title, `16 px` gap between title and tagline, `32 px` gap to status pills.
