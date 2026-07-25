# GitHub Repository Branding & Social Assets Guide

This guide details how to configure the official **Vajra** visual identity across GitHub repository settings, profile avatars, organization pages, and release assets.

---

## 1. Repository Social Preview Image

To set the official social preview card when sharing `github.com/HeetSoni26/vajra` on Twitter, LinkedIn, Discord, or Slack:

1. Navigate to the GitHub repository: `https://github.com/HeetSoni26/vajra/settings`
2. Scroll down to the **Social preview** section.
3. Click **Edit** -> **Upload an image...**
4. Select the asset located at:
   `branding/social-preview-1280x640.png` (or `branding/social-banner.png`).
5. Save changes.

---

## 2. Profile & Organization Avatar

To update your profile avatar or organization icon:

- **GitHub Avatar**: Upload `branding/github-avatar.png` (`512x512px`).
- **Hugging Face Avatar**: Upload `branding/huggingface-avatar.png` (`512x512px`).

---

## 3. README Logo Responsive Display

The `README.md` uses HTML `<picture>` elements for automatic dark/light theme switching:

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="branding/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="branding/logo-light.svg">
    <img alt="Vajra Logo" src="branding/logo-dark.svg" width="600">
  </picture>
</p>
```

---

## 4. Summary of Vector Branding Assets

| Asset Path | Resolution / Format | Primary Application |
| :--- | :--- | :--- |
| `branding/logo.svg` | Vector SVG | Primary Horizontal Lockup |
| `branding/symbol.svg` | Vector SVG | Trident Diamond Symbol |
| `branding/wordmark.svg` | Vector SVG | Geometric Text Wordmark |
| `branding/github-avatar.png` | `512x512` PNG | GitHub Profile & Organization Avatar |
| `branding/huggingface-avatar.png` | `512x512` PNG | Hugging Face Account / Model Profile |
| `branding/social-preview-1280x640.png` | `1280x640` PNG | GitHub Social Card Preview |
| `branding/social-banner.png` | `1200x630` PNG | Social Media Banner / Open Graph |
