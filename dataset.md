# Deepfake Detection Dataset Design

## Target Dataset Size (High Confidence Setup)
- **Recommended total images:** 60,000 to 90,000
- **Per class:** 20,000 to 30,000 images

This scale is strong enough to:
- Reach 80–90% accuracy with ResNet18 or EfficientNet
- Generalize across unseen AI models
- Learn subtle and localized manipulation artifacts
- Remain stable under compression and resizing

---

## Class Breakdown

### CLASS 0 — REAL IMAGES
- **Target:** 22,000 images
- **Purpose:** Real world distribution, camera noise learning, baseline authenticity

#### Sources & Types
- **FFHQ Real Faces:** 5,000 images
  - High resolution face portraits
  - Varied age, ethnicity, lighting
  - No filters, no edits
  - *Why:* Face realism anchor, matches deepfake face domains
- **MS COCO (Real Only):** 6,000 images
  - People in natural scenes, indoor/outdoor, objects, cluttered scenes
  - Mixed lighting
  - *Avoid:* Cartoon or illustration categories
  - *Why:* Non-face realism, real scene distribution
- **Open Images Dataset:** 5,000 images
  - Vehicles, buildings, street scenes, nature, animals
  - Real camera photos only
  - *Why:* Broad environmental diversity
- **ImageNet (Photo Subset Only):** 4,000 images
  - Natural objects, tools, animals, landscapes
  - *Avoid:* Artistic or synthetic images
  - *Why:* Texture and natural image statistics diversity
- **Optional Real Additions:**
  - Flickr, Unsplash, DSLR camera sets

### CLASS 1 — AI GENERATED
- **Target:** 22,000 images
- **Purpose:** Learn synthetic texture patterns, diffusion artifacts, GAN fingerprints

#### Sources & Types
- **StyleGAN/StyleGAN2/StyleGAN3:** 7,000 images
  - Face portraits, high realism, diverse ages/genders/lighting
  - *Why:* GAN artifact baseline, face generation artifacts
- **Stable Diffusion Outputs:** 7,000 images
  - Faces, landscapes, urban scenes, objects
  - *Must include:* Realistic prompts only
  - *Avoid:* Anime, stylized art
  - *Why:* Diffusion noise pattern learning
- **Midjourney/DALL·E (Research Mirrors):** 4,000 images
  - Ultra-realistic photography style, people, nature, interiors, street scenes
  - *Avoid:* Artistic or painterly styles
  - *Why:* Commercial AI fingerprint learning
- **LAION Diffusion Subset:** 4,000 images
  - Realistic internet-style images, mixed realism levels
  - *Why:* Prevent overfitting to one AI model

### CLASS 2 — AI EDITED (REAL + MANIPULATED)
- **Target:** 22,000 images
- **Purpose:** Learn localized manipulation, blending artifacts, inpainting traces

#### Sources & Types
- **FaceForensics++:** 6,000 images
  - Face swaps, reenactment, neural textures, compressed/uncompressed
  - *Why:* Core facial manipulation training
- **ForgeryNet:** 5,000 images
  - Face edits, object edits, scene edits
  - *Why:* Mixed manipulation diversity
- **CASIA Image Tampering:** 4,000 images
  - Splicing, copy-move, object insertion
  - *Why:* Classical edit learning
- **IMD2020:** 4,000 images
  - Inpainting, object removal, scene editing
  - *Why:* Localized artifact detection
- **DEFACTO:** 3,000 images
  - Semantic edits, AI-assisted object replacement, subtle edits
  - *Why:* Hard case manipulation learning

---

## Balanced Class Summary
| Class        | Target   | Percent |
|--------------|----------|---------|
| Real         | 22,000   | ~33%    |
| AI Generated | 22,000   | ~33%    |
| AI Edited    | 22,000   | ~33%    |
| **Total**    | 66,000   | Balanced|

This meets bias control and fairness constraints.

---

## Image Types to Include Per Class

### Real
- Faces (40%)
- Non-face scenes (60%)
- Indoor and outdoor
- Low-light and daylight
- Different camera qualities

### AI Generated
- GAN faces (30%)
- Diffusion faces (20%)
- Diffusion environments (30%)
- Mixed realism objects (20%)
- **Must avoid:** Anime, paintings, stylized fantasy

### AI Edited
- Face swaps (30%)
- Object insertion/removal (25%)
- Inpainting (20%)
- Background replacement (15%)
- Subtle edits (10%)

---

## Upper Limit for Maximum Success (4 TB SSD)
- **120,000 to 180,000 images total**
- **Per class:** 40,000 to 60,000 images

This scale:
- Rivals industry benchmark datasets
- Allows robust transformer or ConvNeXt training
- Handles unseen generative models better
- Supports paper-grade benchmarking

**Storage estimate at 512×512 JPEG Q85:**
- 150 KB per image
- 150,000 images ≈ 22.5 GB (well below 4 TB limit)
