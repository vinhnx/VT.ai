# GPT-Image-1 Advanced Prompt Guide

This guide demonstrates how to create effective prompts for GPT-Image-1 with VT.ai's advanced settings.

## Overview of Available Settings

VT.ai now provides access to all of GPT-Image-1's advanced settings:

- **Image Size**: Choose from auto, 1024x1024 (square), 1536x1024 (landscape), or 1024x1536 (portrait)
- **Image Quality**: auto, high, medium, or low
- **Background Style**: auto, transparent (for PNG), or opaque
- **Output Format**: png (best for graphics), jpeg (best for photos), or webp (web-optimized)
- **Compression Level**: 10-100 (for jpeg/webp, higher = better quality)
- **Moderation Level**: auto or low

## Sample Prompts by Use Case

### 1. Logo Design (Transparent Background)

Setup:

- Image Size: 1024x1024
- Quality: high
- Background: transparent
- Format: png
- Moderation: auto

```
Generate a minimalist logo for a tech startup called "QuantumLeap".
The logo should feature a stylized letter 'Q' that transforms into an upward arrow.
Use a gradient of blue to purple colors. Keep the design clean with plenty of negative space.
Make sure it works well on both light and dark backgrounds.
```

### 2. Realistic Product Photography

Setup:

- Image Size: 1536x1024
- Quality: high
- Background: auto
- Format: jpeg
- Compression: 90
- Moderation: auto

```
Create a professional product photograph of a modern smartphone with a sleek black design.
The phone should be displayed at a 3/4 angle on a minimalist white surface with soft shadows.
There should be a subtle reflection on the surface.
The phone screen should display a colorful home screen with app icons.
Use dramatic studio lighting with a soft blue accent light from the left side.
```

### 3. Artistic Illustration for Web

Setup:

- Image Size: auto
- Quality: medium
- Background: opaque
- Format: webp
- Compression: 80
- Moderation: low

```
Create a whimsical illustration of a tree house library in an ancient oak tree.
The scene should be set at golden hour with warm sunlight filtering through the leaves.
Small fairy lights should be strung around the branches, and a spiral staircase should wrap around the trunk.
The interior of the tree house should be visible through large windows, showing floor-to-ceiling bookshelves filled with colorful books.
A cozy reading nook with cushions should be visible near a window.
The style should be similar to Studio Ghibli films - detailed but with a painterly quality.
```

### 4. Technical Diagram/Infographic

Setup:

- Image Size: 1024x1024
- Quality: high
- Background: transparent
- Format: png
- Moderation: auto

```
Create a clear technical diagram of a renewable energy smart grid system.
The diagram should include solar panels, wind turbines, energy storage systems, and a control center.
Use a clean, modern design with a color-coded legend explaining each component.
Include directional arrows showing energy flow throughout the system.
Add simple, concise labels for each major component.
The style should be professional and suitable for a business presentation.
```

### 5. Portrait Photography

Setup:

- Image Size: 1024x1536
- Quality: high
- Background: auto
- Format: jpeg
- Compression: 100
- Moderation: auto

```
Generate a professional studio portrait of a female CEO in her 40s with short dark hair.
She should be wearing a tailored navy blue suit jacket over a white blouse.
The lighting should be high-key with soft shadows and a subtle rim light.
The background should be a gradient of light gray.
She should have a confident, approachable expression with a slight smile.
The composition should be from chest up, with her body turned slightly and face toward the camera.
```

## Best Practices

1. **Be specific about style**: Mention artistic styles, lighting conditions, and mood
2. **Describe composition**: Include details about perspective, framing, and focus
3. **Specify subject details**: Describe colors, textures, shapes, and arrangements
4. **Match settings to purpose**: Use transparent backgrounds for logos, high quality for detailed work
5. **Experiment with size**: Use portrait orientation (1024x1536) for vertical subjects, landscape (1536x1024) for scenes
6. **Test quality settings**: Sometimes "medium" quality with better prompting creates better results than "high" quality with vague prompts

## Advanced Techniques

### Subject Emphasis

To emphasize specific elements, describe them first and in more detail:

```
A vintage red bicycle leaning against an old brick wall. The bicycle has a wicker basket filled with wildflowers. The scene is set on a cobblestone street in a European town. Morning sunlight casts long shadows across the street.
```

### Style Control

Reference specific artistic styles for more consistent results:

```
In the style of [artist/movement], create a [subject] with [specific details]...
```

### Compositional Control

Control the composition by specifying camera settings:

```
A close-up view of [subject]...
Shot from a bird's-eye perspective of [subject]...
A wide-angle view of [landscape] with [subject] in the foreground...
```
