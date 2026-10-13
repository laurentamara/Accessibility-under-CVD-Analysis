"""
CVD Accessibility Analysis Pipeline for RICO Dataset
=====================================================
Detects "hidden violations" — UI elements that pass WCAG contrast
under normal vision but fail under Color Vision Deficiency simulation.

"""

import json
import os
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
# 1. CVD SIMULATION (Machado et al. 2009)
#    Matrices for severity = 1.0 (full dichromacy)
#    These operate in LINEAR RGB space (not sRGB!)
# =============================================================================

# Protanopia (no L-cones, ~1% of males)
MACHADO_PROTAN = np.array([
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281,  0.099216],
    [-0.003882, -0.048116, 1.051998]
])

# Deuteranopia (no M-cones, ~1% of males)
MACHADO_DEUTAN = np.array([
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501,  0.047413],
    [-0.011820, 0.042940, 0.968881]
])

# Tritanopia (no S-cones, rare, ~0.003%)
MACHADO_TRITAN = np.array([
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900]
])

CVD_MATRICES = {
    'protan': MACHADO_PROTAN,
    'deutan': MACHADO_DEUTAN,
    'tritan': MACHADO_TRITAN,
}


def srgb_to_linear(c):
    """Remove sRGB gamma curve. Input: 0-255 uint8, Output: 0-1 linear float."""
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    """Apply sRGB gamma curve. Input: 0-1 linear float, Output: 0-255 uint8."""
    c = np.clip(c, 0, 1)
    srgb = np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def simulate_cvd(rgb_255, cvd_matrix):
    """
    Simulate CVD for a single RGB color (0-255).
    Proper pipeline: sRGB -> linear -> CVD matrix -> linear -> sRGB
    """
    linear = srgb_to_linear(np.array(rgb_255, dtype=np.float64))
    simulated = cvd_matrix @ linear
    return linear_to_srgb(simulated)


# =============================================================================
# 2. WCAG CONTRAST RATIO (W3C standard)
# =============================================================================

def relative_luminance(rgb_255):
    """Calculate WCAG 2.1 relative luminance from sRGB (0-255)."""
    linear = srgb_to_linear(np.array(rgb_255, dtype=np.float64))
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def wcag_contrast_ratio(rgb1, rgb2):
    """Calculate WCAG 2.1 contrast ratio between two sRGB colors."""
    L1 = relative_luminance(rgb1)
    L2 = relative_luminance(rgb2)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)


# =============================================================================
# 3. RICO JSON PARSER
#    Extract text elements (TextView) with bounding boxes from view hierarchy
# =============================================================================

def extract_text_elements(node, parent_bounds=None, elements=None, depth=0):
    """
    Recursively extract visible text elements from RICO view hierarchy.
    Returns list of dicts with element bounds and parent bounds.
    """
    if elements is None:
        elements = []

    # Skip invisible elements
    visibility = node.get('visibility', 'visible')
    visible_to_user = node.get('visible-to-user', True)
    if visibility != 'visible' or not visible_to_user:
        return elements

    bounds = node.get('bounds', [0, 0, 0, 0])
    class_name = node.get('class', '')

    # Check if this is a text element (exclude ImageButton - icon only, no text)
    is_text = ('TextView' in class_name or 'EditText' in class_name or 
               ('Button' in class_name and 'ImageButton' not in class_name))

    if is_text:
        # Element must have non-zero area
        x1, y1, x2, y2 = bounds
        width = x2 - x1
        height = y2 - y1
        if width > 5 and height > 5:  # Minimum size filter
            elements.append({
                'class': class_name,
                'bounds': bounds,
                'parent_bounds': parent_bounds if parent_bounds else bounds,
                'depth': depth,
            })

    # Recurse into children
    children = node.get('children', [])
    for child in children:
        extract_text_elements(child, parent_bounds=bounds, elements=elements, depth=depth + 1)

    return elements


def parse_rico_json(json_path):
    """Parse a RICO JSON file and extract text elements."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    root = data.get('activity', {}).get('root', {})
    if not root:
        return [], [0, 0, 0, 0]

    root_bounds = root.get('bounds', [0, 0, 0, 0])
    elements = extract_text_elements(root)
    return elements, root_bounds


# =============================================================================
# 4. COLOR SAMPLING
#    Ring-based background sampling with text pixel exclusion
# =============================================================================

def sample_foreground_color(img_array, bounds, scale_x, scale_y):
    """
    Extract text (foreground) color using K-Means K=2 clustering.
    
    Rationale: A text element's bounding box is mostly background pixels.
    K-Means with K=2 separates pixels into two clusters:
      - Larger cluster  = background (majority)
      - Smaller cluster = text foreground (minority)
    
    This is standard, well-understood image processing that avoids
    arbitrary thresholds.
    """
    x1 = int(bounds[0] * scale_x)
    y1 = int(bounds[1] * scale_y)
    x2 = int(bounds[2] * scale_x)
    y2 = int(bounds[3] * scale_y)

    h, w = img_array.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    region = img_array[y1:y2, x1:x2]
    if region.size == 0:
        return None

    pixels = region.reshape(-1, 3).astype(np.float32)
    
    if len(pixels) < 10:
        return None
    
    # K-Means K=2 clustering
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=3, batch_size=max(1024, len(pixels)))
    labels = kmeans.fit_predict(pixels)
    
    # Count pixels in each cluster
    counts = [np.sum(labels == 0), np.sum(labels == 1)]
    centers = kmeans.cluster_centers_
    
    # Smaller cluster = text (foreground), larger = background
    fg_idx = 0 if counts[0] < counts[1] else 1
    bg_idx = 1 - fg_idx
    
    fg_color = centers[fg_idx]
    bg_color = centers[bg_idx]
    
    # Validate: clusters must be sufficiently different (>15 RGB distance)
    cluster_dist = np.sqrt(np.sum((fg_color - bg_color) ** 2))
    if cluster_dist < 15:
        return None  # No distinguishable text
    
    # Text must be minority (<45% of pixels) to be considered text
    fg_ratio = counts[fg_idx] / len(pixels)
    if fg_ratio > 0.45:
        return None  # Not a clear text/bg separation
    
    return fg_color.astype(np.uint8)


def sample_background_color(img_array, text_bounds, parent_bounds, scale_x, scale_y, pad=5):
    """
    Sample background color using ring approach:
    - Expand text bbox by pad pixels
    - Clip to parent bounds
    - Exclude text pixels
    - Take median of remaining pixels
    """
    # Scale bounds to image coordinates
    tx1 = int(text_bounds[0] * scale_x)
    ty1 = int(text_bounds[1] * scale_y)
    tx2 = int(text_bounds[2] * scale_x)
    ty2 = int(text_bounds[3] * scale_y)

    px1 = int(parent_bounds[0] * scale_x)
    py1 = int(parent_bounds[1] * scale_y)
    px2 = int(parent_bounds[2] * scale_x)
    py2 = int(parent_bounds[3] * scale_y)

    h, w = img_array.shape[:2]

    # Expanded ring region (clipped to parent and image bounds)
    rx1 = max(0, max(px1, tx1 - pad))
    ry1 = max(0, max(py1, ty1 - pad))
    rx2 = min(w, min(px2, tx2 + pad))
    ry2 = min(h, min(py2, ty2 + pad))

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    # Extract ring region
    region = img_array[ry1:ry2, rx1:rx2].copy()

    # Create mask: True = background pixel, False = text pixel
    mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)

    # Calculate text area within the ring region
    inner_x1 = max(0, tx1 - rx1)
    inner_y1 = max(0, ty1 - ry1)
    inner_x2 = min(rx2 - rx1, tx2 - rx1)
    inner_y2 = min(ry2 - ry1, ty2 - ry1)

    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    bg_pixels = region[mask]
    if bg_pixels.size == 0:
        # Fallback: just use parent region median
        parent_region = img_array[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
        if parent_region.size == 0:
            return None
        fallback_color = np.median(parent_region.reshape(-1, 3), axis=0).astype(np.uint8)
        fallback_std = float(np.std(parent_region.reshape(-1, 3), axis=0).mean())
        return fallback_color, fallback_std

    # Check background uniformity
    bg_std = np.std(bg_pixels.reshape(-1, 3), axis=0).mean()
    
    return np.median(bg_pixels.reshape(-1, 3), axis=0).astype(np.uint8), float(bg_std)


# =============================================================================
# 5. HIDDEN VIOLATION DETECTION
# =============================================================================

WCAG_AA_THRESHOLD = 4.5  # WCAG 2.1 SC 1.4.3 for normal text
WCAG_AA_LARGE_THRESHOLD = 3.0  # For large text (18pt+ or 14pt+ bold)


def analyze_element(fg_rgb, bg_rgb):
    """
    Analyze a single element for hidden CVD violations.
    Returns dict with normal and CVD contrast ratios.
    """
    normal_ratio = wcag_contrast_ratio(fg_rgb, bg_rgb)
    normal_pass = normal_ratio >= WCAG_AA_THRESHOLD

    results = {
        'fg_r': int(fg_rgb[0]), 'fg_g': int(fg_rgb[1]), 'fg_b': int(fg_rgb[2]),
        'bg_r': int(bg_rgb[0]), 'bg_g': int(bg_rgb[1]), 'bg_b': int(bg_rgb[2]),
        'normal_ratio': round(normal_ratio, 4),
        'normal_pass': normal_pass,
    }

    for cvd_name, cvd_matrix in CVD_MATRICES.items():
        fg_sim = simulate_cvd(fg_rgb, cvd_matrix)
        bg_sim = simulate_cvd(bg_rgb, cvd_matrix)
        cvd_ratio = wcag_contrast_ratio(fg_sim, bg_sim)
        cvd_pass = cvd_ratio >= WCAG_AA_THRESHOLD

        results[f'{cvd_name}_ratio'] = round(cvd_ratio, 4)
        results[f'{cvd_name}_pass'] = cvd_pass
        results[f'{cvd_name}_hidden'] = normal_pass and not cvd_pass

    # Any hidden violation across CVD types
    results['any_hidden'] = any(results[f'{cvd}_hidden'] for cvd in CVD_MATRICES)

    return results


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def process_screenshot(screenshot_id, data_dir):
    """
    Process a single RICO screenshot.
    Returns list of element analysis results.
    """
    img_path = os.path.join(data_dir, f'{screenshot_id}.jpg')
    json_path = os.path.join(data_dir, f'{screenshot_id}.json')

    if not os.path.exists(img_path) or not os.path.exists(json_path):
        return []

    # Load image
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    img_h, img_w = img_array.shape[:2]

    # Parse JSON
    elements, root_bounds = parse_rico_json(json_path)
    if not elements or root_bounds[2] == 0 or root_bounds[3] == 0:
        return []

    # Calculate scale factor (JSON coords -> image coords)
    scale_x = img_w / root_bounds[2]
    scale_y = img_h / root_bounds[3]

    results = []
    for elem in elements:
        # Sample foreground color
        fg_color = sample_foreground_color(img_array, elem['bounds'], scale_x, scale_y)
        if fg_color is None:
            continue

        # Sample background color
        bg_result = sample_background_color(
            img_array, elem['bounds'], elem['parent_bounds'], scale_x, scale_y
        )
        if bg_result is None:
            continue

        bg_color, bg_std = bg_result

        # Analyze element
        analysis = analyze_element(fg_color, bg_color)
        analysis['screenshot_id'] = screenshot_id
        analysis['element_class'] = elem['class']
        analysis['element_bounds'] = str(elem['bounds'])
        analysis['bg_std'] = round(bg_std, 2)  # Background uniformity flag

        results.append(analysis)

    return results


def run_pipeline(data_dir, output_csv, max_screenshots=None, screenshot_ids=None, verbose=True):
    """
    Run the full CVD analysis pipeline on RICO screenshots.

    If `screenshot_ids` is provided (e.g., the list in sampled_screen_ids.txt),
    exactly those screens are processed, matching the paper's sample. Otherwise
    all screens in `data_dir` are used, optionally capped by `max_screenshots`.
    """
    if screenshot_ids is None:
        all_files = os.listdir(data_dir)
        screenshot_ids = sorted(set(
            f.replace('.jpg', '') for f in all_files if f.endswith('.jpg')
        ), key=lambda x: int(x) if x.isdigit() else 0)
        if max_screenshots:
            screenshot_ids = screenshot_ids[:max_screenshots]

    if verbose:
        print(f"Processing {len(screenshot_ids)} screenshots...")

    # CSV columns
    fieldnames = [
        'screenshot_id', 'element_class', 'element_bounds', 'bg_std',
        'fg_r', 'fg_g', 'fg_b', 'bg_r', 'bg_g', 'bg_b',
        'normal_ratio', 'normal_pass',
        'protan_ratio', 'protan_pass', 'protan_hidden',
        'deutan_ratio', 'deutan_pass', 'deutan_hidden',
        'tritan_ratio', 'tritan_pass', 'tritan_hidden',
        'any_hidden',
    ]

    total_elements = 0
    total_hidden = 0
    total_screens_with_hidden = 0
    errors = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, sid in enumerate(screenshot_ids):
            try:
                results = process_screenshot(sid, data_dir)
                for r in results:
                    writer.writerow(r)

                n_elements = len(results)
                n_hidden = sum(1 for r in results if r['any_hidden'])
                total_elements += n_elements
                total_hidden += n_hidden
                if n_hidden > 0:
                    total_screens_with_hidden += 1

                if verbose and (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(screenshot_ids)}] "
                          f"ID={sid}, elements={n_elements}, hidden={n_hidden}, "
                          f"total_hidden={total_hidden}")
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  ERROR on {sid}: {e}")
                    traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Screenshots processed: {len(screenshot_ids)}")
    print(f"Errors: {errors}")
    print(f"Total text elements analyzed: {total_elements}")
    print(f"Total hidden CVD violations: {total_hidden}")
    print(f"Screenshots with hidden violations: {total_screens_with_hidden}")
    if total_elements > 0:
        print(f"Hidden violation rate: {total_hidden/total_elements*100:.2f}%")
    print(f"Results saved to: {output_csv}")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    DATA_DIR = 'unique_uis/combined'      # RICO screenshots + JSON, relative to repo root
    OUTPUT_CSV = 'results/cvd_results.csv'

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Reproduce the paper's sample: process exactly the screens listed in
    # sampled_screen_ids.txt (the 4,381 analyzed screens). Requires the raw RICO
    # screenshots + JSON under DATA_DIR. Delete/rename the file to process all screens.
    ids = None
    if os.path.exists('sampled_screen_ids.txt'):
        ids = [ln.strip() for ln in open('sampled_screen_ids.txt') if ln.strip()]
    run_pipeline(DATA_DIR, OUTPUT_CSV, screenshot_ids=ids, verbose=True)
