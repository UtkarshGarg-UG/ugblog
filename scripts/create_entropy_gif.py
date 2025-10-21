#!/usr/bin/env python3
import time
from playwright.sync_api import sync_playwright
from PIL import Image
import os

def create_entropy_gif():
    # Paths
    html_path = "file://" + os.path.abspath("/Users/utkarshgarg/Documents/Code/my-blog/static/entropy-explorer/entropy_explorer.html")
    output_dir = "/Users/utkarshgarg/Documents/Code/my-blog/scripts/frames"
    os.makedirs(output_dir, exist_ok=True)

    # Sequence of words to demonstrate entropy changes
    word_sequence = [
        ("cat", 0.5, 1.5),  # (word, delay after typing, display duration)
        ("dog", 0.5, 1.5),
        ("pizza", 0.5, 1.5),
        ("rocket", 0.5, 1.5),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        # Navigate to the HTML file
        page.goto(html_path)
        page.wait_for_load_state("networkidle")
        time.sleep(2)  # Wait for initialization

        frame_count = 0

        # Capture initial state
        page.screenshot(path=f"{output_dir}/frame_{frame_count:03d}.png")
        frame_count += 1

        # Iterate through word sequence
        for word, type_delay, display_duration in word_sequence:
            # Clear input first
            input_field = page.locator('input[type="text"]').first
            input_field.click()
            input_field.fill("")
            time.sleep(0.3)

            # Type word character by character
            for char in word:
                input_field.type(char, delay=100)
                time.sleep(type_delay / len(word))
                page.screenshot(path=f"{output_dir}/frame_{frame_count:03d}.png")
                frame_count += 1

            # Hold on the complete word
            for _ in range(int(display_duration * 10)):  # 10 frames per second
                page.screenshot(path=f"{output_dir}/frame_{frame_count:03d}.png")
                frame_count += 1
                time.sleep(0.1)

        browser.close()

    # Create GIF from frames
    print(f"Creating GIF from {frame_count} frames...")
    frames = []
    for i in range(frame_count):
        img = Image.open(f"{output_dir}/frame_{i:03d}.png")
        # Crop to just the visualization area (adjust coordinates as needed)
        # This removes browser chrome and focuses on content
        img = img.crop((0, 0, 1400, 900))
        frames.append(img)

    # Save as GIF
    output_path = "/Users/utkarshgarg/Documents/Code/my-blog/static/entropy-explorer.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame = 10fps
        loop=0,
        optimize=True
    )

    print(f"GIF created: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Clean up frames
    for i in range(frame_count):
        os.remove(f"{output_dir}/frame_{i:03d}.png")
    os.rmdir(output_dir)

if __name__ == "__main__":
    create_entropy_gif()
