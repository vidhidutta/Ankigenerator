from PIL import Image, ImageDraw

# 1. Create a blank RGBA canvas
img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))

# 2. Draw a semi-transparent red square
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0, 128), outline=(255, 0, 0, 255), width=3)

# 3. Save and re-open
img.save("alpha_test.png")
reopened = Image.open("alpha_test.png")

# 4. Check mode and a sample pixel’s alpha
print("Mode:", reopened.mode)
print("Sample pixel RGBA:", reopened.getpixel((60, 60)))

# 5. Confirm a corner outside the box is still fully transparent
print("Background pixel RGBA:", reopened.getpixel((10, 10)))
