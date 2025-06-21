from PIL import Image

# Open the image file
img_path = '/home/jcd/Downloads/picking_ring.png'
img = Image.open(img_path)

# Convert image to RGBA (allows manipulation of pixel values)
img = img.convert("RGBA")

# Load the pixel data
pixels = img.load()

# Define the magenta color
magenta = (255, 0, 255, 255)

# Loop through the image and replace the blue/red color with magenta
for y in range(img.height):
    for x in range(img.width):
        r, g, b, a = pixels[x, y]
        # Replace blue and red tones with magenta
        if (b > 200 and r < 100) or (r > 200 and b < 100):
        # if  (b > 200):
            pixels[x, y] = magenta

# Save and display the modified image
output_path = '/home/jcd/Downloads/picking_ring_o_magenta.png'
# img.save(output_path)

img.show()
