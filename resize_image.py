import os
from PIL import Image

pictures_dir = "pictures"
setup = "me_vs_hdr"

for file_path in os.listdir(os.path.join(pictures_dir, setup)):
    pic_path = os.path.join(pictures_dir, setup, file_path)
    im = Image.open(pic_path)
    new_width = 1300
    new_height = 900
    im = im.resize((new_width, new_height), Image.ANTIALIAS)

    if not os.path.exists(os.path.join(pictures_dir, setup + "_resized")):
        os.mkdir(os.path.join(pictures_dir, setup + "_resized"))

    im.save(os.path.join(pictures_dir, setup + "_resized", file_path))
