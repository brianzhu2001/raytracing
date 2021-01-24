# Ray tracing

This is a Python 3 implementation of a ray tracing based image renderer, including support for texture mapping and smooth shading of triangle meshes. See `creeper.png` and `sheep.png` for examples of this program's output.

## Dependencies
Python 3, NumPy, PIL (Pillow or similar)

## Running the code
Provided examples for use are `creeper.py` and `sheep.py`, which render images of the respective creatures from the game Minecraft. These can be run from the command line with the following arguments.
* `--nx` width of output image, defaults to 256
* `--ny` height of output image, defaults to preserve a given camera aspect relative to `--nx`
* `--white` white point, defaults to 1.0
* `--outFile` output filename to write PNG, defaults to `creeper.png`, `sheep.png`, etc.

## Sources
* Creeper and sheep models courtesy of [Vincent Yanez](https://sketchfab.com/vinceyanez/collections/minecraft).
* `utils.py` provided by Abe Davis and Steve Marschner.
