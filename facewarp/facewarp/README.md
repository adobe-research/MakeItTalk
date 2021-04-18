# Facewarp
Simple triangle-mesh-based warping algorithm

## Cloning
Make sure to clone with the `--recursive` attribute to get all the dependencies.

## Check dependencies
Check if `glfw` and `stb` exsist in `third_party` folder. If not, you manually clone it by
```
cd third_party
git clone https://github.com/glfw/glfw.git
git clone https://github.com/nothings/stb.git
```

## Building
Install [CMake](https://cmake.org/download/) on your system, if it is not
already. Then do this in the root of the repository to create the project files:
```
mkdir build
cd build
cmake -G"<GENERATOR>" ..
```
On Linux, you cam simply run `cmake ..`.

On macOS, youcan simply run `cmake ..`. If the compiler fails, try `<GENERATOR>` as `Xcode`.

On Windows select one of the
Visual Studio flavors, e.g., `Visual Studio 15 2017 Win64`. You can also always
run `cmake --help` to see the list of all available generators.

Then you can build the project by:
```
cmake --build . --config Release
```

## Running
After building, the `bin` (or `bin\Release` in Windows) directory should contain the command
line utility. Here's its help. You can also access it by running it with no
arguments or by calling `facewarp --help`:
```
USAGE:
    facewarp image triangulation reference_points warped_points background_image [-novsync | -wireframe | -dump]

OPTIONAL PARAMETERS:
    -novsync    Disables the VSync. Faster processing, but not so nice to watch.
    -wireframe  Renders a debug wireframe.
    -dump       Dumps every processed frame to an image file with name format %06i.tga.
```

## Testing
You can test the facewarp by our provided test example
```
cd test/output
# Linux / macOS
../../bin/facewarp ../image.png ../triangulation.txt ../reference_points.txt ../warped_points.txt ../background_white.png -dump
ffmpeg -r 62.5 -f image2 -i %06d.tga -pix_fmt yuv420p ../test.mp4
# Windows
..\..\bin\Release\facewarp.exe ..\image.png ..\triangulation.txt ..\reference_points.txt ..\warped_points.txt ..\background_white.png -dump
ffmpeg -r 62.5 -f image2 -i %06d.tga -pix_fmt yuv420p -vf scale=800:-2 ..\test.mp4
```
If you can see the warped cartoon faces during the process and `.tga` image files in `test/output` folder and `test/test.mp4`, facewarp is then successfully built.
