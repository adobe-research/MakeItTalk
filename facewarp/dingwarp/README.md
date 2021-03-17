# dingwarp
Simple triangle-mesh-based warp for Ding's project.

## Cloning
Make sure to clone with the `--recursive` attribute to get all the dependencies.

## Building
Install [CMake](https://cmake.org/download/) on your system, if it is not
already. Then do this in the root of the repository to create the project files:
```
mkdir build
cd build
cmake -G"<GENERATOR>" ..
```
On macOS, `<GENERATOR>` will be `Xcode`, while on Windows select one of the
Visual Studio flavors, e.g., `Visual Studio 15 2017 Win64`. You can also always
run `cmake --help` to see the list of all available generators.

Then you can either build the project from the IDE, or can do (while still in
the build directory):
```
cmake --build . --config Debug
cmake --build . --config Release
```

Make sure you build at least the release version.

## Running
After building, the `bin/<Debug|Release>` directory should contain the command
line utility. Here's its help. You can also access it by running it with no
arguments or by calling `dingwarp --help`:
```
USAGE:
    dingwarp image triangulation reference_points warped_points [-novsync | -wireframe | -dump]

OPTIONAL PARAMETERS:
    -novsync    Disables the VSync. Faster processing, but not so nice to watch.
    -wireframe  Renders a debug wireframe.
    -dump       Dumps every processed frame to an image file with name format %06i.tga.
```

## Testing
For Windows, there's a [testing script](test/test_win.bat) you can run (you have
to run it from its directory). It will run the tool and dump the output images
as a sequence of TGA files.