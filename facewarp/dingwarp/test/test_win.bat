@echo off
setlocal

pushd output

REM USAGE:
     REM TOOL_NAME  image triangulation reference_points warped_points [-novsync | -wireframe | -dump]

REM OPTIONAL PARAMETERS:
    REM -novsync    Disables the VSync. Faster processing, but not so nice to watch.
    REM -wireframe  Renders a debug wireframe.
    REM -dump       Dumps every processed frame to an image file with name format %%06i.tga.

call ..\..\bin\Release\dingwarp ^
    ..\image.png ^
    ..\triangulation.txt ^
    ..\reference_points.txt ^
    ..\warped_points.txt ^
	-novsync ^
	-dump

popd