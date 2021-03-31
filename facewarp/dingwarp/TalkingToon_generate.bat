@echo off
setlocal

pushd output

REM USAGE:
     REM TOOL_NAME  image triangulation reference_points warped_points [-novsync | -wireframe | -dump]

REM OPTIONAL PARAMETERS:
    REM -novsync    Disables the VSync. Faster processing, but not so nice to watch.
    REM -wireframe  Renders a debug wireframe.
    REM -dump       Dumps every processed frame to an image file with name format %%06i.tga.


call ffmpeg -r 100 -f image2 -i "%%06d.tga" -i ..\intern\SemanticLipsync\root\test_wav_files\obama_example.wav  -shortest ooooo2.mp4


popd