# MakeItTalk 


## Required packages
- ffmpeg and ffmpeg-python (version >= 0.2.0)
- pynormalize
- pytorch


## How to use

### Step 1. Git clone

### Step 2. Create root directory
 - create root directory ```ROOT_DIR```
 - add sub folders ```ckpt```, ```dump```, ```nn_result```, ```puppets```, ```raw_wav```, ```test_wav_files```  to it.
 
### Step 3. Import pre-trained model and demo Wilk files
- put pre-trained face expression model under ```ROOT_DIR/ckpt/BEST_CONTENT_MODEL/ckpt_best_model.pth```
- put pre-trained face pose model under ```ROOT_DIR/ckpt/BEST_POSE_MODEL/ckpt_last_epoch.pth```
- put wilk demo files ```wilk_face_close_mouth.txt``` and ```wilk_face_open_mouth.txt``` under ```puppets```

### Step 4. Import your test audio wav file
- put your test audio file like ```example.wav``` under ```test_wav_files``` folder

### Step 5. Run Talking Toon model
- change the ```ROOT_DIR``` in ```main_sneak_demo.py``` line 10 to your own ```ROOT_DIR```
- run
```
python main_sneak_demo.py
```
- its process has 3 steps in details:
    - create input data for network from your test audio file
    - run Talking Toon neural network to get the predicted facial landmarks
    - post process output files into real image scale for later image morphing
    
- its outputs are under ```ROOT_DIR/nn_result/sneak_demo```
    - raw facial landmark prediction visualization mp4 file, i.e. ```*_pos_EVAL_av.mp4```
    - a folder with your test audio name, containing 3 required files for later image morphing
        - ```reference_points.txt```
        - ```triangulation.txt```
        - ```warped_points.txt```
        
### Step 6. Image morphing (through Jakub's code)
- rebuild Jakub's code with my updated ```dingwarp.cpp```
- copy 3 required files to Jakub's code directory ```dingwarp/test/```
- run ```test_win.bat``` or do with normal cmd commands.
- run ``final_ffmpeg_combine.bat`` like this
```
>> final_ffmpeg_combine.bat [YOUR_TEST_AUDIO_FILE_DIR] [OUTPUT_VIDEO_NAME]
```
for exmaple
```
>> final_ffmpeg_combine.bat E:\TalkingToon\test_wav_files\example.wav output.mp4
```


# [License](LICENSE.md)


    