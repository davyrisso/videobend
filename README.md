# Videobend
A library and scripts to apply glitch effects on videos.

## Structure
 - `effects/` 
 - `utils/` 

## Usage
The effects in the library are available for use both as libraries and scripts.

Scripts can be called with the `-m` option.

### Examples

As library:
```python
from videobend.effects import motionmosh
from videobend.utils import video

input_video = video.VideoReader(file_path=input_video_path)
frames = motionmosh.GenerateFrames(input_video, ...)
```

As script:
```sh
$ python -m videobend.effects.motionmosh <input_video> <output_video> [-fx_s <effect_start>] [-fx_e <effect_end>]
```

For help on the effects do:
```sh
$ python -m videobend.effects.motionmosh --help
```

## Effects

### Motionmosh
Reproduces the datamosh effect on a video using Optical Flow

- [ ] TODO(davyrisso): Write doc.