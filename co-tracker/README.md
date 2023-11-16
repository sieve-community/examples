# Co-Tracker

This app takes a video, projects a mask of points onto the first frame, and propagates that mask,
tracking each point over the course of the video. 

If the "visualize" option is checked, the output tracks are projected onto a video, which is returned. Otherwise,
the raw arrays of the tracks and their visibility masks are returned instead.

The "grid size" option refers to the granularity of masks, in pixels, and the "pad size" option refers to how large of a padding to provide the visualization in the event of tracks moving off screen.