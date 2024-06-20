import os
import cv2

class VideoFrameReader:
    def __init__(self, video_path, extensions=['.jpg', '.png']):
        self.fps = 12  # for video writer
        self.video_path = video_path
        self.frame_list = []
        self.extensions = extensions
        self.read_count = 0
        
        # Populate frame_list with valid images
        for im in sorted(os.listdir(video_path)):
            if self.has_extensions(im):
                self.frame_list.append(im)
        
        self.frame_len = len(self.frame_list)
        print(f"{video_path} has {self.frame_len} frames")
        
        self.init_frame_metadata()
    
    def init_frame_metadata(self):
        assert self.frame_len > 0, 'There should be at least one frame'
        
        # Read the first frame to get dimensions
        frame = cv2.imread(os.path.join(self.video_path, self.frame_list[0]))
        
        if frame is None:
            raise ValueError('The first frame could not be read, please check the frame paths.')
        
        self.height, self.width = frame.shape[:2]
        
        # Ensure dimensions are even
        self.height = self.height // 2 * 2
        self.width = self.width // 2 * 2
    
    def has_extensions(self, img_name):
        return any(img_name.endswith(ext) for ext in self.extensions)
    
    def __len__(self):
        return self.frame_len
    
    def __iter__(self):
        self.read_count = 0  # Reset the read count for new iterations
        return self
    
    def __next__(self):
        if self.read_count < self.frame_len:
            frame = self(self.read_count)  # Get the current frame
            self.read_count += 1
            return frame
        else:
            raise StopIteration
    
    def __call__(self, index=None):
        if index is None:
            index = self.read_count
            self.read_count += 1
        
        if index < 0 or index >= self.frame_len:
            raise IndexError(f'Trying to access frame beyond the bounds of video with len {self.frame_len}, and given index {index}')
        
        frame_path = os.path.join(self.video_path, self.frame_list[index])
        frame = cv2.imread(frame_path)
        
        if frame is None:
            raise ValueError(f'Frame at path {frame_path} could not be read.')
        
        # Convert grayscale frames to RGB
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Resize frame if necessary
        if self.width != frame.shape[1] or self.height != frame.shape[0]:
            frame = cv2.resize(frame, (self.width, self.height))
        
        return frame

