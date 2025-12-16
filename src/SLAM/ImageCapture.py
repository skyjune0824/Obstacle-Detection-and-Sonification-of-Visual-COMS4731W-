import os
import cv2

class ImageFolderCapture:
    def __init__(self, folder_path):
        images = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        images.sort(key=lambda x: int(os.path.splitext(x)[0]))

        self.image_paths = [os.path.join(folder_path, f) for f in images]
        self.index = 0
        self.length = len(self.image_paths)

    def read(self):
        if self.index >= self.length:
            return False, None
        frame = cv2.imread(self.image_paths[self.index], cv2.IMREAD_GRAYSCALE)
        self.index += 1
        return True, frame

    def release(self):
        pass 