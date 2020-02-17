

class Frame():

    def __init__(self, pixels, position):
        self.__pixels = pixels
        self.__position = position

    @property
    def width(self):
        return self.__pixels[0].shape[0]

    @property
    def height(self):
        return self.__pixels[0].shape[1]

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def position(self):
        return self.__position

    @property
    def position_frames(self):
        return self.__position[0]

    @property
    def position_milliseconds(self):
        return self.__position[1]

    @property
    def pixels(self):
        return self.__pixels
