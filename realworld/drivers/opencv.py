"""OpenCV RGB driver shared by all workstation configurations."""


class USBCamera:
    def __init__(self, device):
        import cv2

        self.stream = cv2.VideoCapture(device)

    def read(self):
        import cv2

        ok, bgr = self.stream.read()
        if not ok:
            raise RuntimeError("RGB capture failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.stream.release()
