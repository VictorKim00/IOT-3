from gpiozero import DigitalInputDevice
from picamera2 import Picamera2
from time import sleep
from datetime import datetime
from pathlib import Path

pir = DigitalInputDevice(4)

save_dir = Path.home() / "motion_photos"
save_dir.mkdir(exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

print("AM312 motion sensor ready...")
sleep(2)

while True:
    if pir.value:
        print("움직임 감지!")

        filename = save_dir / datetime.now().strftime("image_%Y%m%d_%H%M%S.jpg")
        picam2.capture_file(str(filename))

        print("사진 저장 완료:", filename)
        sleep(5)

    sleep(0.1)