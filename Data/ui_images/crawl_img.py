import urllib.request
import os
import cv2
import imutils
os.chdir(r"C:\Users\KTL\Desktop\FYP-code\Data\ui_images")

URL = r"http://www-scf.usc.edu/~liwenhu/SHM/database/"
for i in range(1, 515):
    filename = "strands{}.png".format(str(i).zfill(5))
    print(URL+filename)
    try:
        img = urllib.request.urlopen(URL+filename).read()
        f = open(filename, 'wb')
        f.write(img)
        f.close()
        img = cv2.imread(filename)
        img = imutils.resize(img, width=128)
        cv2.imwrite(filename, img)
    except:
        pass


