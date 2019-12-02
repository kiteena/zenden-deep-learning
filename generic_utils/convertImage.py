
import cv2
path = '/home/kristina/Downloads/car.jpg'
import base64

with open(path, "rb") as image:
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    print(img)
    file1 = open("MyFile.txt","wb") 
    file1.write(bitearray(img)
    file1.close()

                # bin_encode = open(f'{filename}',"rb").read()
            # fname = os.path.splitext(filename)[0] + '.jpg'
            # bin_decode = open(fname, "wb")
            # bin_decode.write(base64.decodestring(bin_encode))
            # bin_decode.close()
   
            # os.remove(filename)