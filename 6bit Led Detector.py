import cv2
import numpy as np
import imutils
import array as arr
import math
import sys


# upload your image name here
image = cv2.imread('speed.jpg')

font = cv2.FONT_HERSHEY_SIMPLEX
# make a copy  of your image
output = image.copy()
main = image.copy()

boundaries = [([235, 235, 235], [255, 255, 255])]

for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
 
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    blur = cv2.GaussianBlur(output, (5, 5), 0)
    res = cv2.fastNlMeansDenoisingColored(blur,None,10,10,7,21)

# convert into gray scale
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(gray, 128, 255, 0)
contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(main, contours, -1, (0,255,0), 3)

#finding centers of the images
xcenters=[]
ycenters=[]
tempycenters=[]
msg=[]

for c in contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
   
    xcenters.append(cX)
    ycenters.append(cY)

# number of centers found
n = len(xcenters)
dist = [0,0,0,0,0,0,0]

xcenters,ycenters = (list(t) for t in zip(*sorted(zip(xcenters,ycenters))))

# finding distance between each center
distance = math.sqrt( ((xcenters[0]-xcenters[n-1])**2)+((ycenters[0]-ycenters[n-1])**2) )
for x in range(n-1):
    dist[x]= math.sqrt( ((xcenters[x]-xcenters[x+1])**2)+((ycenters[x]-ycenters[x+1])**2) )

#average distance between each center
eD = distance/7

## finding 3 regions to determine how many leds are off
lv=round(eD-((eD*1)*0.33))
uv=round(eD+((eD*1)*0.33))
r1= range(lv,uv)

lv=round((2*eD)-((eD*2)*0.20))
uv=round((2*eD)+((eD*2)*0.20))
r2= range(lv,uv)

lv=round((3*eD)-((eD*3)*0.20))
uv=round((3*eD)+((eD*3)*0.20))
r3= range(lv,uv)

msg.append(1)
ln=1

#Checking for the leds
for x in range(0,n-1):
    temp_d=round(dist[x]-eD)
    
    if (len(msg)<8):
        if temp_d in r1:
            
            msg.append(0)
        elif temp_d in r2:
            msg.append(0)
            msg.append(0)
        elif temp_d in r3:
            
            msg.append(0)
            msg.append(0)
            msg.append(0)
        
        else:
            msg.append(1)
            
    else:
        break

# if our msg was incorrect run it again
if(len(msg)> 8 or len(msg)<7 ):
    msg=[]
    msg.append(1)
    
    for x in range(0,n-1):
        temp_d=round(dist[x]-eD)
        
        if (len(msg)<8):
            if temp_d in r1:
                
                msg.append(0)
            elif temp_d in r2:
                msg.append(0)
                msg.append(0)
            elif temp_d in r3:
                
                msg.append(0)
                msg.append(0)
                msg.append(0)
            elif temp_d<1 and ln==1:
                msg.append(1)
                msg.append(1)
                ln=0
            else:
                msg.append(1)
                
        else:
            break

if(len(msg)==7):
    msg.append(1)

# Final printing
print(msg)
speed=[]
for x in range (1,7):
    speed.append(msg[x])
print(speed)
s = ''.join(str(x) for x in speed)
print(s)

sp=int(s,2)
print(sp)



cv2.imshow('full',res)
cv2.imshow('main',main)
cv2.waitKey(0)
