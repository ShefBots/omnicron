import numpy as np
from matplotlib import pyplot as plt

def main():
    zeros = np.zeros((30,30))
    drawLine(zeros, 0, 0, 15, 30)
    plt.imshow(zeros);
    plt.show()

def drawLine(matrix, x0, y0, x1, y1, color=255):
    xMin = min(x0,x1)
    xMax = max(x0,x1)
    if(abs(x0-x1) > abs(y0-y1)):
        # The x axis difference is bigger
        if(x0 > x1):
            # It does not ascend properly, swap 'em on the x.
            temp = x0
            x0 = x1
            x1 = temp
            temp = y0
            y0 = y1
            y1 = temp
        xdiff = x1-x0
        ydiff = y1-y0
        for x in range(x1-x0):
            y = y0+int((float(x)/xdiff)*ydiff)
            matrix[y, x0+x] = color
    else:
        # The y axis difference is bigger
        if(y0 > y1):
            # It does not ascend (but left?) properly, so flip 'em on the y.
            temp = y0
            y0 = y1
            y1 = temp
            temp = x0
            x0 = x1
            x1 = temp
        ydiff = y1-y0
        xdiff = x1-x0
        for y in range(y1-y0):
            x = x0+int((float(y)/ydiff)*xdiff)
            matrix[y0+y, x] = color

main();
