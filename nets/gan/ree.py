import cv2

image = cv2.imread("rs_masked_test/0000b86e2fd18333_surgical.jpg")

image_dim = [691, 1024]  # Height, Width

# mask_xtl, mask_ytl, mask_xtm, mask_ytm, mask_xtr, mask_ytr, mask_xbr, mask_ybr, mask_xbm, mask_ybm, mask_xbl, mask_ybl
pnts = [733, 413, 759, 411, 850, 428, 841, 467, 758, 488, 741, 464]

x = pnts[0::2]
y = pnts[1::2]

for i in range(len(x)):
    cv2.circle(image, (int(800*x[i]/1024),
                       int(800*y[i]/1024)), 2, (0, 255, 0), -1)
# cv2.circle(image, (733, 413), 10, (0, 255, 0), -1)
# cv2.circle(image, (759, 411), 10, (0, 255, 0), -1)
# cv2.circle(image, (850, 428), 10, (0, 255, 0), -1)
# cv2.circle(image, (841, 467), 10, (0, 255, 0), -1)
# cv2.circle(image, (758, 488), 10, (0, 255, 0), -1)
# cv2.circle(image, (741, 464), 10, (0, 255, 0), -1)

# cv2.circle(image, (733, 411), 10, (0, 0, 255), -1)

# cv2.circle(image, (850, 488), 3, (0, 0, 255), -1)

# def handle_mouse_click(e, x, y, flags, param):

#     if e == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
#         cv2.imshow("Image", image)
#         print("X: " + str(x))
#         print("Y: " + str(y))


cv2.imshow("Image", image)

# cv2.setMouseCallback("Image", handle_mouse_click)

cv2.waitKey(0)

cv2.destroyAllWindows()
