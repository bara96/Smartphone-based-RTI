import cv2
import numpy as np


class ChessboardDetector:

    def __init__(self, min_contour_len=100, polydp_epsilon=50, refineCornerHWinsize=9):
        self.min_contour_len = min_contour_len
        self.polydp_epsilon = polydp_epsilon
        self._generate_chessboard_model()
        self.refineCornerHWinsize = refineCornerHWinsize

    def _generate_chessboard_model(self):

        self.oplane = np.array([[18, 0], [0, 0], [0, 14], [18, 14]])

        (XX, YY) = np.meshgrid(np.linspace(2.0, 16.0, 8), np.linspace(2, 12, 6))
        self.ZZ = np.array([((i + j) % 2) * 255 for i in range(0, 6) for j in range(0, 8)])
        self.ZZ[-1] = 255
        self.ZZ = np.reshape(self.ZZ, (6, 8))
        self.chessboard_points = np.vstack((XX.flatten(), YY.flatten(), np.ones((1, np.size(XX)))))

        (Xmod, Ymod) = np.meshgrid(np.linspace(3.0, 15.0, 7), np.linspace(3, 11, 5))
        self.model_points = np.vstack((Xmod.flatten(), Ymod.flatten(), np.zeros((1, np.size(Xmod)))))
        self.model_points = self.model_points[:, :-1]

    def __call__(self, img):

        if len(img.shape) > 2 and img.shape[2] > 1:
            self.dbg_img = np.copy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.dbg_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        ret, bw = cv2.threshold(img, 50, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        bw_er = cv2.dilate(bw, None, iterations=2)

        for cvec in contours:

            cvec = np.squeeze(cvec)
            if cvec.shape[0] < self.min_contour_len:
                continue

            quad = cv2.approxPolyDP(cvec, self.polydp_epsilon, True)
            quad = np.squeeze(quad)
            if quad.shape[0] != 4:
                continue

            # for ii in range(0, len(quad)-1):
            #    cv2.line( dbg, tuple( quad[ii] ),  tuple( quad[ii+1] ), (0,0,255), 2 )
            # cv2.line( dbg, tuple( quad[0] ),  tuple( quad[3] ), (0,0,255), 2 )

            for rot in range(0, 4):
                curr_oplane = np.roll(self.oplane, rot, axis=0)
                H = np.matrix(cv2.findHomography(curr_oplane, quad, 0)[0])

                ptsI = H * self.chessboard_points
                ptsI = ptsI / ptsI[2, :]

                # for ii in range(0, ptsI.shape[1] ):
                #    cv2.drawMarker( dbg, tuple( ptsI[0:2,ii].getA1().astype( np.uint32 ) ), (100,100,0), cv2.MARKER_CROSS, 10, 2 )
                #    cv2.putText( dbg, "%d"%ii, tuple( ptsI[0:2,ii].getA1().astype( np.uint32 ) ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0) )

                xcoords = np.clip(ptsI[0, :].getA1().astype(np.uint32), 0, bw.shape[1] - 1)
                ycoords = np.clip(ptsI[1, :].getA1().astype(np.uint32), 0, bw.shape[0] - 1)
                imgvals = np.reshape(bw_er[ycoords, xcoords], self.ZZ.shape)

                if np.allclose(imgvals, self.ZZ):
                    aux = np.copy(self.model_points)
                    aux[2, :] = 1
                    corner_points = H * aux
                    self.corners = corner_points[0:2, :] / corner_points[2, :]

                    # Subpix corner refinement
                    # Note: cornerSubPix wants an Nx1x2 tensor
                    #
                    aux = np.ascontiguousarray(np.expand_dims(np.transpose(self.corners), axis=1), dtype=np.float32)
                    aux = cv2.cornerSubPix(img, aux, (self.refineCornerHWinsize, self.refineCornerHWinsize), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.001))

                    self.corners = np.matrix(np.transpose(np.squeeze(aux)))

                    for ii in range(0, self.corners.shape[1]):
                        pos = tuple(self.corners[:, ii].getA1().astype(np.uint32))
                        cv2.drawMarker(self.dbg_img, pos, (100, 100, 0), cv2.MARKER_SQUARE,
                                       self.refineCornerHWinsize * 2 + 1, 2)
                        cv2.putText(self.dbg_img, "%d" % ii, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 155), 2)

                    self.H = H
                    return True

        return False


if __name__ == '__main__':
    cd = ChessboardDetector()

    # I = cv2.imread('snaps/img_00005.png', cv2.IMREAD_GRAYSCALE )
    # cd( I )
    # cv2.imshow('dbg', cd.dbg_img )
    # cv2.waitKey(0)

    from CamViewerDart import CameraViewer

    cam = CameraViewer()

    running = True
    while running:

        img = cam.get_frame()
        if not cam.is_valid():
            running = False
            continue

        cd(img)
        cv2.imshow('camera', cd.dbg_img)
        keyp = cv2.waitKey(1)
        cam.control(keyp)
        running = keyp != 113  # Press q to exit
