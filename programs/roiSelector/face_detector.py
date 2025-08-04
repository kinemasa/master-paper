
import os
import cv2
import yaml
import numpy as np
import pandas as pd
import mediapipe as mp

class Param():
    max_num_faces= 1  # Number of target faces.
    minDetectionCon= 2.0e-1  # Detection confidence
    minTrackingCon= 5.0e-1  # Tracking confidence
    list_roi_num= [[10, 109, 108, 151, 337, 338], 
                    [67, 103, 104, 105, 66, 107, 108, 109], 
                    [297, 338, 337, 336, 296, 334, 333, 332], 
                    [151, 108, 107, 55, 8, 285, 336, 337], 
                    [8, 55, 193, 122, 196, 197, 419, 351, 417, 285], 
                    [197, 196, 3, 51, 5, 281, 248, 419], 
                    [4, 45, 134, 220, 237, 44, 1, 274, 457, 440, 363, 275], 
                    [134, 131, 49, 102, 64, 219, 218, 237, 220], 
                    [363, 440, 457, 438, 439, 294, 331, 279, 360], 
                    [5, 51, 45, 4, 275, 281], 
                    [3, 217, 126, 209, 131, 134], 
                    [248, 363, 360, 429, 355, 437], 
                    [188, 114, 217, 236, 196], 
                    [412, 419, 456, 437, 343], 
                    [2, 97, 167, 37, 0, 267, 393, 326], 
                    [97, 165, 185, 40, 39, 37, 167], 
                    [326, 393, 267, 269, 270, 409, 391], 
                    [97, 98, 203, 186, 185, 165], 
                    [326, 391, 409, 410, 423, 327], 
                    [54, 21, 162, 127, 116, 143, 156, 70, 63, 68], 
                    [284, 298, 293, 300, 383, 372, 345, 356, 389, 251], 
                    [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209], 
                    [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329], 
                    [203, 205, 187, 147, 177, 215, 138, 172, 136, 135, 212, 186, 206], 
                    [423, 426, 410, 432, 364, 365, 397, 367, 435, 401, 376, 411, 425], 
                    [18, 83, 182, 194, 32, 140, 176, 148, 152, 377, 400, 369, 262, 418, 406, 313], 
                    [57, 212, 210, 169, 150, 149, 176, 140, 204, 43], 
                    [287, 273, 424, 369, 400, 378, 379, 394, 430, 432]]
                    # The list containing sequence numbers of selected keypoints of different ROIs. Size = [num_roi].
    list_roi_name= ['lower medial forehead', 
                    'left lower lateral forehead', 
                    'right lower lateral forehead', 
                    'glabella', 
                    'upper nasal dorsum', 
                    'lower nasal dorsum', 
                    'soft triangle', 
                    'left ala', 
                    'right ala', 
                    'nasal tip', 
                    'left lower nasal sidewall', 
                    'right lower nasal sidewall', 
                    'left mid nasal sidewall', 
                    'right mid nasal sidewall', 
                    'philtrum', 
                    'left upper lip', 
                    'right upper lip', 
                    'left nasolabial fold', 
                    'right nasolabial fold', 
                    'left temporal', 
                    'right temporal', 
                    'left malar', 
                    'right malar', 
                    'left lower cheek', 
                    'right lower cheek', 
                    'chin', 
                    'left marionette fold', 
                    'right marionette fold']
                    # The list containing names of different ROIs. Size = [num_roi].

class FaceDetector():
    """A class for face detection, segmentation and RGB signal extraction."""

    def __init__(self, Params):
        """Class initialization.
        Parameters
        ----------
        Params: A class containing the pre-defined parameters.

        Returns
        -------

        """

        # Confidence.
        self.minDetectionCon = Params.minDetectionCon  # Minimal detection confidence.
        self.minTrackingCon = Params.minTrackingCon  # Minimal tracking confidence.
        # Mediapipe utils.
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)  # Face detection.
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utils.
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=Params.max_num_faces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackingCon
            )  # Face mesh.
        # ROI params.
        # The list containing sequence numbers of selected keypoints of different ROIs. Size = [num_roi].
        self.list_roi_num = np.array(Params.list_roi_num, dtype=object)
        # The list containing names of different ROIs. Size = [num_roi].
        self.list_roi_name = np.array(Params.list_roi_name, dtype=object)


    def extract_landmark(self, img):
        """Extract 2D keypoint locations.
        Parameters
        ----------
        img: The input image of the current frame. Channel = [B, G, R].

        Returns
        -------
        loc_landmark: Detected normalized 3D landmarks. Size=[468, 3].
        """
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # Apply face mesh.
        # Draw landmarks on the image.
        if results.multi_face_landmarks:
            # If the face is detected.
            # Loop over all detected faces.
            # In this experiment, we only detect one face in one video.
            for face_landmark in results.multi_face_landmarks:
                # Decompose the 3D face landmarks without resizing into the image size.
                loc_landmark = np.zeros([len(face_landmark.landmark), 3], dtype=np.float32)  # Coordinates of 3D landmarks.
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = face_landmark.landmark[i].x
                    loc_landmark[i, 1] = face_landmark.landmark[i].y
                    loc_landmark[i, 2] = face_landmark.landmark[i].z
        else:
            # If no face is detected.
            loc_landmark = np.nan
        
        return loc_landmark


    def extract_RGB(self, img, loc_landmark):
        """Extract RGB signals from the given image and ROI.
        Parameters
        ----------
        img: 2D image. Default in BGR style. Size=[height, width, 3]
        loc_landmark: Detected normalized (0-1) 3D landmarks. Size=[468, 3].

        Returns
        -------
        sig_rgb: RGB signal of the current frame as a numpy array. Size=[num_roi, 3].
        """

        if (np.isnan(loc_landmark)).any() == True:
            # If no face is detected.
            sig_rgb = np.nan
        else:
            # If the face is detected.
            # BGR -> RGB.
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Rescale the input landmarks location.
            height_img = img.shape[0]
            width_img = img.shape[1]
            loc_landmark[:, 0] = loc_landmark[:, 0] * width_img
            loc_landmark[:, 1] = loc_landmark[:, 1] * height_img
            # RGB signal initialization.
            sig_rgb = np.zeros(shape=[self.list_roi_num.shape[0], 3])
            # Loop over all ROIs.
            zeros = np.zeros(img.shape, dtype=np.uint8)
            for i_roi in range(0, self.list_roi_num.shape[0]):
                # Create the current ROI mask.
                roi_name = self.list_roi_name[i_roi]
                mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[self.list_roi_name==roi_name][0], :2].astype(int)], color=(1, 1, 1))
                # Only compute on a specific ROI.
                img_masked = np.multiply(img_RGB, mask)
                # Compute the RGB signal.
                sig_rgb[i_roi, :] = 3*img_masked.sum(0).sum(0)/(mask.sum())

        return sig_rgb


    def faceMeshDraw(self, img, roi_name):
        """Draw a face mesh annotations on the input image.
        Parameters
        ----------
        img: The input image of the current frame.
        roi_name: Name of the roi. The name should be in the name list.

        Returns
        -------
        img_draw: The output image after drawing the ROI of the current frame. 
        """
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # Apply face mesh.
        mp_face_mesh = mp.solutions.face_mesh_connections
        # Draw landmarks on the image.
        if results.multi_face_landmarks:
            # Loop over all detected faces.
            # In this experiment, we only detect one face in one video.
            for face_landmark in results.multi_face_landmarks:
                # Landmark points.
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmark,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
                )
                # Decompose the 3D face landmarks.
                height_img = img.shape[0]
                width_img = img.shape[1]
                loc_landmark = np.zeros([len(face_landmark.landmark), 2], dtype=np.int32)  # Coordinates of 2D landmarks.
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = face_landmark.landmark[i].x * width_img
                    loc_landmark[i, 1] = face_landmark.landmark[i].y * height_img
                # Create a zero vector for mask construction.
                zeros = np.zeros(img.shape, dtype=np.uint8)
                # ROI-forehead-nose-leftcheek-rightcheek-underlip. Colorization.
                mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[self.list_roi_name==roi_name][0], :]], color=(1, 1, 1))
                img_draw = img + mask * 50
            
        return img_draw
    
    def faceMeshDrawMultiple(self, img, roi_names):
        """Draw multiple ROI masks on the input image.
        
        Parameters
        ----------
        img: The input image of the current frame.
        roi_names: List of ROI names to be drawn.

        Returns
        -------
        img_draw: The output image after drawing multiple ROIs of the current frame. 
        """
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # Apply face mesh.
        mp_face_mesh = mp.solutions.face_mesh_connections

        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                # Base landmark drawing
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmark,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
                )
                # Convert landmarks to image coordinates
                height_img = img.shape[0]
                width_img = img.shape[1]
                loc_landmark = np.zeros([len(face_landmark.landmark), 2], dtype=np.int32)
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = int(face_landmark.landmark[i].x * width_img)
                    loc_landmark[i, 1] = int(face_landmark.landmark[i].y * height_img)
                # Draw all requested ROIs
                img_draw = img.copy()
                zeros = np.zeros(img.shape, dtype=np.uint8)
                for roi_name in roi_names:
                    if roi_name not in self.list_roi_name:
                        print(f"警告: ROI '{roi_name}' は未登録です。スキップします。")
                        continue
                    
                    roi_index = np.where(self.list_roi_name == roi_name)[0][0]
                    roi_points = loc_landmark[self.list_roi_num[roi_index], :]
                    
                    mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[roi_index], :]], color=(1, 1, 1))
                    img_draw += mask * 50  # 明るくハイライト
                    
                    # 輪郭に赤い線を描画
                    cv2.polylines(img_draw, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

                return img_draw
        
        return img
        

