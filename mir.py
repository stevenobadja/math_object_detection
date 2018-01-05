import os
import six.moves.urllib as urllib
import sys
import tarfile
import numpy as np
import tensorflow as tf
import cv2
from mir_help import *
from utils import label_map_util
from utils import visualization_utils as vis_util

# Set video capture from 2nd webcam
cap = cv2.VideoCapture(1)

# Record webcam activity
codec = cv2.VideoWriter_fourcc('D','I','V','X')
videoFile = cv2.VideoWriter();
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoFile = cv2.VideoWriter();
videoFile.open('video.avi', codec, 10, size, 1)

sys.path.append("..")

# Model trained with custom data
MODEL_NAME = 'mir_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
NUM_CLASSES = 23

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            # Start Camera, while true, camera will run
            ret, image_np = cap.read()

            # Set height and width of webcam
            height = 720
            width = 1280

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Detection equivalent to predict, will return confidence scores, classes,
            # box dimensions (ymin, xmin, ymax, xmax) & num of detection
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5)

            # Obtain classes and coordinates (xmin) as a list of tuples
            od_list = [[category_index.get(value).get('name'), boxes[0][index][1] * width] for index,
                   value in enumerate(classes[0]) if scores[0, index] > 0.65]

            # Reorder the tuples by their xmin coordinates
            od_list_seq = sorted(od_list, key=lambda x:(-x[1], x[0]), reverse=True)

            # Return only the classes from the tuples
            od_list_co = [seq[0] for seq in od_list_seq]

            # Convert labels into math operators
            od_list_co = convop(od_list_co)

            # Combine intergers between operators
            co_num_list = combint(od_list_co)

            # Convert all numbers into floats if list contains a division
            exp_result = chkfl(co_num_list)

            # Solve math expression and return result
            result = getresult(co_num_list, exp_result)

            # Convert math expression and result into a string
            if str(result) == '...':
                obj = str(exp_result)
            else:
                obj = str(exp_result) + ' is ' + str(result)

            # Set font, print math expression and result
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_np, obj, (150, 1000), font, 3, (0, 0, 0), 0, cv2.LINE_AA)

            # Record Video
            videoFile.write(image_np)

            # Set camera resolution and create a break function by pressing 'q'
            cv2.imshow('object detection', cv2.resize(image_np, (width, height)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                videoFile.release()
                cv2.destroyAllWindows()
                break
