import argparse
import sys

from PIL import Image
import numpy as np 
from scipy import misc
import tensorflow as tf
import os

import facenet
import align.detect_face

def main(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--model', type=str, help='directory containing pretained model.', default='')
    parser.add_argument('--verImg', type=str, help='first image.', default='')
    parser.add_argument('--regDatasetDir', type=str, help='second image.', default='')
    # parser.add_argument('--image_size', type=int,
    #         help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--thr', type=float,
            help='threshold to verification.', default=0.03)

    parser = parser.parse_args()

    if not (parser.model and parser.verImg and parser.regDatasetDir):
        print('usage: --im1 <path for image 1> --im2 <path for image 2>')
        exit(-1)

    # image_size = (parser.image_size, parser.image_size)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    # sess_d = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    # pnet, rnet, onet = align.detect_face.create_mtcnn(sess_d, None)

    # image_1 = get_deteted_image(pnet, rnet, onet, parser.i1)
    # Image.fromarray(np.uint8(image_1)).show()

    # image_2 = get_deteted_image(pnet, rnet, onet, parser.i2)
    # Image.fromarray(np.uint8(image_2)).show()

    # sess_d.close()

    with tf.Session() as sess:

        image_file_placeholder = tf.placeholder(tf.string, shape=(None), name='image_file')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # load model 
        input_map = {'phase_train': phase_train_placeholder}
        facenet.load_model(parser.model, input_map=input_map)

        # read image using tf
        file_contents = tf.read_file(image_file_placeholder)
        image = tf.image.decode_image(file_contents, 3)
        image = (tf.cast(image, tf.float32) - 127.5)/128.0
        distanceList = []
        fittedNameList = []
        datasetDir = parser.regDatasetDir
        if datasetDir[len(datasetDir)-1] != '/':
            datasetDir = datasetDir + '/'
        image_1 = sess.run(image, feed_dict={image_file_placeholder: parser.verImg})
        image_1 = image_1[np.newaxis, ...]
        number = 1
        for root,dirs,files in os.walk(datasetDir):
            for imgDir in dirs:
                fullDir = datasetDir+imgDir+'/'
                subDistance = []
                print('\nVelidating %d...\n'%number)
                for imgFile in os.listdir(fullDir):
                    image_2 = sess.run(image, feed_dict={image_file_placeholder:fullDir+imgFile})
                    image_2 = image_2[np.newaxis, ...]
                    graph = tf.get_default_graph()
                    image_batch = graph.get_tensor_by_name("image_batch:0")
                    embeddings = graph.get_tensor_by_name("embeddings:0")
                    feature_1 = sess.run(embeddings, feed_dict={phase_train_placeholder: False, image_batch: image_1})
                    feature_2 = sess.run(embeddings, feed_dict={phase_train_placeholder: False, image_batch: image_2})

                    distance = np.sum(np.sqrt((feature_1 - feature_2) ** 2)) / feature_1.shape[1]
                    if distance<parser.thr:
                        subDistance.append(distance)
                if len(subDistance)>0:
                    distanceList.append(min(subDistance))
                    fittedNameList.append(imgDir)
                number+=1
        print (distanceList)
        print (fittedNameList)
        if len(distanceList) == 0:
            print('This person is not in the register dataset')
        else:
            minDis = min(distanceList)
            index =0
            while index < len(distanceList):
                if distanceList[index] == minDis:
                    # [name, extension] = fittedNameList[index]
                    print('This person is in the register dataset. He/She is %s.'%fittedNameList[index])
                    break;

        # image_2 = sess.run(image, feed_dict={image_file_placeholder: parser.i2})
        # image_2 = image_2[np.newaxis, ...]

        # evaluate on test image
        # graph = tf.get_default_graph()
        # image_batch = graph.get_tensor_by_name("image_batch:0")
        # embeddings = graph.get_tensor_by_name("embeddings:0")
        # feature_1 = sess.run(embeddings, feed_dict={phase_train_placeholder:False, image_batch:image_1})
        # feature_2 = sess.run(embeddings, feed_dict={phase_train_placeholder:False, image_batch:image_2})
        #
        # distance = np.sum( np.sqrt((feature_1-feature_2)**2) )/feature_1.shape[1]
        # if distance < parser.thr:
        #     print('the same persons, distance: %f'%distance)
        # else:
        #     print('different persons,  distance: %f'%distance)


# def get_deteted_image(pnet, rnet, onet, image_path):
#     minsize = 20 # minimum size of face
#     threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
#     factor = 0.709 # scale factor
#     margin = 32
#     image_size = 160

#     # Add a random key to the filename to allow alignment using multiple processes
#     random_key = np.random.randint(0, high=99999)

#     try:
#         img = misc.imread(image_path)
#     except (IOError, ValueError, IndexError) as e:
#         errorMessage = '{}: {}'.format(image_path, e)
#         print(errorMessage)
#     else:
#         if img.ndim<2:
#             print('Unable to align "%s" because img.ndim<2' % image_path)
#             return None
#         if img.ndim == 2:
#             img = facenet.to_rgb(img)
#         img = img[:,:,0:3]

#         bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
#         nrof_faces = bounding_boxes.shape[0]
#         if nrof_faces>0:
#             det = bounding_boxes[:,0:4]
#             det_arr = []
#             img_size = np.asarray(img.shape)[0:2]
#             if nrof_faces>1:
#                 bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
#                 img_center = img_size / 2
#                 offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
#                 offset_dist_squared = np.sum(np.power(offsets,2.0),0)
#                 index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
#                 det_arr.append(det[index,:])
#             else:
#                 det_arr.append(np.squeeze(det))

#             for i, det in enumerate(det_arr):
#                 det = np.squeeze(det)
#                 bb = np.zeros(4, dtype=np.int32)
#                 bb[0] = np.maximum(det[0]-margin/2, 0)
#                 bb[1] = np.maximum(det[1]-margin/2, 0)
#                 bb[2] = np.minimum(det[2]+margin/2, img_size[1])
#                 bb[3] = np.minimum(det[3]+margin/2, img_size[0])
#                 cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
#                 scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                
#                 return scaled # return the first one only
#         else:
#             print('Unable to align "%s" because nrof_faces>1' % image_path)
#             return None

if __name__ == '__main__':
    main(sys.argv[1:])