import wget

import easydict
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import face_embedding
import argparse
import os
import logging

parser = argparse.ArgumentParser(description='face model test')
args = easydict.EasyDict()
args["image_size"] = '112,112'
args["model"] = '../models/model,0'
args["gpu"] = "-1"
args["det"] = 2
args["flip"] = 0
args["threshold"] = 1.24
model = face_embedding.FaceModel(args)

people_list = ["victor", "zhang", "zhou"]
info = ["You have a dental appointment today at 10:30 am in Santa Clara Hospital"
    , "Your daughter will visit you at 12 am at your home."
    , "Please don't forget to take pills every 8 hours from 8 am."]
print ("Here are the people we recognize: {}".format(people_list))
imgs_list = [[cv2.imread("{}/{}.jpeg".format(person, i)) for i in range(1, 5)] for person in people_list]
features_list = [[model.get_feature(im) for im in imgs] for imgs in imgs_list]


def match(f1s, img):
    f2 = model.get_feature(img)
    if f2 is None:
        return None
    dist = [np.sum(np.square(f1 - f2)) for f1 in f1s]
    # print(dist)
    sim = [np.dot(f1, f2.T) for f1 in f1s]
    # print(sim)
    matches_list = [d < 1.4 and s > 0.5 for (d, s) in zip(dist, sim)]
    print (sim)
    print (dist)
    logging.debug("Matches Scores list is {}".format(matches_list))
    if np.sum(matches_list) > 0:
        print ("Matched")
        return True
    return False


def faceRecognition(img2):
    # matching
    logging.debug("Start Matching")
    id = False
    for p, fList, mes in zip(people_list, features_list, info):
        id = match(fList, img2)
        if id is None:
            print ("No face is detected. Please move closer to the camera")
            return "Unknown", ""

        if id:
            print ("{}, Good to see you again :) How are you feeling today :)".format(p))
            print (mes)
            return p, mes

    if not id:
        return "Unknown", ""

# if __name__ == "__main__":
#     # load model
#     parser = argparse.ArgumentParser(description='face model test')
#     args = easydict.EasyDict()
#     args["image_size"] = '112,112'
#     args["model"] = '../models/model,0'
#     args["gpu"] = "-1"
#     args["det"] = 2
#     args["flip"] = 0
#     args["threshold"] = 1.24
#     model = face_embedding.FaceModel(args)
#
#     people_list = ["victor", "zhang", "zhou"]
#     info = ["You have a dental appointment today at 10:30 am in Santa Clara Hospital"
#         , "Your daughter will visit you at 12 am at your home."
#         , "Please don't forget to take pills every 8 hours from 8 am."]
#     print ("Here are the people we recognize: {}".format(people_list))
#     imgs_list = [[cv2.imread("{}/{}.jpeg".format(person, i)) for i in range(1, 5)] for person in people_list]
#
#     features_list = [[model.get_feature(im) for im in imgs] for imgs in imgs_list]
#     # for e in features_list:
#     #     for i in range(len(e)):
#     #         if e[i] is None:
#     #             print i
#     while True:
#         start = time.time()
#         # load img from wget
#         logging.debug("Load image from wget")
#         if os.path.exists("image.jpg"):
#             os.remove("image.jpg")
#         url = "http://192.168.1.101:5000/image.jpg"
#         # "http://192.168.0.12:5000/image.jpg"
#         filename = wget.download(url)
#         img2 = cv2.imread(filename)
#         # img2 = cv2.imread('zhang/1.jpeg')
#
#         # matchingls
#         logging.debug("Start Matching")
#         id = False
#         for p, fList, mes in zip(people_list, features_list, info):
#             id = match(fList, img2)
#             if id is None:
#                 print ("No face is detected. Please move closer to the camera")
#                 break
#
#             if id:
#                 print ("{}, Good to see you again :) How are you feeling today :)".format(p))
#                 print (mes)
#                 break
#
#         if not id:
#             print ("Matching Failed.")
#
#         print ("It takes {}s to do facial recognition".format(str(time.time() - start)[:4]))
#         if id:
#             plt.imshow(img2[:, :, ::-1])
#             plt.show()
#             # if recognize sleep 10 seconds
#             time.sleep(10)
