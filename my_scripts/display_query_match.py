#!/usr/bin/env python3

import cv2
import numpy as np
import h5py


class MatchingResult(object):

    def __init__(self, pairs_file_path, feat_matches_file_path, keypoints_file_path, name_filter = "_c0_"):
        self.image_pairs = self.read_image_pairs(pairs_file_path, name_filter)
        self.feat_matches_h5file = h5py.File(feat_matches_file_path, 'r')
        self.keypoints_h5file = h5py.File(keypoints_file_path, 'r')

    @staticmethod
    def read_image_pairs(pairs_file_path, name_filter):
        with open(pairs_file_path, 'r') as file:
            lines = file.readlines()
        pairs = [line.strip().split() for line in lines if not name_filter or name_filter in line ]
        return pairs

    def queries_list(self):
        return list(self.feat_matches_h5file.keys())

    def retrieve(self, query):
        return list(self.feat_matches_h5file[query].keys())

    def get_feature_matches(self, query, retrieved):
        point_matches = self.feat_matches_h5file[query][retrieved]
        matches0 = point_matches['matches0']
        matching_scores0 = point_matches['matching_scores0']
        assert(len(matches0) == len(matching_scores0))
        matches_and_scores = list()
        for i in range(len(matches0)):
            if matches0[i] != -1:
                matches_and_scores.append((i, matches0[i], matching_scores0[i]))
        return matches_and_scores

    def get_keypoints(self, query):
        keypoints = self.keypoints_h5file[query]['keypoints']
        keypoints = np.array(keypoints)
        # descriptors = self.keypoints_h5file[query]['descriptors']
        # scores = self.keypoints_h5file[query]['scores']
        # image_size = self.keypoints_h5file[query]['image_size']
        return keypoints

    def get_keypoint_matches(self, query, retrieved, top_k = 200):
        matches_and_scores = self.get_feature_matches(query, retrieved)
        matches_and_scores = sorted(matches_and_scores, key=lambda x: x[2], reverse=True)[:top_k]
        keypoints0 = []
        keypoints1 = []
        for i, j, score in matches_and_scores:
            keypoints0.append(self.get_keypoints(query)[i])
            keypoints1.append(self.get_keypoints(retrieved)[j])
        return keypoints0, keypoints1

# Draw matches
def draw_matches(image0_np, image1_np, points0_np, points1_np):
    # Create a new output image that concatenates the two images together
    h0, w0, _ = image0_np.shape
    h1, w1, _ = image1_np.shape
    output_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    output_img[:h0, :w0] = image0_np
    output_img[:h1, w0:] = image1_np

    # Draw lines between matching points
    for (x0, y0), (x1, y1) in zip(points0_np, points1_np):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(output_img, (int(x0), int(y0)), (int(x1) + w0, int(y1)), color, 1)
        cv2.circle(output_img, (int(x0), int(y0)), 2, color, -1)
        cv2.circle(output_img, (int(x1) + w0, int(y1)), 2, color, -1)
    
    cv2.putText(output_img, "Matches", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output_img, "num of matches: {}".format(len(points0_np)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return output_img

def display_matches(matching_result, dataset):
    index = 0
    total_pairs = len(matching_result.image_pairs)

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    
    while True:
        img0_key, img1_key = matching_result.image_pairs[index]
        img0_path = dataset + "/query/" + img0_key
        img1_path = dataset + "/database/" + img1_key
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)

        if img0 is None or img1 is None:
            print(f"Error loading images: {img0_path}, {img1_path}")
            break

        # # Resize images to the same height
        # height = min(img0.shape[0], img1.shape[0])
        # img0 = cv2.resize(img0, (int(img0.shape[1] * height / img0.shape[0]), height))
        # img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))

        # # Concatenate images horizontally
        # combined_image = np.hstack((img0, img1))

        keypoints0, keypoints1 = matching_result.get_keypoint_matches(img0_key, img1_key)
        combined_image = draw_matches(img0, img1, keypoints0, keypoints1)

        cv2.imshow('Matches', combined_image)

        while True:
            if cv2.getWindowProperty('Matches', cv2.WND_PROP_VISIBLE) < 1:
                key = 27  # The user closed the window, so treat it as an ESC key press
                break
            key = cv2.waitKey(100)
            if key != -1:
                break

        key = key & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 81:  # Left arrow key
            index = (index - 1) % total_pairs
        elif key == 83:  # Right arrow key
            index = (index + 1) % total_pairs

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    pairs_file_path = sys.argv[1]
    feat_matches_file_path = sys.argv[2]
    keypoints_file_path = sys.argv[3]
    dataset = sys.argv[4]
    if len(sys.argv) > 5:
        name_filter = sys.argv[5]
    else:
        name_filter = "_c0_"

    matching_result = MatchingResult(pairs_file_path, feat_matches_file_path, keypoints_file_path, name_filter=name_filter)

    display_matches(matching_result, dataset)


