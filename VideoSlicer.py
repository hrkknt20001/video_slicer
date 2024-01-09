import os
import shutil
import cv2
import argparse


def video_slicer(video_file, image_dir, image_file='img_%s.png',
                   frame_per_sec=6, frame_dist=100, start_idx=0, angle=0.0, scale=1.0):
    """
    Save frame image from video file.
    :param video_file: Path of video file. (Input)
    :param image_dir: Directory path of save frame image. (output)
    :param image_file: Naming rule of image file. (default is img_%s.png)
    :param frame_per_sec: Number of frame per second.
    :param frame_dist: Distance of each frames.
    :param start_idx: start index of image file.
    :param angle: angle of rotate to image.
    :param scale: scale of image
    :param rule: naming rule of image file. (defalut is img_%s.png)
    :return:
    """
    # Delete the entire directory tree if it exists.
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    # Make the directory if it doesn't exist.
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Video to frames
    i = start_idx
    cap = cv2.VideoCapture(video_file)

    if frame_dist is not None:
        IMG_SIZE = (400, 400)
        base_des = None
        # detector = cv2.ORB_create()
        detector = cv2.AKAZE_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    while cap.isOpened():

        # load frame
        flag, frame = cap.read()

        # Is a frame left?
        if flag is False:
            break

        height = frame.shape[0]
        width = frame.shape[1]
        frame = frame[0 : height, int((width - height)/2): int((width - height)/2) + height]

        if frame_dist is None and cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_per_sec == 0:

            height = frame.shape[0]
            width = frame.shape[1]
            center = (int(width / 2), int(height / 2))

            # Geometric Image Transformations(use Affine Transform)
            trans = cv2.getRotationMatrix2D(center, angle, scale)
            frame = cv2.warpAffine(frame, trans, (width, height))

            cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)  # Save a frame
            print('Save', image_dir+image_file % str(i).zfill(6))
            i += 1

        else:
            if base_des is None :
                # Calculate feature point
                frame_resized = cv2.resize(frame, IMG_SIZE)
                (base_kp, base_des) = detector.detectAndCompute(frame_resized, None)

                height = frame.shape[0]
                width = frame.shape[1]
                center = (int(width / 2), int(height / 2))

                # Geometric Image Transformations(use Affine Transform)
                trans = cv2.getRotationMatrix2D(center, angle, scale)
                frame = cv2.warpAffine(frame, trans, (width, height))

                cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)  # Save a frame
                print('Save', image_dir+image_file % str(i).zfill(6))
                i += 1

            else:
                # Calculate feature point
                frame_resized = cv2.resize(frame, IMG_SIZE)
                (comp_kp, comp_des) = detector.detectAndCompute(frame_resized, None)
        
                # Finding distances from the results of feature matching
                matches = bf.match(base_des, comp_des)
                dist = [m.distance for m in matches]
                ret = sum(dist) / len(dist)

                if frame_dist < ret:

                    height = frame.shape[0]
                    width = frame.shape[1]
                    center = (int(width / 2), int(height / 2))

                    # Geometric Image Transformations(use Affine Transform)
                    trans = cv2.getRotationMatrix2D(center, angle, scale)
                    frame = cv2.warpAffine(frame, trans, (width, height))

                    cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)  # Save a frame
                    print('Save', image_dir+image_file % str(i).zfill(6))

                    base_des = comp_des
                    i += 1

    cap.release()  # When everything done, release the capture


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is save frame image from video file. ')
    parser.add_argument('arg1', help='input video file.')
    parser.add_argument('arg2', help='output directory.')
    parser.add_argument('--rule', help='naming rule of image file. ex) img_%s.png')
    parser.add_argument('--fps', help='Number of frame per second.', type=int, default=6)
    parser.add_argument('--dist', help='Distance of each frames.', type=int, default=None)
    parser.add_argument('--start_index', help='start index of image file.', type=int, default=0)
    parser.add_argument('--angle', help='angle of rotate to image.', type=float, default=0.0)
    parser.add_argument('--scale', help='scale of image', type=float, default=1.0)

    args = parser.parse_args()

    if args.rule is None:
        args.arg3 = 'img_%s.png'

    video_slicer(video_file=args.arg1, image_dir=args.arg2,
                 image_file=args.rule, frame_per_sec=args.fps, frame_dist=args.dist,
                 start_idx=args.start_index, angle=args.angle, scale=args.scale)