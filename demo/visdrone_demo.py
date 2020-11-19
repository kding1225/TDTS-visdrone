# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from fcos_core.config import cfg
from predictor import VisDroneDemo
import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/visdrone_tdts/tdts_R_50_FPN_1x_640x1024_visdrone_cn_mw1.5-nms0.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="models/tdts_R_50_FPN_1x_640x1024_visdrone_cn_mw1.5-nms0.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--results-dir",
        default="demo/results",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=640,  # 800
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    
    thresholds_for_classes = [1.0, 0.4543384611606598, 0.4528161883354187, 0.4456373155117035,
                              0.4930519461631775, 0.49669983983039856, 0.4916415810585022,
                              0.43324407935142517, 0.4070464074611664, 0.49178892374038696,
                              0.43258824944496155, 1.0]

    demo_im_names = os.listdir(args.images_dir)
    demo_im_names.sort()
    print('{} images to test'.format(len(demo_im_names)))

    # prepare object that handles inference plus adds predictions on top of image
    demo = VisDroneDemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    if args.results_dir:

        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        # plt
        for i, im_name in enumerate(demo_im_names):
            img = cv2.imread(os.path.join(args.images_dir, im_name))
            if img is None:
                continue
            start_time = time.time()
            demo.run_det_on_opencv_image_plt(img, os.path.join(args.results_dir, im_name))
            print("{}, {}\tinference time: {:.2f}s".format(i, im_name, time.time() - start_time))
        print("Done!")
    else:
        for im_name in demo_im_names:
            img = cv2.imread(os.path.join(args.images_dir, im_name))
            if img is None:
                continue
            start_time = time.time()
            composite = demo.run_on_opencv_image(img)
            print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
            cv2.imwrite(os.path.join('result', im_name), composite)
            # cv2.imshow(im_name, composite)
        print("Press any keys to exit ...")
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

