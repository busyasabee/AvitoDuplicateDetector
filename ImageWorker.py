import cv2
from matplotlib import pyplot as plt

class ImageWorker:
    img_num1 = None
    img_num2 = None
    path = 'E:/Study/Diploma/Avito duplicates/'
    images_paths = []
    images = []
    nums = []

    def read_color_images(self):
        for path in self.images_paths:
            image = cv2.imread(path)
            cv2.cvtColor(image,code=cv2.COLOR_BGR2RGB, dst=image)
            self.images.append(image)

    def read_lab_images(self):
        for path in self.images_paths:
            image = cv2.imread(path)
            self.images.append(image)
            cv2.cvtColor(image,code=cv2.COLOR_BGR2Lab, dst=image)

    def read_hls_images(self):
        for path in self.images_paths:
            image = cv2.imread(path)
            self.images.append(image)
            cv2.cvtColor(image,code=cv2.COLOR_BGR2HLS, dst=image)

    def show_images(self):
        fig = plt.figure("IMAGES")
        for (i,image) in enumerate(self.images):
            fig.add_subplot(1, len(self.images), i + 1)
            plt.imshow(image)

    def set_images_nums(self, nums):
        self.nums = nums
        for num in nums:
            path = self.create_image_path(num)
            self.images_paths.append(path)

    def show_rgb_histograms(self):
        plt.figure("Histograms")
        color = ('b','g','r')
        for (i, image) in enumerate(self.images):
            ax = plt.subplot(1, len(self.images), i+1)
            ax.set_title(self.nums[i])
            for (j, col) in enumerate(color):
                hist = cv2.calcHist([image],[j],None,[256],[0, 256])
                plt.plot(hist,color = col)
                plt.xlim([0, 256])

    def show_lab_histograms(self, histograms):
        plt.figure("Histograms")
        color = ('g','b')
        for (i, image) in enumerate(self.images):
            ax = plt.subplot(1, len(self.images), i+1)
            ax.set_title(self.nums[i])
            for (j, col) in enumerate(color):
                hist = cv2.calcHist([image],[j+1],None,[64],[-128, 127])
                plt.plot(hist,color = col)
                plt.xlim([-128, 127])

    def compare_histograms(self, histograms):
        OPENCV_METHODS = (
            ("Correlation", cv2.HISTCMP_CORREL),
            ("Chi-Squared", cv2.HISTCMP_CHISQR),
            ("Intersection", cv2.HISTCMP_INTERSECT),
            ("Hellinger", cv2.HISTCMP_HELLINGER),
            ("Bhattacharyya distance", cv2.HISTCMP_BHATTACHARYYA),
            ("Alternative Chi-Square", cv2.HISTCMP_CHISQR_ALT),
            ("Kullback-Leibler divergence",cv2.HISTCMP_KL_DIV)
        )

        for (i, histogram) in enumerate(histograms):
            if i != 0:
                print("Compare " + self.nums[0] + " " + self.nums[i])
                for (name, method) in OPENCV_METHODS:
                    d = cv2.compareHist(histograms[0], histogram, method)
                    print(name, d)
                print()

    def compute_histograms(self, space = "rgb", hist_size = 8):
        histograms = []
        for image in self.images:
            if space == "rgb":
                hist = cv2.calcHist([image], [0, 1, 2], None, [hist_size, hist_size, hist_size], [0, 256, 0, 256, 0, 256])
            else:
                hist = cv2.calcHist([image], [1, 2], None, [hist_size, hist_size], [0, 256, 0, 256])

            hist = cv2.normalize(hist, hist).flatten() # flatten - make one-dimensional list
            print("hist ", hist)
            histograms.append(hist)
        return histograms

    def set_two_images(self, im1, im2):
        self.img_num1 = im1
        self.img_num2 = im2
        path1 = self.create_image_path(im1)
        path2 = self.create_image_path(im2)
        self.images_paths.append(path1)
        self.images_paths.append(path2)

    def create_image_path(self, num):
        last2 = num[-2:]
        if last2[0] == '0':
            img_path = self.path + 'Images_' + last2[0] + '/' + last2[1] + '/' + num + ".jpg"
        else:
            img_path = self.path + 'Images_' + last2[0] + '/' + last2 + '/' + num + ".jpg"

        return img_path

    def read_grayscale_image(self, num):
        path = self.create_image_path(num)
        img = cv2.imread(path,0) # 0 - grayscale
        return img

    def orbTest(self):
        img1 = self.read_grayscale_image(self.img_num1)
        img2 = self.read_grayscale_image(self.img_num2)
        orb = cv2.ORB_create(nlevels=20)
        kp = orb.detect(img1,None)
        kp1, des1 = orb.compute(img1, kp)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        dist = [m.distance for m in matches]

        print('ORB')
        print('distance: min: %.3f' % min(dist))
        print('distance: mean: %.3f' % (sum(dist) / len(dist)))
        print('distance: max: %.3f' % max(dist))
        print('matches number', len(matches))

        mean = sum(dist) / len(dist)

        good = [m for m in matches if m.distance < mean]
        print('len good', len(good))
        # Sort them in the order of their distance.
        good = sorted(good, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good[:6],None, flags=2)

        return img3, 'orb'

    def akazeTest(self):

        img1 = self.read_grayscale_image(self.img_num1)
        img2 = self.read_grayscale_image(self.img_num2)
        orb = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, descriptor_size=0) # work only with  MLDB
        kp = orb.detect(img1,None)
        kp1, des1 = orb.compute(img1, kp)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        dist = [m.distance for m in matches]

        print('AKAZE')
        print('distance: min: %.3f' % min(dist))
        print('distance: mean: %.3f' % (sum(dist) / len(dist)))
        print('distance: max: %.3f' % max(dist))
        print('matches number', len(matches))

        mean = sum(dist) / len(dist)
        # keep only the reasonable matches
        good = [m for m in matches if m.distance < mean]
        print('len good', len(good))
        good = sorted(good, key = lambda x:x.distance)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good[:6],None, flags=2)

        return img3, 'akaze'


    def briskTest(self):

        img1 = self.read_grayscale_image(self.img_num1)
        img2 = self.read_grayscale_image(self.img_num2)
        orb = cv2.BRISK_create()
        kp = orb.detect(img1,None)
        kp1, des1 = orb.compute(img1, kp)
        kp2, des2 = orb.detectAndCompute(img2, None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        dist = [m.distance for m in matches]

        print('BRISK')
        print('distance: min: %.3f' % min(dist))
        print('distance: mean: %.3f' % (sum(dist) / len(dist)))
        print('distance: max: %.3f' % max(dist))
        print('matches number', len(matches))

        mean = sum(dist) / len(dist)
        # keep only the reasonable matches
        good = [m for m in matches if m.distance < mean]
        print('len good', len(good))
        # Sort them in the order of their distance.
        good = sorted(good, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good[:6], None, flags=2)

        return (img3,'brisk')

    def surfTest(self):
        img1 = self.read_grayscale_image(self.img_num1)
        img2 = self.read_grayscale_image(self.img_num2)
        surf = cv2.xfeatures2d.SURF_create()
        kp = surf.detect(img1,None)
        kp1, des1 = surf.compute(img1, kp)
        kp2, des2 = surf.detectAndCompute(img2, None)
        # create BFMatcher object
        bf = cv2.BFMatcher(crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        dist = [m.distance for m in matches]

        print('SURF')
        print('distance: min: %.3f' % min(dist))
        print('distance: mean: %.3f' % (sum(dist) / len(dist)))
        print('distance: max: %.3f' % max(dist))
        print('matches number', len(matches))

        mean = sum(dist) / len(dist)
        # keep only the reasonable matches
        good = [m for m in matches if m.distance < mean]
        print('len good', len(good))
        # Sort them in the order of their distance.
        good = sorted(good, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good[:6], None, flags=2)
        return img3, 'surf'

    def plot_images(self, img_list, titles):
        size = len(img_list)
        plt.figure()
        for i in range(size):
            plt.subplot(size, 1, i+1)
            plt.title(titles[i])
            plt.imshow(img_list[i])

        plt.show()

    def plot_images_other_windows(self, img_list, titles):
        size = len(img_list)
        for i in range(size):
            plt.figure()
            plt.title(titles[i])
            plt.imshow(img_list[i])

        plt.show()