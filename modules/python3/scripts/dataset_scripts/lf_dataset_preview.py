from random import seed, random
import os
import sys
import pickle
from time import sleep, time
import math
from shutil import copyfile

# from random_lf import create_random_lf_cameras
# from random_clip import random_clip_lf, restore_clip
# from random_clip import random_plane_clip
# import welford

import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import vec3, ivec2, normalize

import numpy as np


def cross_product(vec_1, vec_2):
    result = vec3(
        (vec_1.y * vec_2.z) - (vec_1.z * vec_2.y),
        (vec_1.z * vec_2.x) - (vec_1.x * vec_2.z),
        (vec_1.x * vec_2.y) - (vec_1.y * vec_2.x))
    return result


def cam_to_string(cam):
    """Returns some important Inviwo camera properties as a string"""
    cam_string = ("near;{:8f}\n").format(cam.nearPlane)
    cam_string += ("far;{:8f}\n").format(cam.farPlane)
    cam_string += ("focal_length;{:8f}\n".format(cam.projectionMatrix[0][0]))
    cam_string += ("fov;{}").format(cam.fov)
    return cam_string


class LightFieldCamera:
    def __init__(self, look_from, look_to, look_up=vec3(0, 1, 0),
                 interspatial_distance=1.0,
                 spatial_rows=8, spatial_cols=8):
        """
        Create a light field camera array.

        Keyword arguments:
        look_from, look_to, look_up: vectors for top left cam (default up y)
        interspatial_distance: distance between cameras in array (default 1.0)
        spatial_rows, spatial_cols: camera array dimensions (default 8 x 8)
        """
        self.set_look(look_from, look_to, look_up)
        self.spatial_rows = spatial_rows
        self.spatial_cols = spatial_cols
        self.interspatial_distance = interspatial_distance

    def set_look(self, look_from, look_to, look_up):
        """Set the top left camera to look_from, look_to, and look_up"""
        self.look_from = look_from
        self.look_to = look_to
        self.look_up = look_up

    def get_row_col_number(self, index):
        """Get the row and column for an index"""
        row_num = index // self.spatial_cols
        col_num = index % self.spatial_cols
        return row_num, col_num

    def __str__(self):
        lf_string = ("baseline;{}\n").format(self.interspatial_distance)
        lf_string += ("grid_rows;{}\n").format(self.spatial_rows)
        lf_string += ("grid_cols;{}\n").format(self.spatial_rows)
        lf_string += ("look_from;{}\n").format(self.look_from)
        lf_string += ("look_to;{}\n").format(self.look_to)
        lf_string += ("look_up;{}").format(self.look_up)
        return lf_string

    def print_metadata(self, camera, pixel_size, fileout=sys.stdout):
        """
        Prints the metadata about a this grid with a camera

        Keyword arguments:
        camera -- input inviwo camera object
        pixel_size -- output image size (assumed x_dim = y_dim)
        file -- the file to print to (default sys.stdout)
        """
        print(self, end='\n', file=fileout)
        print(cam_to_string(camera), end='\n', file=fileout)
        print("pixels;{}".format(pixel_size), file=fileout)

    def view_array(self, cam, save=False, save_dir=os.path.expanduser('~'),
                   should_time=False):
        """Move the inviwo camera through the array for the current workspace.

        Keyword arguments:
        cam -- the camera to move through the light field array
        save -- save the images to png files (default False)
        save_dir -- the main directory to save the png images to (default home)
        """
        if not os.path.isdir(save_dir):
            raise ValueError("save_dir is not a valid directory.")

        print("Viewing array for lf camera with:")
        print(self)
        # Save the current camera position
        prev_cam_look_from = cam.lookFrom
        prev_cam_look_to = cam.lookTo
        prev_cam_look_up = cam.lookUp

        cam.lookUp = self.look_up

        if should_time:
            times = []

        for idx, val in enumerate(self._calculate_camera_array()):
            (look_from, look_to) = val

            if should_time:
                start_time = time()

            cam.lookFrom = look_from
            cam.lookTo = look_to

            inviwo_utils.update()

            if should_time:
                time_taken = time() - start_time
                times.append(time_taken)

            if save:
                # TODO: Canvas is not necessarily the name of the node we want
                canvas = inviwopy.app.network.Canvas
                row_num, col_num = self.get_row_col_number(idx)

                file_name = str(row_num) + str(col_num) + '.png'
                file_path = os.path.join(os.path.abspath(save_dir), file_name)

                canvas.snapshot(file_path)
                print("Saving : {}".format(file_path))

        # canvas = inviwopy.app.network.canvases[0]
        # pixel_dim = canvas.inputSize.dimensions.value[0]
        # if save:
            # metadata_filename = os.path.join(full_save_dir, 'metadata.csv')
            # with open(metadata_filename, 'w') as f:
                # self.print_metadata(cam, pixel_dim, f)

        # Reset the camera to original position
        if should_time:
            print("Timings : {}".format(times))
            pickle_loc = os.path.join(os.path.abspath(save_dir), 'timing.pkl')
            with open(pickle_loc, 'wb') as time_pkl_file:
                pickle.dump(times, time_pkl_file)

        print()
        cam.lookFrom = prev_cam_look_from
        cam.lookTo = prev_cam_look_to
        cam.lookUp = prev_cam_look_up

    def _get_look_right(self):
        """Get the right look vector for the top left camera"""
        view_direction = self.look_to - self.look_from
        right_vec = normalize(cross_product(view_direction, self.look_up))
        return right_vec

    def _calculate_camera_array(self):
        """Returns list of (look_from, look_to) tuples for the camera array"""
        look_list = []

        row_step_vec = normalize(self.look_up) * self.interspatial_distance
        col_step_vec = self._get_look_right() * self.interspatial_distance

        # Start at the top left camera position
        for i in range(self.spatial_rows):
            row_movement = row_step_vec * (-i)
            row_look_from = self.look_from + row_movement
            row_look_to = self.look_to + row_movement

            for j in range(self.spatial_cols):
                col_movement = col_step_vec * j
                cam_look_from = row_look_from + col_movement
                cam_look_to = row_look_to + col_movement

                look_list.append((cam_look_from, cam_look_to))

        return look_list


def get_random_look_to(scale=10.0):
    look_to_point = np.random.normal(loc=0.0, scale=scale, size=3)
    return vec3(look_to_point[0], look_to_point[1], look_to_point[2])


def get_random_look_from(inner_radius=50.0, outer_radius=100.0):
    radius = inner_radius + (random() * (outer_radius - inner_radius))
    theta = math.pi * random()
    phi = 2 * math.pi * random()

    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    return vec3(x, y, z)


def get_random_look_up(scale=0.25):
    fuzz = np.random.normal(loc=0.0, scale=scale, size=3)

    return vec3(fuzz[0], -1 + fuzz[1], fuzz[2])


def create_random_lf_cameras(num_to_create,
                             interspatial_distance=1.0,
                             spatial_rows=8,
                             spatial_cols=8,
                             inner_radius=50.0,
                             outer_radius=100.0,
                             look_to_scale=10.0,
                             look_up_scale=0.25):
    lf_cameras = []
    for _ in range(num_to_create):
        look_to = get_random_look_to(look_to_scale)
        look_from = get_random_look_from(inner_radius, outer_radius)
        look_up = get_random_look_up(look_up_scale)
        lf_cam = LightFieldCamera(look_from, look_to, look_up,
                                  interspatial_distance,
                                  spatial_rows, spatial_cols)
        lf_cameras.append(lf_cam)

    return lf_cameras


def main(pixel_dim, num_random_lf_samples, save_dir, use_clip, use_plane,
         interspatial_distance=0.5, spatial_rows=8, spatial_cols=8,
         look_from_inner_radius=50.0, look_from_outer_radius=100.0,
         look_to_scale=10.0, look_up_scale=0.25):
    # Setup
    app = inviwopy.app
    network = app.network
    cam = network.EntryExitPoints.camera
    cam.nearPlane = 6.0
    cam.farPlane = 1000.0
    canvases = inviwopy.app.network.canvases
    for canvas in canvases:
        canvas.inputSize.dimensions.value = ivec2(pixel_dim, pixel_dim)
    inviwo_utils.update()

    random_lfs = create_random_lf_cameras(
        num_random_lf_samples,
        interspatial_distance,
        spatial_rows,
        spatial_cols,
        look_from_inner_radius,
        look_from_outer_radius,
        look_to_scale
    )

    # time_accumulator = (0.0, 0.0, 0.0)

    for idx, lf_cam in enumerate(random_lfs):
        sub_dir_name = str(idx).zfill(4)
        sample_dir = os.path.join(save_dir, sub_dir_name)
        os.mkdir(sample_dir)
        # if use_clip:
            # _, clip_type = random_clip_lf(network, lf)
        # elif use_plane:
            # random_plane_clip(network, lf)

        lf_cam.view_array(cam, save=True, save_dir=sample_dir, should_time=True)
        # time_taken = lf.view_array(cam, save=False, should_time=True)
        # time_accumulator = welford.update(
            # time_accumulator, time_taken)
        # if clip:
            # restore_clip(network, clip_type)
    # mean, variance, _ = welford.finalize(time_accumulator)
    # print("Time taken per grid, average {:4f}, std_dev {:4f}".format(
        # mean, math.sqrt(variance)))


if __name__ == '__main__':
    seed(42)
    np.random.seed(42)
    pixel_dim = 256
    num_random_lf_samples = 2000
    save_dir = '/media/tapa/lfsubclavia'
    use_clip = False
    use_plane = False
    interspatial_distance = 1.0
    spatial_rows = 8
    spatial_cols = 8
    look_from_inner_radius = 50.0
    look_from_outer_radius = 250.0
    look_to_scale = 10.0
    look_up_scale = 0.25

    main(pixel_dim, num_random_lf_samples, save_dir, use_clip, use_plane,
         interspatial_distance, spatial_rows, spatial_cols,
         look_from_inner_radius, look_from_outer_radius,
         look_to_scale, look_up_scale)
