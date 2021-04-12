from data.kitti_util_tracking import *


class Track:
    def __init__(self, maxObject=50, threshold=60):
        self.maxObject = maxObject
        self.threshold = threshold  # the threshold distance of association
        self.trackers = [{}, {}, {}]

    def new_Tracker(self, entity, index):
        id = np.random.randint(0, self.maxObject)
        while id in self.trackers[index].keys():
            id = np.random.randint(0, self.maxObject)
        return {entity['dep']: Tracker(id, dep=entity['dep'], center=entity['center'], dim=entity['dim'],
                                       ort=entity['ort'], cat=index+1)}

    def step(self, output):
        output_ped = [entity for entity in output if entity['cat'] == 1]
        output_car = [entity for entity in output if entity['cat'] == 2]
        output_cyc = [entity for entity in output if entity['cat'] == 3]
        idx = 0
        for out in [output_ped, output_car, output_cyc]:
            if len(out) == 0:
                for tracker in self.trackers[idx].values():
                    tracker.noMatch()
                idx += 1
                continue
            if len(self.trackers[idx]) == 0:
                for j in range(len(out)):
                    self.trackers[idx].update(self.new_Tracker(out[j], idx))
            else:
                pre_deps = list(self.trackers[idx].keys())
                frame_trackers = {entity['dep']: entity for entity in out}
                frame_deps = list(frame_trackers.keys())
                dep_margin_dict = {abs(dep + self.trackers[idx][dep].dep_vec - frame_dep): [dep, frame_dep]
                                   for dep in pre_deps for frame_dep in frame_deps}
                # dep_margin_dict = {abs(dep - frame_dep): [dep, frame_dep] for dep in pre_deps for frame_dep in frame_deps}
                dep_margin = list(dep_margin_dict.keys())
                dep_margin.sort()
                while len(dep_margin) != 0 and dep_margin[0] < self.threshold:
                    dep_pre = dep_margin_dict[dep_margin[0]][0]
                    dep_frame = dep_margin_dict[dep_margin[0]][1]
                    if dep_pre in pre_deps and dep_frame in frame_deps:
                        pre_deps.pop(pre_deps.index(dep_pre))
                        frame_deps.pop(frame_deps.index(dep_frame))
                        self.trackers[idx][dep_pre].update(frame_trackers[dep_frame])
                        self.trackers[idx][dep_frame] = self.trackers[idx].pop(dep_pre)  # update the key
                    dep_margin.pop(0)
                while len(pre_deps) != 0:  # which means there are some pre_deps hasn't been associated
                    self.trackers[idx][pre_deps[0]].noMatch()
                    pre_deps.pop(0)
                while len(frame_deps) != 0:  # which means there are some frame_deps hasn't been associated
                    self.trackers[idx].update(self.new_Tracker(frame_trackers[frame_deps[0]], idx))
                    frame_deps.pop(0)
                dep_pop = []
                for tracker in self.trackers[idx].values():
                    if tracker.life == 0:
                        dep_pop.append(tracker.dep)
                for j in range(len(dep_pop)):
                    self.trackers[idx].pop(dep_pop[j])
                idx += 1

        # if len(self.trackers) == 0:  # init
        #     for i in range(len(output)):
        #         self.trackers.update(self.new_Tracker(output[i]))
        # else:  # associate
        #     pre_deps = list(self.trackers.keys())
        #     frame_trackers = {entity['dep']: entity for entity in output}
        #     frame_deps = [entity['dep'] for entity in output]
        #     dep_margin_dict = {abs(dep + self.trackers[dep].dep_vec - frame_dep): [dep, frame_dep]
        #                        for dep in pre_deps for frame_dep in frame_deps}
        #     # dep_margin_dict = {abs(dep - frame_dep): [dep, frame_dep] for dep in pre_deps for frame_dep in frame_deps}
        #     dep_margin = list(dep_margin_dict.keys())
        #     dep_margin.sort()
        #     while len(dep_margin) != 0 and dep_margin[0] < self.threshold:
        #         dep_pre = dep_margin_dict[dep_margin[0]][0]
        #         dep_frame = dep_margin_dict[dep_margin[0]][1]
        #         if dep_pre in pre_deps and dep_frame in frame_deps:
        #             pre_deps.pop(pre_deps.index(dep_pre))
        #             frame_deps.pop(frame_deps.index(dep_frame))
        #             self.trackers[dep_pre].update(frame_trackers[dep_frame])
        #             self.trackers[dep_frame] = self.trackers.pop(dep_pre)  # update the key
        #         dep_margin.pop(0)
        #     while len(pre_deps) != 0:  # which means there are some pre_deps hasn't been associated
        #         self.trackers[pre_deps[0]].noMatch()
        #         pre_deps.pop(0)
        #     while len(frame_deps) != 0:  # which means there are some frame_deps hasn't been associated
        #         self.trackers.update(self.new_Tracker(frame_trackers[frame_deps[0]]))
        #         frame_deps.pop(0)
        #     dep_pop = []
        #     for tracker in self.trackers.values():
        #         if tracker.life == 0:
        #             dep_pop.append(tracker.dep)
        #     for i in range(len(dep_pop)):
        #         self.trackers.pop(dep_pop[i])

        # pre_deps = list(self.trackers.keys())
        # pre_deps.sort()
        # frame_trackers = {entity['dep']: entity for entity in output}
        # frame_deps = [entity['dep'] for entity in output]
        # frame_deps.sort()
        # # dep_margin = [[abs(dep - frame_dep) for frame_dep in frame_deps] for dep in pre_deps]
        # dep_margin = [[abs(dep - frame_dep) for dep in pre_deps] for frame_dep in frame_deps]
        # idx = 0  # index of tracker under association now
        # num_been_tracked = 0
        # while True:
        #     if num_been_tracked >= len(self.trackers) or len(frame_trackers) == 0 or idx >= len(dep_margin):
        #         # if all pre trackers had been associated(num_been_tracked >= len(self.trackers)
        #         # or all frame trackers had been associated(len(frame_tracker) == 0)
        #         # or remains can't be associated(idx >= len(dep_margin))
        #         break
        #     if min(dep_margin[idx]) > self.threshold:
        #         idx += 1
        #         continue
        #     ind = dep_margin[idx].index(min(dep_margin[idx]))
        #     if ind > idx:
        #         # no match
        #         for i in range(idx, ind):
        #             self.trackers[pre_deps[i]].noMatch()
        #     new_dep = frame_deps[idx]
        #     self.trackers[pre_deps[ind]].update(frame_trackers[frame_deps[idx]])
        #     self.trackers[new_dep] = self.trackers.pop(pre_deps[ind])  # update the key
        #     frame_trackers.pop(frame_deps[idx])
        #     idx += 1
        #     num_been_tracked += 1
        # if len(frame_trackers) != 0:
        #     new_deps = list(frame_trackers.keys())
        #     for i in range(len(frame_trackers)):
        #         self.trackers.update(self.new_Tracker(frame_trackers[new_deps[i]]))
        # dep_pop = []
        # for tracker in self.trackers.values():
        #     if tracker.life == 0:
        #         dep_pop.append(tracker.dep)
        # for i in range(len(dep_pop)):
        #     self.trackers.pop(dep_pop[i])


class Tracker:
    def __init__(self, id, dep, center, dim, ort, cat):
        self.color = None
        self.cat = cat
        self.dep_vec = 0
        self.life = 3
        self.id = id
        self.dep = dep
        self.center = center
        self.dim = dim
        self.ort = ort

    def update(self, entity):
        self.life = 3
        self.dep_vec = entity['dep'] - self.dep
        self.dep = entity['dep']
        self.center = entity['center']
        self.dim = (self.dim + entity['dim']) / 2
        self.ort = entity['ort']

    def noMatch(self):
        self.life -= 1

    def __str__(self):
        return ('id:{}, cat:{}, dep:{}, dim:[{},{},{}]'.format(self.id, self.cat, self.dep, self.dim[0],
                                                               self.dim[1], self.dim[2]))


def compute_box_3d_tracker(tracker, P):
    """ Takes an tracker object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(tracker.ort)

    # 3d bounding box dimensions
    l = tracker.dim[2]
    w = tracker.dim[1]
    h = tracker.dim[0]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + tracker.center[0]
    corners_3d[1, :] = corners_3d[1, :] + tracker.center[1]
    corners_3d[2, :] = corners_3d[2, :] + tracker.center[2]
    # print 'corners_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def draw_projected_box3d_label(image, qs, tracker, thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        adjust: 2021. 4. 8 Yuwei Yan
    """
    if qs is None:
        return image
    qs = qs.astype(np.int32)
    if tracker.color is None:
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        tracker.color = color
    else:
        color = tracker.color
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    # get label
    label = 'id:{}, cat:{}'.format(tracker.id, tracker.cat)
    # font size and style
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get label's height and width
    label_size = cv2.getTextSize(label, font, 0.7, 2)
    # set origin for label
    text_origin = np.array([qs[4][0], qs[4][1] - label_size[0][1]])
    cv2.putText(image, label, (qs[4][0], qs[4][1]), font, 0.7, (0, 0, 0), 2)
    cv2.rectangle(image, tuple(text_origin), tuple(text_origin + label_size[0]),
                  color=color, thickness=2)
    return image
