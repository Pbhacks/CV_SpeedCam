import math

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs, each time a new object id detected, the count will increase by one
        self.id_count = 0
        # Store how many frames an object has not been detected
        self.missing_frames = {}

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new objects
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if the object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # Adjust the distance threshold if necessary
                if dist < 50:  # Adjust based on speed and frame rate (e.g., increase if fast-moving vehicles)
                    self.center_points[id] = (cx, cy)
                    self.missing_frames[id] = 0  # Reset the missing frame count
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # If no match found, it's a new object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.missing_frames[self.id_count] = 0  # Track its missing frames
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Update missing frames for all tracked objects
        for id in list(self.center_points.keys()):
            if id not in [obj_bb_id[4] for obj_bb_id in objects_bbs_ids]:
                self.missing_frames[id] += 1

        # Remove objects not seen for more than 10 frames
        self.center_points = {id: pt for id, pt in self.center_points.items() if self.missing_frames[id] <= 10}

        return objects_bbs_ids
