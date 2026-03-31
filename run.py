from .src.Trajectory_Matching.Main import Trajectory_Initialisation

MOT_labels = ""
video_path = ""
video_fps = 15
min_time_in_scene = 3
img_size = (3840, 2160)

runner = Trajectory_Initialisation(MOT_labels, video_path, video_fps=video_fps, min_time_in_scene=min_time_in_scene, img_size=img_size)
average_trajs, zones = runner.run()

# save to picture