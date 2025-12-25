Metaworld Backend Notes

- Run locally (viewer):
  METAWORLD_PYTHON C:\path\to\metaworld\venv\Scripts\python.exe
  C:\path\to\metaworld\venv\Scripts\python.exe providers\metaworld_server.py --env sawyer_door_v3 --viewer

- Run via main.py (headless server):
  set METAWORLD_PYTHON=C:\path\to\metaworld\venv\Scripts\python.exe
  set METAWORLD_REPO=D:\NLP\Robotics\Simulators_Envs\Metaworld
  python main.py --sim metaworld --task sawyer_door_v3 --depth-format norm_zfar

- Depth formats:
  raw       => meters (Z), clipped to [znear, zfar]
  norm_zfar => Z normalized by zfar
  norm_1m   => clip Z to [0, 1] m (PyBullet-compatible)

- Calibration:
  CAPTURE_IMAGES returns K, znear, zfar, width, height per camera, and depth_encoding="opengl".

