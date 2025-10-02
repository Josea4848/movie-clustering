from .extract_color_attr import extract_color_features
from .extract_geo_attr import extract_geo_features
from .extract_tex_attr import extract_texture_features

def extract_all(img_path):
  all_data = dict()
  all_data.update(extract_color_features(img_path))
  #all_data.update(extract_geo_features(img_path))
  all_data.update(extract_texture_features(img_path))
  return all_data

