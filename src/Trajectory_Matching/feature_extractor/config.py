from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated

class Reid_config(BaseModel):
    reid_config_path: str = 'src/Trajectory_Matching/feature_extractor/sbs_R50-ibn.yml'
    reid_weight: str = 'src/Trajectory_Matching/feature_extractor/market_sbs_R50-ibn.pth'
    reid_device: str = 'cuda'  # or 'cpu' or dla0

class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_id: str
    input_stream_prefix: str = 'objectdetector'
    output_stream_prefix: str = 'featureextractor'

class TrackletDataBase(BaseModel):
    searching_time: int
    lost_time: int

class FeatureExtrator(BaseSettings):
    frame_info: bool = False
    reid_config:Reid_config
    max_queue_size: int = 64
    sampling_rate: int = 5
    last_frame_id: int = 8999

    model_config = SettingsConfigDict(env_nested_delimiter='__')