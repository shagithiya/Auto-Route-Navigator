
import cv2
from streamlit_webrtc import  WebRtcMode,RTCConfiguration, VideoTransformerBase, webrtc_streamer, AudioProcessorBase
from DistanceEstimation import *
from typing import Awaitable, Callable, Generic, List, Optional, TypeVar
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import time

audio_counter = 0
new_audio_file = open('audio.mp3', 'rb')
audio_bytes = new_audio_file.read()
new_audio_file.close() 
st.audio(audio_bytes, format='audio/ogg')   
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

count = st_autorefresh(interval=4500, limit=1000000, key="fizzbuzzcounter")

import av
from tts import *



class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.frame_count = 0


    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        new_img = get_frame_output(img, self.frame_count)
        return new_img

# input - webcam video => transform() => final output arrived(retured to browser)
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        new_image = self.transform(frame)
        
        return av.VideoFrame.from_ndarray(new_image, format="bgr24")

class AudioProcessor(AudioProcessorBase):

    def reset_audio(self):
        time.sleep(0.1)
        self.new_audio_file = open('audio.mp3', 'rb')
        self.audio_bytes = self.new_audio_file.read()
        print(len(self.audio_bytes))
        self.new_audio_file.close() 
        st.audio(self.audio_bytes, format='audio/ogg')
        
        
    async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        get_audio() #tts happens
        self.reset_audio() # for UI updation
        return []

if __name__ == "__main__":
    # webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
    webrtc_streamer(    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": True},
    video_processor_factory=VideoTransformer,
    audio_processor_factory=AudioProcessor,
    )