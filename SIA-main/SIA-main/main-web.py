import streamlit as st
import numpy as np
import cv2
import av
from keras.models import load_model
import mediapipe as mp
from streamlit_webrtc import  WebRtcMode,RTCConfiguration,webrtc_streamer,VideoTransformerBase
st.title("SIA")
tabs_title=["Emotion-detection","          ","Hand-detection","           ","About"]
tabs=st.tabs(tabs_title)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
with tabs[0]:
    mpDraw=mp.solutions.drawing_utils
    mpface=mp.solutions.face_mesh
    facemesh=mpface.FaceMesh(max_num_faces=2)
    drawspec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)
    notice="<p><b style='color:red;'>Instruction :</b>Sit on good place not in dark <br> if any problem occurs refresh the page or mail us at 'hrithikpaul2001@gmail.com' ,<br> Also click 'Select device' for selecting the device <br> Allow to give access </p>"

    h="<p>none</p>"
    st.markdown(notice,unsafe_allow_html=True)
    model=load_model("model_emotion.hdf5")
    class VideoTransformer(VideoTransformerBase):    
        def recv(self, frame):
            value=0
            labels=["happy","sad","angry"]
            image=frame.to_ndarray(format="bgr24")
            if(len(image)>0):
                image=cv2.flip(image,1)
                img=image
                gray1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                results=facemesh.process(gray1)
                if results.multi_face_landmarks:
                    for facelms in results.multi_face_landmarks:
                        #mpDraw.draw_landmarks(image,facelms,mpface.FACEMESH_CONTOURS,drawspec,drawspec)
                        h, w, c = image.shape
                        x_max = 0
                        y_max = 0
                        x_min = w
                        y_min = h
                        for id, lm in enumerate(facelms.landmark):                        
                            cx, cy = int(lm.x *w), int(lm.y*h)                        
                            if cx > x_max:
                                x_max = cx
                            if cx < x_min:
                                x_min = cx
                            if cy > y_max:
                                y_max = cy
                            if cy < y_min:
                                y_min = cy
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        value+=1
                        cv2.putText(image,str(value),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1)
                        hand_value=image[y_min:y_max,x_min:x_max]
                        hand_value=np.array(hand_value)
                        lower=np.array([200,200,200])
                        higher=np.array([255,255,255])
                        mask=cv2.inRange(hand_value,lower,higher)
                        mask=cv2.resize(mask,(100,100))                
                        mask=mask.reshape(1,100,100)
                        p=model.predict(mask)[0]
                        p2=np.argmax(p)
                        p1=labels[np.argmax(p)-1]
                        #confiedence=(p[p2]*100)
                        cv2.putText(image,str(p1),(120,120),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1)
                        #cv2.putText(image,str(confiedence),(30,220),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1)
    
            return av.VideoFrame.from_ndarray(image,format="bgr24")
    webrtc_streamer(key="key0", video_processor_factory=VideoTransformer,rtc_configuration=RTC_CONFIGURATION,async_processing=True,mode=WebRtcMode.SENDRECV)
with tabs[2]:
    notice="<p><b style='color:red;'>Instruction :</b>Sit on good place not in dark <br> if any problem occurs refresh the page or mail us at 'hrithikpaul2001@gmail.com' ,<br> Also click 'Select device' for selecting the device <br> Allow to give access </p>"
    st.markdown(notice,unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        mphands=mp.solutions.hands
        hands=mphands.Hands()
        mpDraw=mp.solutions.drawing_utils
        model=load_model("model_hand2.hdf5")
        class VideoTransformer(VideoTransformerBase):        
            def recv(self, frame):
                value=0
                labels=["a","b","c","d","e","f"]
                image=frame.to_ndarray(format="bgr24")
                if len(image)>0:
                    image=cv2.flip(image,1)
                    gray1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)            
                    results=hands.process(gray1)
                    if results.multi_hand_landmarks:
                        for hand_frame in results.multi_hand_landmarks:
                            h, w, c = image.shape
                            x_max = 0
                            y_max = 0
                            x_min = w
                            y_min = h
                            for id, lm in enumerate(hand_frame.landmark):                            
                                cx, cy = int(lm.x *w), int(lm.y*h)                            
                                if cx > x_max:
                                    x_max = cx
                                if cx < x_min:
                                    x_min = cx
                                if cy > y_max:
                                    y_max = cy
                                if cy < y_min:
                                    y_min = cy
                                cv2.circle(image, (cx,cy),5, (216,216,216),10)
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            value+=1
                            #mpDraw.draw_landmarks(image,hand_frame,mphands.HAND_CONNECTIONS)
                            hand_value=image[y_min:y_max,x_min:x_max]
                            #hand_value=cv2.cvtColor(hand_value,cv2.COLOR_BGR2GRAY)
                            if len(hand_value>0):
                                hand_value=np.array(hand_value)
                                lower=np.array([200,200,200])
                                higher=np.array([255,255,255])
                                mask=cv2.inRange(hand_value,lower,higher)                        
                                mask=cv2.resize(mask,(100,100))
                                mask=np.reshape(mask,(1,100,100))
                                p=model.predict(mask)[0]
                                p1=np.argmax(p)
                                confiedence="{:.2f}%".format(p[p1]*100)
                                cv2.putText(image,labels[p1],(120,120),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0),2)
                                #cv2.putText(image,str(confiedence),(110,110),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0),2)
                            
                return av.VideoFrame.from_ndarray(image,format="bgr24")
        webrtc_streamer(key="key", video_processor_factory=VideoTransformer,rtc_configuration=RTC_CONFIGURATION,async_processing=True,mode=WebRtcMode.SENDRECV)
    with col2:
        
        st.image("sia.png",caption="Hand-sign",width=250)
with tabs[4]:
    st.write("Hey! We are presenting our concept of Hrithik Paul, Sayani Ghatak and Devashis Show which is based on Air Gesture. Our scheme will be useful for all but especially for senior citizens and disabled persons.This platform is completely unique and modern. This smart automation will help them to use computers. No human will be left behind.Stay Stunt…")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header{visibility:hidden;}
            a{
                visibility:hidden;
            }
  
            footer:after{
                visibility:visible;
                content:'Made by Hrithik ,Sayani and Debashis';
                display:block;
                color:red;
                padding:5px;
                top:3px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
