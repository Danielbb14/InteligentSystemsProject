from furhat_remote_api import FurhatRemoteAPI
furhat = FurhatRemoteAPI("localhost")
available_gestures = furhat.get_gestures()
print(available_gestures.values)
furhat.say(text="Hello, I will now perform a gesture.", blocking=True)
furhat.gesture(name="Smile", blocking=True)