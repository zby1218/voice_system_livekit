from livekit import api
import os
os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"
# will automatically use the LIVEKIT_API_KEY and LIVEKIT_API_SECRET env vars
token = api.AccessToken() \
    .with_identity("python-bot") \
    .with_name("Python Bot") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room="my-room",
    )).to_jwt()

print(token)