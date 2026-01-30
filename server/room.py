import os
from livekit import api
import asyncio
os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"

async def main():
    lkapi = api.LiveKitAPI("http://localhost:7880")
    room_info = await lkapi.room.create_room(
        api.CreateRoomRequest(name="my-room"),
    )
    print(room_info)
    results = await lkapi.room.list_rooms(api.ListRoomsRequest())
    print(f"results: {results}")
    await lkapi.aclose()

asyncio.run(main())