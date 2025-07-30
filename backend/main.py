# backend/main.py

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import child_monitor.shared_state as shared_state
import cv2
import numpy as np
import uvicorn
from child_monitor.core import ChildMonitor
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure logging to show only warnings and errors
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)


def _log_with_timestamp(message: str):
    """
    Function to log messages with a timestamp
        :param message: Message to log
        :return: None
    """
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp} {message}")


# Define lifespan to handle startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events
        :param app: FastAPI application instance
        :return: None
    """
    yield
    _log_with_timestamp("🔄 Shutdown requested, stopping monitoring...")


app = FastAPI(lifespan=lifespan)
active_connections = []
connections_lock = asyncio.Lock()

# Configure templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Add CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any origin (should be restricted later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast_to_websockets(message):
    """
    Sends a message to all active WebSocket connections
        :param message: Message to be sent
        :return: None
    """
    to_remove = []
    async with connections_lock:
        for ws in active_connections:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            active_connections.remove(ws)

    if to_remove:
        _log_with_timestamp(
            f"📡 Removed {len(to_remove)} invalid WebSocket connections"
        )


async def send_detection_data(persons, zone_coords, alert_active):
    """
    Sends person detection data to the frontend
        :param persons: List of detected persons with coord. and dim.
        :param zone_coords: Coordinates of the monitored zone
        :param alert_active: Boolean indicating if the alert is active
        :return: None
    """
    detection_data = {
        "type": "detection",
        "persons": [
            {
                "x": person[0],
                "y": person[1],
                "width": person[2],
                "height": person[3],
                "in_zone": True,
            }
            for person in persons
        ],
        "zone": (
            {
                "x1": zone_coords[0][0] if zone_coords else 0,
                "y1": zone_coords[0][1] if zone_coords else 0,
                "x2": zone_coords[1][0] if zone_coords else 0,
                "y2": zone_coords[1][1] if zone_coords else 0,
            }
            if zone_coords
            else None
        ),
        "alert_active": alert_active,
        "timestamp": time.time(),
    }
    await broadcast_to_websockets(detection_data)


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Serve the main HTML interface
        :param request: FastAPI request object
        :return: HTMLResponse: Rendered index.html template
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to handle connections
    Allows any origin
        :param websocket: WebSocket connection object
        :return: None
    """
    await websocket.accept()
    async with connections_lock:
        active_connections.append(websocket)

    _log_with_timestamp(
        (
            f"📡 New WebSocket connection established. "
            f"Total: {len(active_connections)}"
        )
    )

    # Keep the connection open and listen for messages
    # This is necessary to prevent the connection from closing immediately
    # and to handle disconnections gracefully
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with connections_lock:
            if websocket in active_connections:
                active_connections.remove(websocket)

        _log_with_timestamp(
            (
                f"📡 WebSocket connection removed. "
                f"Total: {len(active_connections)}"
            )
        )


@app.post("/send_alert")
async def send_alert(alert_data: dict):
    """
    Receives alerts from the core and sends them to WebSocket
        :param alert_data: Alert data to be sent
        :return: dict: Status of the alert sending
    """
    await broadcast_to_websockets(alert_data)
    return {"status": "sent"}


@app.post("/send_detection")
async def send_detection(detection_data: dict):
    """
    Receives detection data from the core and sends it to WebSocket
        :param detection_data: Detection data to be sent
        :return: dict: Status of the detection sending
    """
    await broadcast_to_websockets(detection_data)
    return {"status": "sent"}


@app.get("/health")
async def health():
    """
    Health check endpoint to verify server status
        :return: dict: Health status and active connections
    """
    _log_with_timestamp("🏥 Health check requested")
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "connections": len(active_connections),
    }


@app.post("/shutdown")
async def shutdown():
    """
    Endpoint to gracefully shut down the server
        :return: dict: Status of the shutdown request
    """
    _log_with_timestamp("🛑 Shutdown endpoint called")

    def delayed_shutdown():
        time.sleep(1)
        _log_with_timestamp("💤 Shutting down server...")
        os._exit(0)

    threading.Thread(target=delayed_shutdown, daemon=True).start()
    return {"status": "shutting down"}


@app.get("/video_feed")
async def video_feed():
    """
    Endpoint to stream video frames from the camera
        :return: StreamingResponse: Video stream in multipart format
    """

    def gen_frames():
        while True:
            frame = None
            with shared_state.frame_lock:
                if shared_state.latest_frame is not None:
                    frame = shared_state.latest_frame.copy()

            if frame is not None:
                try:
                    _, buffer = cv2.imencode(".jpg", frame)
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        + b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_bytes
                        + b"\r\n"
                    )
                except Exception as e:
                    _log_with_timestamp(f"❌ Error encoding frame: {e}")
            else:
                # Create placeholder frame when no camera is available
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Add informative text
                placeholder_frame = placeholder_frame.astype(np.uint8)

                cv2.putText(
                    placeholder_frame,
                    "Waiting for camera...",
                    (200, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    placeholder_frame,
                    "Connect a camera to",
                    (170, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    placeholder_frame,
                    "start monitoring",
                    (170, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                try:
                    _, buffer = cv2.imencode(".jpg", placeholder_frame)
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        + b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_bytes
                        + b"\r\n"
                    )
                except Exception as e:
                    _log_with_timestamp(f"❌ Error encoding placeholder: {e}")

            time.sleep(0.1)  # Limit FPS

    media_type = "multipart/x-mixed-replace; boundary=frame"
    return StreamingResponse(gen_frames(), media_type=media_type)


@app.get("/manifest.json")
async def get_manifest():
    return FileResponse("static/manifest.json", media_type="application/json")


@app.get("/sw.js")
async def get_service_worker():
    return FileResponse("sw.js", media_type="application/javascript")


@app.get("/offline.html")
async def get_offline_page():
    return HTMLResponse(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SafeZone - Offline</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h1>🚨 SafeZone - Offline</h1>
        <p>Você está offline. Reconecte-se à internet para usar o sistema.</p>
    </body>
    </html>
    """
    )


if __name__ == "__main__":
    _log_with_timestamp("🚀 Starting Child Monitor Backend...")

    # Initialize monitor
    monitor = ChildMonitor()
    should_exit = threading.Event()

    # Function to run monitoring
    def run_monitor():
        """
        Function to run monitoring loop
        """
        try:
            # First select the zone
            if monitor.select_zone_with_mouse():
                monitor.run_monitoring(should_exit)
        except Exception as e:
            _log_with_timestamp(f"❌ Monitoring error: {e}")
        finally:
            _log_with_timestamp("🛑 Monitoring stopped.")

    # Start monitoring in a background thread
    monitoring_thread = threading.Thread(target=run_monitor, daemon=True)
    monitoring_thread.start()

    # Run FastAPI server with Uvicorn
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="warning",  # Shows only warnings and errors
            access_log=False,  # Disables HTTP access logs
        )
    except KeyboardInterrupt:
        _log_with_timestamp("🔌 Interrupt received, exiting...")
        should_exit.set()
        monitoring_thread.join()
