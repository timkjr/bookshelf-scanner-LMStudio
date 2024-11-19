import asyncio
import os
import shutil
from typing import AsyncGenerator
import uuid
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ...ai.src.predict import BookPredicter

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

book_predictor = BookPredicter()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> StreamingResponse:
    """
    Predict the title and author of the books in the uploaded image file.
    Args:
        request (Request): The request object.
        file (UploadFile): The uploaded image file.
    Returns:
        StreamingResponse: The predicted titles and authors of the books.
    """
    # Save the uploaded file to a temporary location
    temp_image_path = f"output/temp_{uuid.uuid4()}_{file.filename}"

    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create an async generator for the streaming response
    async def stream_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        client_disconnected = False
        try:
            # Run the predict method in a separate thread to avoid blocking
            book_predictions = await loop.run_in_executor(None, book_predictor.predict, temp_image_path)
            
            for result in book_predictions:
                if await request.is_disconnected():
                    client_disconnected = True
                    break  # Stop processing if client disconnected
                yield f"{result}\n"
                await asyncio.sleep(0)  # Yield control to the event loop
        except Exception as e:
            yield f"Error: {str(e)}\n"
        finally:
            if client_disconnected:
                print("Client disconnected. Cleaning up...")

            # Remove temporary files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    return StreamingResponse(stream_generator(), media_type="text/plain")
