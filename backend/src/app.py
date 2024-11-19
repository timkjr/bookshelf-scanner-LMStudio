import asyncio
import os
import shutil
import uuid
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from bookscanner_ai import BookPredictor
from .models import ResultWithData

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

book_predictor = BookPredictor()
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

    try:
        segmented_output, result_generator = await book_predictor.predict(temp_image_path)
    except Exception as e:
        error_result = ResultWithData.fail(str(e))
        return StreamingResponse(
            iter([error_result.model_dump_json(by_alias=True) + "\n"]),
            media_type="application/json"
        )

    async def stream_generator():
        client_disconnected = False
        try:
            # First, send the segmented image
            if segmented_output:
                image_result = ResultWithData[str].succeed(segmented_output)
            else:
                image_result = ResultWithData[str].fail("No books detected.")
            
            yield image_result.model_dump_json(by_alias=True) + "\n"

            # If no books detected, stop processing
            if not segmented_output:
                return

            # Then, send the prediction results
            async for result in result_generator:
                if await request.is_disconnected():
                    client_disconnected = True
                    break

                prediction_result = ResultWithData[str].succeed(result)

                yield prediction_result.model_dump_json(by_alias=True) + "\n"
                await asyncio.sleep(0)
        except Exception as e:
            error_result = ResultWithData.fail(str(e))
            yield error_result.model_dump_json(by_alias=True) + "\n"
        finally:
            if client_disconnected:
                print("Client disconnected. Cleaning up...")

            # Remove temporary files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
            # Clean up output directory
            shutil.rmtree(book_predictor.output_dir, ignore_errors=True)

    return StreamingResponse(stream_generator(), media_type="application/json")
